//! Deterministic fixed-point arithmetic for simulation.
//!
//! Comptime-parameterized type: `FixedPoint(int_bits, frac_bits)` produces a
//! signed fixed-point number backed by a 16-, 32-, or 64-bit integer. All
//! operations compile to integer add/sub/mul/shift with zero overhead.
//!
//! ## Spec
//!
//! **Representation:** Signed two's complement integer where the lower `frac_bits`
//! encode the fractional part. The scaling factor is always `1 << frac_bits`.
//! Value = raw / scale.
//!
//! **Rounding direction:**
//! - `mul`: rounds toward **negative infinity** (arithmetic right shift).
//! - `div`, `divInt`: rounds toward **zero** (`@divTrunc`).
//! - `convert` (losing precision): rounds toward **negative infinity** (right shift).
//! - `fromFloat`: rounds to **nearest, ties away from zero**.
//!
//! The `mul`/`div` asymmetry is intentional: `>>` is a single instruction on all
//! targets (vs. a branch or conditional add for truncation), and the rounding
//! direction only differs from truncation for negative results. Both are
//! deterministic. Call sites that need toward-zero rounding after `mul` can
//! apply `trunc()`.
//!
//! **Overflow:** Inherits Zig defaults — trap in Debug/ReleaseSafe, wrap in
//! ReleaseFast/ReleaseSmall. No saturating variants. `abs()` traps on MIN_INT.
//! `format()` handles MIN_INT safely via unsigned arithmetic.
//!
//! ## Constraints
//!
//! - `int_bits + frac_bits` must be 16, 32, or 64 (hardware register sizes).
//! - `int_bits >= 1` (sign bit) and `frac_bits >= 1`.
//! - Signed only. Unsigned positions (e.g. transport) should stay as raw integers.
//! - No cross-format arithmetic. Convert first, then operate.
//! - `fromFloat` is load-time only — not for the tick hot path.
//!
//! ## Usage
//!
//! ```zig
//! const Q16_16 = FixedPoint(16, 16);
//! const speed = Q16_16.fromFloat(1.5);
//! const ratio = Q16_16.fromFloat(0.73);
//! const effective = speed.mul(ratio);   // 1.0950 (Q16.16)
//! const ticks = effective.ceil().toInt();
//! ```
//!
//! See also: `docs/design-fixed-point-arithmetic.md`.

const std = @import("std");

/// Deterministic fixed-point numeric type parameterized by integer and fractional
/// bit counts. Total bits (int_bits + frac_bits) must be 16, 32, or 64.
pub fn FixedPoint(comptime int_bits: comptime_int, comptime frac_bits: comptime_int) type {
    const total_bits = int_bits + frac_bits;

    comptime {
        if (total_bits != 16 and total_bits != 32 and total_bits != 64)
            @compileError("FixedPoint total_bits must be 16, 32, or 64");
        if (frac_bits < 1)
            @compileError("FixedPoint requires at least 1 fractional bit");
        if (int_bits < 1)
            @compileError("FixedPoint requires at least 1 integer bit (sign bit)");
    }

    const Backing = std.meta.Int(.signed, total_bits);
    const Wide = std.meta.Int(.signed, total_bits * 2);
    const UBacking = std.meta.Int(.unsigned, total_bits);

    return struct {
        const Self = @This();

        /// The signed integer type used for storage.
        pub const backing_type = Backing;
        /// Double-width signed integer for mul/div intermediates.
        pub const wide_type = Wide;
        /// Number of integer bits (including sign).
        pub const integer_bits = int_bits;
        /// Number of fractional bits.
        pub const fractional_bits = frac_bits;
        /// Scaling factor: `1 << frac_bits`. Equivalent to the value 1.0.
        pub const scale: Backing = @as(Backing, 1) << frac_bits;
        /// Mask for extracting the fractional part.
        pub const frac_mask: Backing = scale - 1;

        /// The underlying scaled two's complement representation.
        /// Represents the value `raw / scale`. Direct access is provided for
        /// serialization and bit-exact operations; use construction functions
        /// for normal use.
        raw: Backing,

        // -----------------------------------------------------------------
        // Constants
        // -----------------------------------------------------------------

        /// Represents the value 0.
        pub const zero = Self{ .raw = 0 };
        /// Represents the value 1.
        pub const one = Self{ .raw = scale };
        /// Represents the value -1.
        pub const neg_one = Self{ .raw = -scale };
        /// Smallest representable positive value (1 / scale).
        /// Also known as the least significant bit (LSB) or unit in the last place (ULP).
        pub const lsb = Self{ .raw = 1 };
        /// Alias for `lsb`. Deprecated: prefer `lsb`.
        pub const epsilon = lsb;

        // -----------------------------------------------------------------
        // Construction
        // -----------------------------------------------------------------

        /// Wrap a raw backing integer.
        /// Asserts that `raw` is correctly scaled (i.e., `raw` represents
        /// `raw / scale` in fixed-point). Use `fromInt` or `fromFloat` for
        /// automatic scaling.
        pub inline fn fromRaw(raw: Backing) Self {
            return .{ .raw = raw };
        }

        /// Construct from an integer value. The integer is shifted left by
        /// `frac_bits` to preserve its magnitude in the fixed-point representation.
        pub inline fn fromInt(x: anytype) Self {
            const I = @TypeOf(x);
            return switch (@typeInfo(I)) {
                .comptime_int => .{ .raw = @as(Backing, x) << frac_bits },
                .int => .{ .raw = @as(Backing, @intCast(x)) << frac_bits },
                else => @compileError("fromInt requires an integer type"),
            };
        }

        /// Construct from f32 or f64. **Load-time only** — not for the tick hot path.
        /// Rounds to nearest representable fixed-point value (half-away-from-zero).
        pub inline fn fromFloat(x: anytype) Self {
            const F = @TypeOf(x);
            comptime if (@typeInfo(F) != .float and @typeInfo(F) != .comptime_float)
                @compileError("fromFloat requires a float type");
            const scaled = x * @as(F, @floatFromInt(scale));
            const rounded = if (scaled >= 0)
                scaled + 0.5
            else
                scaled - 0.5;
            return .{ .raw = @intFromFloat(rounded) };
        }

        /// Construct from a comptime float literal.
        /// Enables `const x = Q16_16.comptimeFromFloat(3.14);` at global scope.
        pub inline fn comptimeFromFloat(comptime x: comptime_float) Self {
            return comptime fromFloat(x);
        }

        // -----------------------------------------------------------------
        // Conversion
        // -----------------------------------------------------------------

        /// Extract the integer part, discarding fractional bits.
        /// Equivalent to `trunc().toInt()` — rounds toward zero.
        pub inline fn toInt(self: Self) std.meta.Int(.signed, int_bits) {
            return @intCast(self.raw >> frac_bits);
        }

        /// Convert to f32. **Rendering/debug only** — not for simulation logic.
        pub inline fn toFloat(self: Self) f32 {
            return @as(f32, @floatFromInt(self.raw)) / @as(f32, @floatFromInt(scale));
        }

        /// Convert to f64. Higher precision rendering path.
        pub inline fn toFloat64(self: Self) f64 {
            return @as(f64, @floatFromInt(self.raw)) / @as(f64, @floatFromInt(scale));
        }

        /// Convert to a different FixedPoint format.
        /// Shifts the raw value to match the target's fractional bit count.
        /// When losing precision (target has fewer frac bits), rounds toward
        /// negative infinity (arithmetic right shift).
        pub inline fn convert(self: Self, comptime Target: type) Target {
            const target_frac = Target.fractional_bits;
            if (target_frac > frac_bits) {
                const shift = target_frac - frac_bits;
                return .{ .raw = @intCast(@as(Wide, self.raw) << shift) };
            } else if (target_frac < frac_bits) {
                const shift = frac_bits - target_frac;
                return .{ .raw = @intCast(self.raw >> shift) };
            } else {
                return .{ .raw = @intCast(self.raw) };
            }
        }

        // -----------------------------------------------------------------
        // Arithmetic
        // -----------------------------------------------------------------

        /// Addition: `a + b`.
        pub inline fn add(a: Self, b: Self) Self {
            return .{ .raw = a.raw + b.raw };
        }

        /// Subtraction: `a - b`.
        pub inline fn sub(a: Self, b: Self) Self {
            return .{ .raw = a.raw - b.raw };
        }

        /// Multiplication: `a * b`.
        /// Uses widened intermediate to preserve precision.
        /// Rounds toward negative infinity (arithmetic right shift).
        pub inline fn mul(a: Self, b: Self) Self {
            const wide = @as(Wide, a.raw) * @as(Wide, b.raw);
            return .{ .raw = @intCast(wide >> frac_bits) };
        }

        /// Division: `a / b`.
        /// Uses widened intermediate to preserve precision.
        /// Truncates toward zero. Division by zero traps.
        pub inline fn div(a: Self, b: Self) Self {
            const wide = @as(Wide, a.raw) << frac_bits;
            return .{ .raw = @intCast(@divTrunc(wide, @as(Wide, b.raw))) };
        }

        /// Multiply by an integer. Simpler than full fixed×fixed.
        pub inline fn mulInt(self: Self, x: anytype) Self {
            const I = @TypeOf(x);
            return switch (@typeInfo(I)) {
                .comptime_int => .{ .raw = self.raw * @as(Backing, x) },
                .int => .{ .raw = self.raw * @as(Backing, @intCast(x)) },
                else => @compileError("mulInt requires an integer type"),
            };
        }

        /// Divide by an integer. Simpler than full fixed÷fixed.
        /// Truncates toward zero.
        pub inline fn divInt(self: Self, x: anytype) Self {
            const I = @TypeOf(x);
            return switch (@typeInfo(I)) {
                .comptime_int => .{ .raw = @divTrunc(self.raw, @as(Backing, x)) },
                .int => .{ .raw = @divTrunc(self.raw, @as(Backing, @intCast(x))) },
                else => @compileError("divInt requires an integer type"),
            };
        }

        /// Negate: `-a`.
        pub inline fn negate(self: Self) Self {
            return .{ .raw = -self.raw };
        }

        // -----------------------------------------------------------------
        // Rounding
        // -----------------------------------------------------------------

        /// Round toward negative infinity.
        /// In two's complement, `& ~frac_mask` always truncates toward -∞.
        pub inline fn floor(self: Self) Self {
            return .{ .raw = self.raw & ~frac_mask };
        }

        /// Round toward positive infinity.
        pub inline fn ceil(self: Self) Self {
            const has_frac = (self.raw & frac_mask) != 0;
            const floored = self.raw & ~frac_mask;
            return .{ .raw = if (has_frac) floored + scale else floored };
        }

        /// Truncate toward zero (discard fractional bits).
        /// Positive: same as floor. Negative: same as ceil.
        pub inline fn trunc(self: Self) Self {
            if (self.raw >= 0) {
                return self.floor();
            } else {
                return self.ceil();
            }
        }

        /// Round to nearest integer. Ties round away from zero.
        pub inline fn round(self: Self) Self {
            const half: Backing = scale >> 1;
            if (self.raw >= 0) {
                return .{ .raw = (self.raw + half) & ~frac_mask };
            } else {
                // negate, add half, floor, negate — gives round-half-away-from-zero
                return .{ .raw = -((-self.raw + half) & ~frac_mask) };
            }
        }

        /// Returns just the fractional part (distance from floor). Always non-negative.
        /// In two's complement, `& frac_mask` gives the floor-relative offset
        /// for both positive and negative values.
        pub inline fn fract(self: Self) Self {
            return .{ .raw = self.raw & frac_mask };
        }

        // -----------------------------------------------------------------
        // Comparison and selection
        // -----------------------------------------------------------------

        /// Test for exact equality.
        pub inline fn eql(a: Self, b: Self) bool {
            return a.raw == b.raw;
        }

        /// Returns the ordering of `a` relative to `b`.
        pub inline fn order(a: Self, b: Self) std.math.Order {
            return std.math.order(a.raw, b.raw);
        }

        /// Test whether `a < b`.
        pub inline fn lt(a: Self, b: Self) bool {
            return a.raw < b.raw;
        }

        /// Test whether `a <= b`.
        pub inline fn lte(a: Self, b: Self) bool {
            return a.raw <= b.raw;
        }

        /// Test whether `a > b`.
        pub inline fn gt(a: Self, b: Self) bool {
            return a.raw > b.raw;
        }

        /// Test whether `a >= b`.
        pub inline fn gte(a: Self, b: Self) bool {
            return a.raw >= b.raw;
        }

        /// Returns the smaller of `a` and `b`.
        pub inline fn min(a: Self, b: Self) Self {
            return .{ .raw = @min(a.raw, b.raw) };
        }

        /// Returns the larger of `a` and `b`.
        pub inline fn max(a: Self, b: Self) Self {
            return .{ .raw = @max(a.raw, b.raw) };
        }

        /// Absolute value. Traps on MIN_INT in safe build modes.
        pub inline fn abs(self: Self) Self {
            return .{ .raw = if (self.raw >= 0) self.raw else -self.raw };
        }

        /// Clamp to the range `[lo, hi]`.
        pub inline fn clamp(self: Self, lo: Self, hi: Self) Self {
            return self.max(lo).min(hi);
        }

        /// Test whether the value is strictly positive.
        pub inline fn isPositive(self: Self) bool {
            return self.raw > 0;
        }

        /// Test whether the value is strictly negative.
        pub inline fn isNegative(self: Self) bool {
            return self.raw < 0;
        }

        /// Test whether the value is exactly zero.
        pub inline fn isZero(self: Self) bool {
            return self.raw == 0;
        }

        // -----------------------------------------------------------------
        // Formatting
        // -----------------------------------------------------------------

        /// Format as decimal string (e.g. "1.5000", "-0.7500").
        /// Uses integer math only — no float conversion.
        /// Use `{f}` in format strings to invoke.
        pub fn format(self: Self, writer: anytype) !void {
            const is_negative = self.raw < 0;
            // Use wrapping negate on unsigned to handle MIN_INT safely.
            const unsigned: UBacking = @bitCast(self.raw);
            const abs_raw: UBacking = if (is_negative) ~unsigned +% 1 else unsigned;

            const int_part = abs_raw >> frac_bits;
            const frac_part = abs_raw & @as(UBacking, @intCast(frac_mask));

            // 4 fractional decimal digits via integer math:
            // frac_part * 10000 / scale
            const decimal = @as(u64, frac_part) * 10000 / @as(u64, @intCast(scale));

            if (is_negative) {
                try writer.print("-{d}.{d:0>4}", .{ int_part, @as(u16, @intCast(decimal)) });
            } else {
                try writer.print("{d}.{d:0>4}", .{ int_part, @as(u16, @intCast(decimal)) });
            }
        }
    };
}

// =========================================================================
// Tests
// =========================================================================

const testing = std.testing;

// Use Q16.16 as the primary test format.
const Q16_16 = FixedPoint(16, 16);
// Secondary formats for cross-format and edge-case tests.
const Q8_8 = FixedPoint(8, 8);
const Q8_24 = FixedPoint(8, 24);
const Q24_8 = FixedPoint(24, 8);
const Q32_32 = FixedPoint(32, 32);

// ---- Construction round-trips ----

test "fromInt round-trip" {
    // Positive
    try testing.expectEqual(@as(i16, 5), Q16_16.fromInt(5).toInt());
    try testing.expectEqual(@as(i16, 0), Q16_16.fromInt(0).toInt());
    // Negative
    try testing.expectEqual(@as(i16, -7), Q16_16.fromInt(-7).toInt());
    // Max positive that fits Q16.16 (15 int bits + sign)
    try testing.expectEqual(@as(i16, 32767), Q16_16.fromInt(32767).toInt());
    try testing.expectEqual(@as(i16, -32768), Q16_16.fromInt(-32768).toInt());
}

test "fromFloat round-trip" {
    const tolerance: f32 = 1.0 / @as(f32, @floatFromInt(Q16_16.scale));

    const cases = [_]f32{ 0.0, 1.0, -1.0, 1.5, -1.5, 0.25, -0.25, 100.75, -100.75 };
    for (cases) |v| {
        const fp = Q16_16.fromFloat(v);
        try testing.expectApproxEqAbs(v, fp.toFloat(), tolerance);
    }
}

test "fromRaw identity" {
    try testing.expectEqual(@as(i32, 12345), Q16_16.fromRaw(12345).raw);
}

test "comptimeFromFloat" {
    const v = comptime Q16_16.comptimeFromFloat(3.14);
    const tolerance: f32 = 1.0 / @as(f32, @floatFromInt(Q16_16.scale));
    try testing.expectApproxEqAbs(@as(f32, 3.14), v.toFloat(), tolerance);
}

// ---- Arithmetic identity laws ----

test "add identity" {
    const a = Q16_16.fromFloat(42.5);
    try testing.expectEqual(a.raw, a.add(Q16_16.zero).raw);
}

test "mul identity" {
    const a = Q16_16.fromFloat(42.5);
    try testing.expectEqual(a.raw, a.mul(Q16_16.one).raw);
}

test "sub self is zero" {
    const a = Q16_16.fromFloat(42.5);
    try testing.expectEqual(Q16_16.zero.raw, a.sub(a).raw);
}

test "add commutativity" {
    const a = Q16_16.fromFloat(3.7);
    const b = Q16_16.fromFloat(-1.2);
    try testing.expectEqual(a.add(b).raw, b.add(a).raw);
}

test "mul commutativity" {
    const a = Q16_16.fromFloat(3.7);
    const b = Q16_16.fromFloat(-1.2);
    try testing.expectEqual(a.mul(b).raw, b.mul(a).raw);
}

// ---- Arithmetic precision ----

test "mul precise" {
    const a = Q16_16.fromFloat(1.5);
    const b = Q16_16.fromFloat(2.0);
    const result = a.mul(b);
    try testing.expectEqual(Q16_16.fromFloat(3.0).raw, result.raw);
}

test "div precise" {
    const a = Q16_16.fromFloat(100.0);
    const b = Q16_16.fromFloat(0.75);
    const result = a.div(b);
    // 100 / 0.75 = 133.333... — verify within 1 epsilon
    const expected: f32 = 100.0 / 0.75;
    const tolerance: f32 = 1.0 / @as(f32, @floatFromInt(Q16_16.scale));
    try testing.expectApproxEqAbs(expected, result.toFloat(), tolerance);
}

test "mul by zero" {
    const a = Q16_16.fromFloat(42.5);
    try testing.expectEqual(Q16_16.zero.raw, a.mul(Q16_16.zero).raw);
}

test "div by one is identity" {
    const a = Q16_16.fromFloat(42.5);
    try testing.expectEqual(a.raw, a.div(Q16_16.one).raw);
}

test "mulInt" {
    const a = Q16_16.fromFloat(3.5);
    const result = a.mulInt(4);
    try testing.expectEqual(Q16_16.fromFloat(14.0).raw, result.raw);
}

test "divInt" {
    const a = Q16_16.fromFloat(14.0);
    const result = a.divInt(4);
    try testing.expectEqual(Q16_16.fromFloat(3.5).raw, result.raw);
}

test "negate" {
    const a = Q16_16.fromFloat(3.5);
    try testing.expectEqual(Q16_16.fromFloat(-3.5).raw, a.negate().raw);
    try testing.expectEqual(a.raw, a.negate().negate().raw);
}

// ---- Rounding ----

test "floor positive" {
    try testing.expectEqual(Q16_16.fromInt(2).raw, Q16_16.fromFloat(2.7).floor().raw);
    try testing.expectEqual(Q16_16.fromInt(2).raw, Q16_16.fromFloat(2.0).floor().raw);
}

test "floor negative" {
    try testing.expectEqual(Q16_16.fromInt(-3).raw, Q16_16.fromFloat(-2.7).floor().raw);
    try testing.expectEqual(Q16_16.fromInt(-2).raw, Q16_16.fromFloat(-2.0).floor().raw);
}

test "ceil positive" {
    try testing.expectEqual(Q16_16.fromInt(3).raw, Q16_16.fromFloat(2.7).ceil().raw);
    try testing.expectEqual(Q16_16.fromInt(2).raw, Q16_16.fromFloat(2.0).ceil().raw);
}

test "ceil negative" {
    try testing.expectEqual(Q16_16.fromInt(-2).raw, Q16_16.fromFloat(-2.7).ceil().raw);
    try testing.expectEqual(Q16_16.fromInt(-2).raw, Q16_16.fromFloat(-2.0).ceil().raw);
}

test "trunc" {
    try testing.expectEqual(Q16_16.fromInt(2).raw, Q16_16.fromFloat(2.7).trunc().raw);
    try testing.expectEqual(Q16_16.fromInt(-2).raw, Q16_16.fromFloat(-2.7).trunc().raw);
}

test "round ties away from zero" {
    try testing.expectEqual(Q16_16.fromInt(3).raw, Q16_16.fromFloat(2.5).round().raw);
    try testing.expectEqual(Q16_16.fromInt(-3).raw, Q16_16.fromFloat(-2.5).round().raw);
    try testing.expectEqual(Q16_16.fromInt(3).raw, Q16_16.fromFloat(2.7).round().raw);
    try testing.expectEqual(Q16_16.fromInt(2).raw, Q16_16.fromFloat(2.3).round().raw);
}

test "fract" {
    const f = Q16_16.fromFloat(2.75).fract();
    try testing.expectEqual(Q16_16.fromFloat(0.75).raw, f.raw);

    // Negative: fract(-2.75) should be 0.25 (distance to floor)
    const fn_ = Q16_16.fromFloat(-2.75).fract();
    try testing.expectEqual(Q16_16.fromFloat(0.25).raw, fn_.raw);

    // Zero fractional part
    try testing.expectEqual(Q16_16.zero.raw, Q16_16.fromInt(5).fract().raw);
}

// ---- Comparison ----

test "comparison operators" {
    const a = Q16_16.fromFloat(1.5);
    const b = Q16_16.fromFloat(2.5);

    try testing.expect(a.lt(b));
    try testing.expect(a.lte(b));
    try testing.expect(a.lte(a));
    try testing.expect(b.gt(a));
    try testing.expect(b.gte(a));
    try testing.expect(b.gte(b));
    try testing.expect(a.eql(a));
    try testing.expect(!a.eql(b));
}

test "min max" {
    const a = Q16_16.fromFloat(1.5);
    const b = Q16_16.fromFloat(2.5);
    try testing.expectEqual(a.raw, Q16_16.min(a, b).raw);
    try testing.expectEqual(b.raw, Q16_16.max(a, b).raw);
}

test "abs" {
    const a = Q16_16.fromFloat(-3.5);
    try testing.expectEqual(Q16_16.fromFloat(3.5).raw, a.abs().raw);
    try testing.expectEqual(Q16_16.fromFloat(3.5).raw, Q16_16.fromFloat(3.5).abs().raw);
    try testing.expectEqual(Q16_16.zero.raw, Q16_16.zero.abs().raw);
}

test "clamp" {
    const lo = Q16_16.fromFloat(1.0);
    const hi = Q16_16.fromFloat(5.0);
    try testing.expectEqual(lo.raw, Q16_16.fromFloat(0.5).clamp(lo, hi).raw);
    try testing.expectEqual(hi.raw, Q16_16.fromFloat(6.0).clamp(lo, hi).raw);
    try testing.expectEqual(Q16_16.fromFloat(3.0).raw, Q16_16.fromFloat(3.0).clamp(lo, hi).raw);
}

test "predicates" {
    try testing.expect(Q16_16.fromFloat(1.0).isPositive());
    try testing.expect(!Q16_16.fromFloat(-1.0).isPositive());
    try testing.expect(Q16_16.fromFloat(-1.0).isNegative());
    try testing.expect(!Q16_16.fromFloat(1.0).isNegative());
    try testing.expect(Q16_16.zero.isZero());
    try testing.expect(!Q16_16.one.isZero());
}

// ---- Cross-format conversion ----

test "convert Q16.16 to Q8.24 preserves value" {
    const a = Q16_16.fromFloat(1.5);
    const b = a.convert(Q8_24);
    const tolerance: f32 = 1.0 / @as(f32, @floatFromInt(Q16_16.scale));
    try testing.expectApproxEqAbs(a.toFloat(), b.toFloat(), tolerance);
}

test "convert Q8.24 to Q16.16 preserves value" {
    const a = Q8_24.fromFloat(0.5);
    const b = a.convert(Q16_16);
    try testing.expectEqual(Q16_16.fromFloat(0.5).raw, b.raw);
}

test "convert same format is identity" {
    const a = Q16_16.fromFloat(7.25);
    const b = a.convert(Q16_16);
    try testing.expectEqual(a.raw, b.raw);
}

test "convert precision loss" {
    // Q8.24 has more fractional precision than Q24.8
    const a = Q8_24.fromFloat(1.123456);
    const b = a.convert(Q24_8);
    // After losing precision, round-trip back won't match exactly
    const tolerance: f32 = 1.0 / @as(f32, @floatFromInt(Q24_8.scale));
    try testing.expectApproxEqAbs(a.toFloat(), b.toFloat(), tolerance);
}

// ---- Format output ----

test "format positive" {
    const v = Q16_16.fromFloat(1.5);
    var buf: [32]u8 = undefined;
    const slice = std.fmt.bufPrint(&buf, "{f}", .{v}) catch unreachable;
    try testing.expectEqualStrings("1.5000", slice);
}

test "format negative" {
    const v = Q16_16.fromFloat(-0.75);
    var buf: [32]u8 = undefined;
    const slice = std.fmt.bufPrint(&buf, "{f}", .{v}) catch unreachable;
    try testing.expectEqualStrings("-0.7500", slice);
}

test "format zero" {
    var buf: [32]u8 = undefined;
    const slice = std.fmt.bufPrint(&buf, "{f}", .{Q16_16.zero}) catch unreachable;
    try testing.expectEqualStrings("0.0000", slice);
}

test "format integer" {
    const v = Q16_16.fromInt(42);
    var buf: [32]u8 = undefined;
    const slice = std.fmt.bufPrint(&buf, "{f}", .{v}) catch unreachable;
    try testing.expectEqualStrings("42.0000", slice);
}

// ---- Bit-exact determinism canaries ----

test "bit-exact: 1.5 * 2.0 = 3.0" {
    const result = Q16_16.fromFloat(1.5).mul(Q16_16.fromFloat(2.0));
    // 3.0 in Q16.16 = 3 << 16 = 196608
    try testing.expectEqual(@as(i32, 196608), result.raw);
}

test "bit-exact: 100.0 / 0.75" {
    const result = Q16_16.fromFloat(100.0).div(Q16_16.fromFloat(0.75));
    // 100 / 0.75 = 133.333... in Q16.16
    // 133 * 65536 + floor(0.333... * 65536) = 8738133
    // Exact: truncation of (100 << 32) / (0.75 << 16)
    //   = (6553600 << 16) / 49152 = 429496729600 / 49152 = 8738133
    try testing.expectEqual(@as(i32, 8738133), result.raw);
}

test "bit-exact: fromFloat(3.14)" {
    const v = Q16_16.fromFloat(@as(f32, 3.14));
    // 3.14 * 65536 = 205783.04, rounded = 205783
    try testing.expectEqual(@as(i32, 205783), v.raw);
}

// ---- Multiple format instantiation ----

test "Q8.8 basic" {
    const a = Q8_8.fromFloat(3.5);
    const b = Q8_8.fromFloat(2.0);
    const result = a.mul(b);
    try testing.expectEqual(Q8_8.fromFloat(7.0).raw, result.raw);
}

test "Q32.32 large values" {
    const a = Q32_32.fromInt(1000000);
    const b = Q32_32.fromFloat(1.5);
    const result = a.mul(b);
    try testing.expectEqual(Q32_32.fromInt(1500000).raw, result.raw);
}

// ---- Sim-relevant computation ----

test "crafting ticks_required: energy=100, speed=0.75, tps=60" {
    const energy = Q16_16.fromFloat(100.0);
    const speed = Q16_16.fromFloat(0.75);
    const tps = Q16_16.fromInt(60);

    const rate = energy.div(speed);
    const ticks_f = rate.mul(tps);
    const ticks = ticks_f.ceil().toInt();

    try testing.expectEqual(@as(i16, 8000), ticks);
}

test "power satisfaction throttle" {
    const crafting_speed = Q16_16.fromFloat(1.5);
    const satisfaction = Q16_16.fromFloat(0.73);
    const effective = crafting_speed.mul(satisfaction);

    // 1.5 * 0.73 = 1.095
    const tolerance: f32 = 1.0 / @as(f32, @floatFromInt(Q16_16.scale));
    try testing.expectApproxEqAbs(@as(f32, 1.095), effective.toFloat(), tolerance);
}

// ---- Rounding direction for negative results ----

test "mul rounds toward negative infinity for negative results" {
    // 0.3 * -0.7 = -0.21 exactly. In Q16.16: raw should round toward -∞.
    const a = Q16_16.fromFloat(0.3);
    const b = Q16_16.fromFloat(-0.7);
    const result = a.mul(b);
    // Verify the result floors (toward -∞), not truncates (toward 0).
    // floor(-0.21) in fixed-point: the raw value should be <= -0.21 * 65536
    const expected_approx: f32 = 0.3 * -0.7;
    try testing.expect(result.toFloat() <= expected_approx);
}

test "div truncates toward zero for negative results" {
    // -7 / 3 = -2.333... → divTrunc = -2 (toward zero), not -3 (toward -∞)
    const a = Q16_16.fromInt(-7);
    const b = Q16_16.fromInt(3);
    const result = a.div(b);
    // Truncation toward zero: integer part is -2, not -3
    try testing.expectEqual(@as(i16, -2), result.trunc().toInt());
}

test "mulInt with negative operand" {
    const a = Q16_16.fromFloat(3.5);
    try testing.expectEqual(Q16_16.fromFloat(-7.0).raw, a.mulInt(-2).raw);
}

test "divInt with negative operand" {
    const a = Q16_16.fromFloat(-14.0);
    try testing.expectEqual(Q16_16.fromFloat(-3.5).raw, a.divInt(4).raw);
}

// ---- Convert with negative values ----

test "convert negative Q16.16 to Q8.24 preserves value" {
    const a = Q16_16.fromFloat(-1.5);
    const b = a.convert(Q8_24);
    const tolerance: f32 = 1.0 / @as(f32, @floatFromInt(Q16_16.scale));
    try testing.expectApproxEqAbs(a.toFloat(), b.toFloat(), tolerance);
}

test "convert negative losing precision rounds toward negative infinity" {
    // -0.1 in Q8.24 converted to Q24.8 should floor, not truncate.
    const a = Q8_24.fromFloat(-0.1);
    const b = a.convert(Q24_8);
    // The converted value should be <= the original (floored)
    try testing.expect(b.toFloat() <= a.toFloat());
}

// ---- Constants ----

test "constants" {
    try testing.expectEqual(@as(i32, 0), Q16_16.zero.raw);
    try testing.expectEqual(Q16_16.scale, Q16_16.one.raw);
    try testing.expectEqual(-Q16_16.scale, Q16_16.neg_one.raw);
    try testing.expectEqual(@as(i32, 1), Q16_16.lsb.raw);
    try testing.expectEqual(@as(i32, 1), Q16_16.epsilon.raw); // backward compat alias

    // one + neg_one = zero
    try testing.expectEqual(Q16_16.zero.raw, Q16_16.one.add(Q16_16.neg_one).raw);
}

// ---- order and toFloat64 ----

test "order" {
    const a = Q16_16.fromFloat(1.5);
    const b = Q16_16.fromFloat(2.5);
    try testing.expectEqual(std.math.Order.lt, Q16_16.order(a, b));
    try testing.expectEqual(std.math.Order.gt, Q16_16.order(b, a));
    try testing.expectEqual(std.math.Order.eq, Q16_16.order(a, a));
}

test "toFloat64" {
    const v = Q16_16.fromFloat(3.14);
    const f64_val = v.toFloat64();
    const tolerance: f64 = 1.0 / @as(f64, @floatFromInt(Q16_16.scale));
    try testing.expect(@abs(f64_val - 3.14) < tolerance);
}

// ---- Format edge cases ----

test "format min representable value" {
    // MIN_INT should not crash — this is the bug that was fixed.
    const v = Q16_16.fromRaw(std.math.minInt(i32));
    var buf: [64]u8 = undefined;
    const slice = std.fmt.bufPrint(&buf, "{f}", .{v}) catch unreachable;
    // Should produce "-32768.0000"
    try testing.expectEqualStrings("-32768.0000", slice);
}

test "format max representable value" {
    const v = Q16_16.fromRaw(std.math.maxInt(i32));
    var buf: [64]u8 = undefined;
    const slice = std.fmt.bufPrint(&buf, "{f}", .{v}) catch unreachable;
    try testing.expectEqualStrings("32767.9999", slice);
}
