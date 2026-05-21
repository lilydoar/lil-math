//! Fixed-point trigonometry using lookup tables and analytical methods.
//!
//! All angles are in **turns** (1.0 = 360°), not radians. This makes wrapping
//! free for FixedPoint — just mask off the integer bits. No need to represent
//! π or 2π in fixed-point.
//!
//! Functions are generic over `FixedPoint(I, F)` configurations, not hardcoded
//! to a specific format.
//!
//! ## Lookup Table Strategy
//!
//! - sin() uses a 1024-entry comptime-generated table covering [0, 0.25) turns
//!   (one quadrant). Other quadrants are derived via symmetry.
//! - Linear interpolation between table entries.
//! - cos(x) = sin(x + 0.25 turns).
//! - Table generation uses Zig's comptime float math; all runtime lookups are
//!   fixed-point integer arithmetic only.
//!
//! ## CORDIC for atan2
//!
//! CORDIC (Coordinate Rotation Digital Computer) iteratively rotates a vector
//! toward the x-axis, accumulating the rotation angle. Converges in ~16
//! iterations for typical fixed-point precision.
//!
//! ## Newton-Raphson for sqrt
//!
//! Standard iterative method: x_{n+1} = (x_n + N/x_n) / 2. Initial guess from
//! leading zeros. Converges in 4-5 iterations for 16.16 fixed-point.

const std = @import("std");
const FixedPoint = @import("fixed_point.zig").FixedPoint;

// -------------------------------------------------------------------------
// Sine/Cosine Lookup Tables
// -------------------------------------------------------------------------

/// Number of entries in the sine lookup table (one quadrant).
const sine_table_size: comptime_int = 1024;

/// Comptime-generated sine lookup table for [0, 0.25) turns (0° to 90°).
/// Values are in the range [0, 1]. Type is the fixed-point format to use.
fn SineTable(comptime Fp: type) type {
    return struct {
        const Table = @This();

        /// Precomputed sine values for one quadrant.
        /// table[i] = sin(i / sine_table_size * 0.25 turns)
        ///          = sin(i / sine_table_size * π/2 radians)
        const values: [sine_table_size]Fp = blk: {
            @setEvalBranchQuota(10000);
            var result: [sine_table_size]Fp = undefined;
            var i: usize = 0;
            while (i < sine_table_size) : (i += 1) {
                // Convert table index to radians: i/1024 * π/2
                const angle_radians = (@as(f64, @floatFromInt(i)) / @as(f64, sine_table_size)) * (std.math.pi / 2.0);
                const sine_value = @sin(angle_radians);
                result[i] = Fp.fromFloat(sine_value);
            }
            break :blk result;
        };

        /// Look up sine value with linear interpolation.
        /// Input: fractional_turns in [0, 0.25) as a fixed-point value.
        /// Output: sine value in [0, 1].
        fn lookup(fractional_turns: Fp) Fp {
            // Scale fractional_turns to table index range [0, sine_table_size)
            // fractional_turns is in [0, 0.25), multiply by 4 to get [0, 1)
            // then multiply by sine_table_size to get index.
            // To avoid overflow, we do this in steps: multiply by 4, then by sine_table_size
            // But use the wide type to avoid overflow.
            const scaled_wide = @as(Fp.wide_type, fractional_turns.raw) * @as(Fp.wide_type, 4 * sine_table_size);

            // Extract integer part (table index) and fractional part (for interpolation).
            const index = @as(usize, @intCast((scaled_wide >> Fp.fractional_bits) & 0xFFFF));
            const frac_raw = @as(Fp.backing_type, @intCast(scaled_wide & Fp.frac_mask));
            const frac = Fp.fromRaw(frac_raw);

            // Clamp index to valid range (should not be needed but for safety)
            const idx0 = @min(index, sine_table_size - 1);
            const idx1 = if (idx0 >= sine_table_size - 1) sine_table_size - 1 else idx0 + 1;

            const v0 = values[idx0];
            const v1 = values[idx1];

            // Linear interpolation: v0 + (v1 - v0) * frac
            return v0.add(v1.sub(v0).mul(frac));
        }
    };
}

/// Compute sine using lookup table and quadrant symmetry.
/// Input: angle in turns (1.0 = 360°).
/// Output: sine value in [-1, 1].
///
/// Generic over any FixedPoint(I, F) configuration.
pub fn sin(angle: anytype) @TypeOf(angle) {
    const Fp = @TypeOf(angle);

    // Wrap angle to [0, 1) turns by masking off integer bits.
    // For signed fixed-point, we need to handle negative angles.
    var normalized = angle;
    // Make angle positive by adding enough full turns
    while (normalized.raw < 0) {
        normalized = normalized.add(Fp.fromInt(1));
    }
    // Mask to [0, 1) range by zeroing out integer bits
    const int_mask = ~@as(Fp.backing_type, 0) << Fp.fractional_bits;
    normalized.raw = normalized.raw & ~int_mask;

    // Now normalized is in [0, 1) turns.
    // Use quadrant symmetry to reduce to [0, 0.25) turns.
    const quarter = Fp.fromFloat(0.25);
    const half = Fp.fromFloat(0.5);
    const three_quarter = Fp.fromFloat(0.75);

    const table = SineTable(Fp);

    if (normalized.lt(quarter)) {
        // First quadrant: [0, 0.25) → [0, 1]
        return table.lookup(normalized);
    } else if (normalized.lt(half)) {
        // Second quadrant: [0.25, 0.5) → [1, 0]
        // sin(0.25 + x) = sin(0.25 - x) = cos(x)
        const reflected = half.sub(normalized);
        return table.lookup(reflected);
    } else if (normalized.lt(three_quarter)) {
        // Third quadrant: [0.5, 0.75) → [0, -1]
        // sin(0.5 + x) = -sin(x)
        const reflected = normalized.sub(half);
        return table.lookup(reflected).negate();
    } else {
        // Fourth quadrant: [0.75, 1.0) → [-1, 0]
        // sin(0.75 + x) = -sin(0.25 - x)
        const reflected = Fp.fromInt(1).sub(normalized);
        return table.lookup(reflected).negate();
    }
}

/// Compute cosine using sine.
/// cos(x) = sin(x + 0.25 turns) = sin(x + 90°).
pub fn cos(angle: anytype) @TypeOf(angle) {
    const Fp = @TypeOf(angle);
    const quarter = Fp.fromFloat(0.25);
    return sin(angle.add(quarter));
}

// -------------------------------------------------------------------------
// CORDIC atan2
// -------------------------------------------------------------------------

/// CORDIC arctangent lookup table (angles in turns).
/// Each entry is atan(2^-i) in turns.
fn AtanTable(comptime Fp: type) type {
    return struct {
        /// Precomputed atan(2^-i) values in turns for i = 0..31.
        const values: [32]Fp = blk: {
            var result: [32]Fp = undefined;
            var i: usize = 0;
            while (i < 32) : (i += 1) {
                const power = @as(f64, 1.0) / @as(f64, @floatFromInt(@as(u64, 1) << @intCast(i)));
                const atan_radians = std.math.atan(power);
                // Convert to turns: radians / (2π)
                const atan_turns = atan_radians / (2.0 * std.math.pi);
                result[i] = Fp.fromFloat(atan_turns);
            }
            break :blk result;
        };
    };
}

/// CORDIC gain constant (approximately 1.646760258).
/// After CORDIC rotation, vectors are scaled by this factor.
fn cordicGain(comptime Fp: type) Fp {
    // K = prod(sqrt(1 + 2^(-2i))) for i = 0..31
    // Approximation: 1.646760258121
    return Fp.fromFloat(1.646760258121);
}

/// Four-quadrant arctangent using CORDIC algorithm.
/// Returns angle in turns [0, 1) (or [-0.5, 0.5) depending on normalization).
/// Handles all quadrants correctly.
///
/// Special cases:
/// - atan2(0, 0) returns 0 (arbitrary but deterministic).
/// - atan2(y, 0) returns 0.25 or 0.75 depending on sign of y.
pub fn atan2(y: anytype, x: anytype) @TypeOf(y) {
    const Fp = @TypeOf(y);

    // Handle zero cases
    if (x.isZero() and y.isZero()) return Fp.zero;
    if (x.isZero()) {
        return if (y.isPositive()) Fp.fromFloat(0.25) else Fp.fromFloat(0.75);
    }
    if (y.isZero()) {
        return if (x.isPositive()) Fp.zero else Fp.fromFloat(0.5);
    }

    const table = AtanTable(Fp);

    // CORDIC works in the range [-π/2, π/2] ([-0.25, 0.25] turns).
    // Reduce to this range and track quadrant offset.
    var angle_offset: Fp = Fp.zero;
    var x_work = x;
    var y_work = y;

    // Move to right half-plane if necessary
    if (x_work.isNegative()) {
        x_work = x_work.negate();
        y_work = y_work.negate();
        angle_offset = Fp.fromFloat(0.5); // 180°
    }

    // Now x_work > 0. CORDIC will find angle in [-0.25, 0.25] turns.
    var angle = Fp.zero;

    // CORDIC iterations
    var i: usize = 0;
    while (i < 16) : (i += 1) {
        if (y_work.isNegative()) {
            // Rotate clockwise
            const y_shifted = Fp.fromRaw(y_work.raw >> @intCast(i));
            const x_shifted = Fp.fromRaw(x_work.raw >> @intCast(i));
            const new_x = x_work.sub(y_shifted);
            const new_y = y_work.add(x_shifted);
            x_work = new_x;
            y_work = new_y;
            angle = angle.sub(table.values[i]);
        } else {
            // Rotate counter-clockwise
            const y_shifted = Fp.fromRaw(y_work.raw >> @intCast(i));
            const x_shifted = Fp.fromRaw(x_work.raw >> @intCast(i));
            const new_x = x_work.add(y_shifted);
            const new_y = y_work.sub(x_shifted);
            x_work = new_x;
            y_work = new_y;
            angle = angle.add(table.values[i]);
        }
    }

    // Combine with quadrant offset and wrap to [0, 1)
    var result = angle.add(angle_offset);

    // Normalize to [0, 1) turns
    while (result.raw < 0) {
        result = result.add(Fp.fromInt(1));
    }
    const int_mask = ~@as(Fp.backing_type, 0) << Fp.fractional_bits;
    result.raw = result.raw & ~int_mask;

    return result;
}

// -------------------------------------------------------------------------
// Newton-Raphson sqrt
// -------------------------------------------------------------------------

/// Square root via Newton-Raphson iteration.
/// Input: non-negative fixed-point value.
/// Output: square root.
///
/// Returns zero for negative inputs (undefined behavior).
pub fn sqrt(x: anytype) @TypeOf(x) {
    const Fp = @TypeOf(x);

    if (x.lte(Fp.zero)) return Fp.zero;
    if (x.eql(Fp.one)) return Fp.one;

    // Initial guess using leading zeros.
    // For a value with n bits, sqrt has approximately n/2 bits.
    // We can estimate by shifting right by half the effective bit position.
    var guess = initialSqrtGuess(Fp, x);

    // Newton-Raphson: x_{n+1} = (x_n + N/x_n) / 2
    // Iterate until convergence (typically 4-6 iterations).
    var i: usize = 0;
    while (i < 8) : (i += 1) {
        const quotient = x.div(guess);
        const next_guess = guess.add(quotient).divInt(2);

        // Check for convergence (difference < 1 ULP)
        const diff = if (next_guess.gt(guess))
            next_guess.sub(guess)
        else
            guess.sub(next_guess);

        if (diff.lte(Fp.lsb)) break;

        guess = next_guess;
    }

    return guess;
}

/// Compute initial guess for square root using bit position.
fn initialSqrtGuess(comptime Fp: type, x: Fp) Fp {
    // Count leading zeros to find the bit position of the most significant bit.
    const leading_zeros = @clz(x.raw);
    const total_bits = @bitSizeOf(Fp.backing_type);
    const significant_bits = total_bits - leading_zeros;

    // For sqrt, the result has roughly half the bits.
    // Shift right by half the bit position.
    const shift = @max(1, significant_bits / 2);

    // Create initial guess by shifting
    const guess_raw = @max(Fp.scale, x.raw >> @intCast(shift));

    return Fp.fromRaw(guess_raw);
}

// =========================================================================
// Tests
// =========================================================================

const testing = std.testing;

const Q16_16 = FixedPoint(16, 16);
const Q8_24 = FixedPoint(8, 24);

// ---- sin/cos accuracy ----

test "sin accuracy vs @sin" {
    // Test 1000 evenly spaced angles in [0, 1) turns
    var i: usize = 0;
    const tolerance = Q16_16.fromFloat(1e-4);

    while (i < 1000) : (i += 1) {
        const angle_turns = @as(f32, @floatFromInt(i)) / 1000.0;
        const angle = Q16_16.fromFloat(angle_turns);

        const result = sin(angle);
        const expected = @sin(angle_turns * 2.0 * std.math.pi);
        const expected_fp = Q16_16.fromFloat(expected);

        const error_fp = result.sub(expected_fp).abs();

        if (error_fp.gt(tolerance)) {
            std.debug.print("\nsin({d} turns) = {f}, expected {f}, error {f}\n", .{
                angle_turns, result, expected_fp, error_fp,
            });
        }
        try testing.expect(error_fp.lte(tolerance));
    }
}

test "cos accuracy vs @cos" {
    var i: usize = 0;
    const tolerance = Q16_16.fromFloat(1e-4);

    while (i < 1000) : (i += 1) {
        const angle_turns = @as(f32, @floatFromInt(i)) / 1000.0;
        const angle = Q16_16.fromFloat(angle_turns);

        const result = cos(angle);
        const expected = @cos(angle_turns * 2.0 * std.math.pi);
        const expected_fp = Q16_16.fromFloat(expected);

        const error_fp = result.sub(expected_fp).abs();
        try testing.expect(error_fp.lte(tolerance));
    }
}

// ---- sin²+cos² = 1 identity ----

test "sin² + cos² ≈ 1" {
    var i: usize = 0;
    const tolerance = Q16_16.fromFloat(1e-4);
    const one = Q16_16.one;

    while (i < 100) : (i += 1) {
        const angle_turns = @as(f32, @floatFromInt(i)) / 100.0;
        const angle = Q16_16.fromFloat(angle_turns);

        const s = sin(angle);
        const c = cos(angle);
        const sum = s.mul(s).add(c.mul(c));

        const error_fp = sum.sub(one).abs();
        try testing.expect(error_fp.lte(tolerance));
    }
}

// ---- Edge cases ----

test "sin/cos edge cases" {
    const tolerance = Q16_16.fromFloat(1e-4);

    // sin(0) = 0
    try testing.expect(sin(Q16_16.zero).abs().lte(tolerance));

    // sin(0.25 turns) = sin(90°) = 1
    const sin_90 = sin(Q16_16.fromFloat(0.25));
    try testing.expect(sin_90.sub(Q16_16.one).abs().lte(tolerance));

    // sin(0.5 turns) = sin(180°) = 0
    const sin_180 = sin(Q16_16.fromFloat(0.5));
    try testing.expect(sin_180.abs().lte(tolerance));

    // sin(0.75 turns) = sin(270°) = -1
    const sin_270 = sin(Q16_16.fromFloat(0.75));
    try testing.expect(sin_270.add(Q16_16.one).abs().lte(tolerance));

    // cos(0) = 1
    const cos_0 = cos(Q16_16.zero);
    try testing.expect(cos_0.sub(Q16_16.one).abs().lte(tolerance));

    // cos(0.25 turns) = cos(90°) = 0
    const cos_90 = cos(Q16_16.fromFloat(0.25));
    try testing.expect(cos_90.abs().lte(tolerance));

    // cos(0.5 turns) = cos(180°) = -1
    const cos_180 = cos(Q16_16.fromFloat(0.5));
    try testing.expect(cos_180.add(Q16_16.one).abs().lte(tolerance));
}

// ---- Wrapping ----

test "sin wrapping: sin(1.5 turns) == sin(0.5 turns)" {
    const angle1 = Q16_16.fromFloat(0.5);
    const angle2 = Q16_16.fromFloat(1.5);

    const s1 = sin(angle1);
    const s2 = sin(angle2);

    const tolerance = Q16_16.fromFloat(1e-4);
    try testing.expect(s1.sub(s2).abs().lte(tolerance));
}

test "sin wrapping: sin(-0.25 turns) == sin(0.75 turns)" {
    const angle1 = Q16_16.fromFloat(-0.25);
    const angle2 = Q16_16.fromFloat(0.75);

    const s1 = sin(angle1);
    const s2 = sin(angle2);

    const tolerance = Q16_16.fromFloat(1e-4);
    try testing.expect(s1.sub(s2).abs().lte(tolerance));
}

// ---- atan2 quadrant correctness ----

test "atan2 quadrants" {
    const tolerance = Q16_16.fromFloat(1e-3);

    // Quadrant I: (1, 1) → ~0.125 turns (45°)
    const q1 = atan2(Q16_16.one, Q16_16.one);
    const expected_q1 = Q16_16.fromFloat(0.125);
    try testing.expect(q1.sub(expected_q1).abs().lte(tolerance));

    // Quadrant II: (1, -1) → ~0.375 turns (135°)
    const q2 = atan2(Q16_16.one, Q16_16.one.negate());
    const expected_q2 = Q16_16.fromFloat(0.375);
    try testing.expect(q2.sub(expected_q2).abs().lte(tolerance));

    // Quadrant III: (-1, -1) → ~0.625 turns (225°)
    const q3 = atan2(Q16_16.one.negate(), Q16_16.one.negate());
    const expected_q3 = Q16_16.fromFloat(0.625);
    try testing.expect(q3.sub(expected_q3).abs().lte(tolerance));

    // Quadrant IV: (-1, 1) → ~0.875 turns (315°)
    const q4 = atan2(Q16_16.one.negate(), Q16_16.one);
    const expected_q4 = Q16_16.fromFloat(0.875);
    try testing.expect(q4.sub(expected_q4).abs().lte(tolerance));
}

test "atan2 axis cases" {
    const tolerance = Q16_16.fromFloat(1e-4);

    // Positive x-axis: (0, 1) → 0
    const pos_x = atan2(Q16_16.zero, Q16_16.one);
    try testing.expect(pos_x.abs().lte(tolerance));

    // Negative x-axis: (0, -1) → 0.5 turns
    const neg_x = atan2(Q16_16.zero, Q16_16.one.negate());
    const expected_neg_x = Q16_16.fromFloat(0.5);
    try testing.expect(neg_x.sub(expected_neg_x).abs().lte(tolerance));

    // Positive y-axis: (1, 0) → 0.25 turns
    const pos_y = atan2(Q16_16.one, Q16_16.zero);
    const expected_pos_y = Q16_16.fromFloat(0.25);
    try testing.expect(pos_y.sub(expected_pos_y).abs().lte(tolerance));

    // Negative y-axis: (-1, 0) → 0.75 turns
    const neg_y = atan2(Q16_16.one.negate(), Q16_16.zero);
    const expected_neg_y = Q16_16.fromFloat(0.75);
    try testing.expect(neg_y.sub(expected_neg_y).abs().lte(tolerance));
}

test "atan2 origin" {
    // atan2(0, 0) is defined to return 0
    const result = atan2(Q16_16.zero, Q16_16.zero);
    try testing.expectEqual(Q16_16.zero.raw, result.raw);
}

// ---- sqrt accuracy ----

test "sqrt accuracy" {
    const test_values = [_]f32{ 0.0, 1.0, 2.0, 4.0, 9.0, 16.0, 25.0, 100.0, 0.25, 0.5 };

    for (test_values) |v| {
        const x = Q16_16.fromFloat(v);
        const result = sqrt(x);
        const expected = @sqrt(v);
        const expected_fp = Q16_16.fromFloat(expected);

        const tolerance = Q16_16.fromFloat(1e-3);
        const error_fp = result.sub(expected_fp).abs();

        if (error_fp.gt(tolerance)) {
            std.debug.print("\nsqrt({d}) = {f}, expected {f}, error {f}\n", .{
                v, result, expected_fp, error_fp,
            });
        }
        try testing.expect(error_fp.lte(tolerance));
    }
}

test "sqrt edge cases" {
    // sqrt(0) = 0
    try testing.expectEqual(Q16_16.zero.raw, sqrt(Q16_16.zero).raw);

    // sqrt(1) = 1
    try testing.expectEqual(Q16_16.one.raw, sqrt(Q16_16.one).raw);

    // sqrt(negative) = 0 (defined behavior)
    const neg_result = sqrt(Q16_16.fromInt(-1));
    try testing.expectEqual(Q16_16.zero.raw, neg_result.raw);
}

// ---- Generic over FixedPoint parameters ----

test "sin works with Q8.24" {
    const angle = Q8_24.fromFloat(0.25); // 90°
    const result = sin(angle);
    const tolerance = Q8_24.fromFloat(1e-4);
    try testing.expect(result.sub(Q8_24.one).abs().lte(tolerance));
}

test "atan2 works with Q8.24" {
    const angle = atan2(Q8_24.one, Q8_24.one);
    const expected = Q8_24.fromFloat(0.125); // 45°
    const tolerance = Q8_24.fromFloat(1e-3);
    try testing.expect(angle.sub(expected).abs().lte(tolerance));
}

test "sqrt works with Q8.24" {
    const x = Q8_24.fromFloat(4.0);
    const result = sqrt(x);
    const expected = Q8_24.fromFloat(2.0);
    const tolerance = Q8_24.fromFloat(1e-3);
    try testing.expect(result.sub(expected).abs().lte(tolerance));
}
