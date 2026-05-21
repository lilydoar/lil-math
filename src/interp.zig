//! Animated value interpolation with pluggable easing.
//!
//! The central abstraction is `InterpType(T, blendFn)` — a stateful wrapper
//! around a value of type `T` that smoothly transitions between a start and
//! end value over time, applying an optional easing curve.
//!
//! ## Quick start
//!
//! ```zig
//! var pos = Interp3.init(.{ 0, 0, 0 });
//! pos.setDuration(0.5);          // transition takes 0.5 s
//! pos.set(.{ 10, 0, 0 }, now()); // start moving toward (10,0,0)
//! ...
//! const current = pos.get(now()); // sample at any point in time
//! ```
//!
//! ## Pre-defined aliases
//!
//! | Alias | Type | Blend |
//! |---|---|---|
//! | `Interp` / `Interpd` | f32 / f64 | linear scalar |
//! | `Interp2..4` / `Interp2d..4d` | @Vector(n, f32/f64) | linear vector |
//! | `InterpRot2/3` / `InterpRot2d/3d` | Rotor | slerp |
//!
//! Use `InterpType` directly to define interpolation over custom types.
//!
//! ## Easing functions
//!
//! All easing functions satisfy `f(0) == 0` and `f(1) == 1`.
//! Assign to `InterpType.easing_fn` after construction to change the curve.

const std = @import("std");
const math = std.math;
const linalg = @import("linalg.zig");
const expectApproxEqAbs = std.testing.expectApproxEqAbs;

// ---------------------------------------------------------------------------
// InterpType — unified interpolation for any type with a blend kernel
// ---------------------------------------------------------------------------

/// Generic interpolation type parameterized by value type `T` and a blend
/// function that knows how to interpolate between two `T` values.
///
/// The blend function signature is `fn (T, T, f32) T` — it receives the
/// start value, end value, and an eased parameter `t` in [0, 1], and returns
/// the interpolated result.
///
/// Different domains require different blend kernels:
/// - Scalars/vectors: linear interpolation (`a + (b - a) * t`)
/// - Rotations: spherical linear interpolation (slerp on the unit hypersphere)
///
/// Use the `Blend` namespace to obtain kernel functions for common types,
/// or provide a custom kernel for domain-specific interpolation.
///
/// ## Usage
///
/// ```
/// // Via pre-defined aliases:
/// var pos = Interp3.init(.{ 0, 0, 0 });
/// var rot = InterpRot3.init(Rot3.identity);
///
/// // Via InterpType directly with a custom blend:
/// const MyInterp = InterpType(MyType, &myBlendFn);
/// ```
pub fn InterpType(comptime T: type, comptime Time: type, comptime blendFn: *const fn (T, T, f32) T) type {
    return struct {
        /// Value at the beginning of the current transition.
        /// Updated by `set` to the sampled value at the moment of retargeting,
        /// ensuring continuity across mid-flight changes.
        start: T,
        /// Target value the transition is moving toward.
        end: T,
        /// Absolute time (same unit as arguments to `get`/`set`) at which the
        /// current transition began.
        start_time: Time,
        /// Reciprocal of the transition duration in time units.
        /// A `speed` of 2.0 means the transition completes in 0.5 time units.
        /// Defaults to 1.0 (one time unit per transition). Set via `setDuration`.
        speed: Time = 1.0,
        /// Easing curve applied to the normalized `t` before blending.
        /// Must map [0, 1] → [0, 1] with `f(0) == 0` and `f(1) == 1`.
        /// Defaults to `linear`. Assign any function from this module or a custom one.
        easing_fn: *const fn (f32) f32 = &linear,

        const Self = @This();

        /// Returns an interpolator whose `start` and `end` are both `initial_value`,
        /// so `get` returns `initial_value` at any time until the first `set` call.
        pub fn init(initial_value: T) Self {
            return .{
                .start = initial_value,
                .end = initial_value,
                .start_time = 0.0,
            };
        }

        /// Samples the interpolated value at `current_time`.
        ///
        /// Returns `end` once the transition is complete. The easing function is
        /// applied to the normalized elapsed time before passing it to the blend kernel.
        pub fn get(self: Self, current_time: Time) T {
            const elapsed = current_time - self.start_time;
            const t: f32 = @floatCast(elapsed * self.speed);
            if (t >= 1.0) return self.end;
            const eased = self.easing_fn(t);
            return blendFn(self.start, self.end, eased);
        }

        /// Begins a new transition toward `new_value` starting at `current_time`.
        ///
        /// The current in-progress value is sampled via `get` and becomes the new
        /// `start`, ensuring no discontinuity at the transition boundary.
        /// The speed (duration) from the previous transition is preserved; call
        /// `setDuration` after `set` to change it.
        pub fn set(self: *Self, new_value: T, current_time: Time) void {
            self.start = self.get(current_time);
            self.end = new_value;
            self.start_time = current_time;
        }

        /// Sets the transition duration in time units, updating `speed` accordingly.
        ///
        /// Assumes `duration` is positive and non-zero.
        pub fn setDuration(self: *Self, duration: Time) void {
            self.speed = 1.0 / duration;
        }

        /// Returns true when the interpolation has reached its end value.
        pub fn isDone(self: Self, current_time: Time) bool {
            const t: f32 = @floatCast((current_time - self.start_time) * self.speed);
            return t >= 1.0;
        }

        /// Normalized progress [0, 1]. Clamped — returns 1.0 after completion.
        pub fn progress(self: Self, current_time: Time) f32 {
            const t: f32 = @floatCast((current_time - self.start_time) * self.speed);
            return @min(1.0, @max(0.0, t));
        }
    };
}

// ---------------------------------------------------------------------------
// Blend — kernel functions for common type domains
// ---------------------------------------------------------------------------

/// Blend kernel generators for use with `InterpType`.
///
/// Each function returns a comptime function pointer suitable for the
/// `blendFn` parameter of `InterpType`.
pub const Blend = struct {
    /// Linear interpolation for scalar float types (f32, f64).
    /// Kernel: `a + (b - a) * t`
    pub fn scalar(comptime T: type) *const fn (T, T, f32) T {
        return &struct {
            fn f(a: T, b: T, t: f32) T {
                const tt: T = @floatCast(t);
                return a + (b - a) * tt;
            }
        }.f;
    }

    /// Linear interpolation for `@Vector(n, S)` types.
    /// Kernel: `a + (b - a) * @splat(t)` — easing broadcast across all lanes.
    pub fn vector(comptime n: comptime_int, comptime S: type) *const fn (@Vector(n, S), @Vector(n, S), f32) @Vector(n, S) {
        const V = @Vector(n, S);
        return &struct {
            fn f(a: V, b: V, t: f32) V {
                const st: S = @floatCast(t);
                const tt: V = @splat(st);
                return a + (b - a) * tt;
            }
        }.f;
    }

    /// Spherical linear interpolation for rotor/quaternion types.
    /// Delegates to `R.slerp()` which stays on the unit hypersphere,
    /// takes the shortest arc, and produces constant angular velocity.
    pub fn rotation(comptime R: type) *const fn (R, R, f32) R {
        return &struct {
            fn f(a: R, b: R, t: f32) R {
                return R.slerp(a, b, @floatCast(t));
            }
        }.f;
    }
};

// ---------------------------------------------------------------------------
// Type aliases — declarative definitions over common types
// ---------------------------------------------------------------------------

// Scalar
pub const Interp = InterpType(f32, f32, Blend.scalar(f32));
pub const Interpd = InterpType(f64, f64, Blend.scalar(f64));

// Vector (f32)
pub const Interp2 = InterpType(@Vector(2, f32), f32, Blend.vector(2, f32));
pub const Interp3 = InterpType(@Vector(3, f32), f32, Blend.vector(3, f32));
pub const Interp4 = InterpType(@Vector(4, f32), f32, Blend.vector(4, f32));

// Vector (f64)
pub const Interp2d = InterpType(@Vector(2, f64), f64, Blend.vector(2, f64));
pub const Interp3d = InterpType(@Vector(3, f64), f64, Blend.vector(3, f64));
pub const Interp4d = InterpType(@Vector(4, f64), f64, Blend.vector(4, f64));

// Rotation (f32)
pub const InterpRot2 = InterpType(linalg.Rotor2Type(f32), f32, Blend.rotation(linalg.Rotor2Type(f32)));
pub const InterpRot3 = InterpType(linalg.Rotor3Type(f32), f32, Blend.rotation(linalg.Rotor3Type(f32)));

// Rotation (f64)
pub const InterpRot2d = InterpType(linalg.Rotor2Type(f64), f64, Blend.rotation(linalg.Rotor2Type(f64)));
pub const InterpRot3d = InterpType(linalg.Rotor3Type(f64), f64, Blend.rotation(linalg.Rotor3Type(f64)));

// Versor (f32) — interpolation within same parity (rotation or reflection)
pub const InterpVersor2 = InterpType(linalg.Versor2Type(f32), f32, Blend.rotation(linalg.Versor2Type(f32)));
pub const InterpVersor3 = InterpType(linalg.Versor3Type(f32), f32, Blend.rotation(linalg.Versor3Type(f32)));

// Versor (f64)
pub const InterpVersor2d = InterpType(linalg.Versor2Type(f64), f64, Blend.rotation(linalg.Versor2Type(f64)));
pub const InterpVersor3d = InterpType(linalg.Versor3Type(f64), f64, Blend.rotation(linalg.Versor3Type(f64)));

// ---------------------------------------------------------------------------
// Easing functions
// ---------------------------------------------------------------------------

/// Instant snap — always returns 1.0, skipping the transition entirely.
/// Useful as a no-op easing when you want `get` to jump straight to `end`.
pub fn none(_: f32) f32 {
    return 1.0;
}

/// Identity easing — no curve applied; `t` passes through unchanged.
pub fn linear(t: f32) f32 {
    return t;
}

/// Quadratic ease-in: slow start, fast finish. Rate of change begins at zero.
pub fn easeInQuad(t: f32) f32 {
    return t * t;
}

/// Quadratic ease-out: fast start, slow finish. Rate of change ends at zero.
pub fn easeOutQuad(t: f32) f32 {
    return 1.0 - (1.0 - t) * (1.0 - t);
}

/// Quadratic ease-in-out: slow at both ends, fastest in the middle.
pub fn easeInOutQuad(t: f32) f32 {
    if (t < 0.5) {
        return 2.0 * t * t;
    }
    const u = -2.0 * t + 2.0;
    return 1.0 - u * u / 2.0;
}

/// Cubic ease-in: stronger acceleration than quadratic.
pub fn easeInCubic(t: f32) f32 {
    return t * t * t;
}

/// Cubic ease-out: stronger deceleration than quadratic.
pub fn easeOutCubic(t: f32) f32 {
    const u = 1.0 - t;
    return 1.0 - u * u * u;
}

/// Cubic ease-in-out: smooth S-curve with zero velocity at both endpoints.
pub fn easeInOutCubic(t: f32) f32 {
    if (t < 0.5) {
        return 4.0 * t * t * t;
    }
    const u = -2.0 * t + 2.0;
    return 1.0 - u * u * u / 2.0;
}

/// Sinusoidal ease-in: based on `1 - cos(t * π/2)`. Gentler than quadratic.
pub fn easeInSine(t: f32) f32 {
    return 1.0 - @cos(t * math.pi / 2.0);
}

/// Sinusoidal ease-out: based on `sin(t * π/2)`. Gentler than quadratic.
pub fn easeOutSine(t: f32) f32 {
    return @sin(t * math.pi / 2.0);
}

/// Sinusoidal ease-in-out: cosine-based S-curve, the smoothest standard option.
pub fn easeInOutSine(t: f32) f32 {
    return -((@cos(math.pi * t) - 1.0) / 2.0);
}

/// Cubic Hermite S-curve: `3t² - 2t³`. Equivalent to GLSL `smoothstep`.
/// Zero first-derivative at endpoints; good default for most UI transitions.
pub fn smoothstep(t: f32) f32 {
    return t * t * (3.0 - 2.0 * t);
}

/// Ken Perlin's improved smoothstep: `6t⁵ - 15t⁴ + 10t³`.
/// Also has zero second-derivative at endpoints, eliminating visible curvature discontinuities.
pub fn smootherstep(t: f32) f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// ===========================================================================
// Tests
// ===========================================================================

const tolerance: f32 = 1e-6;

// -- Scalar tests --

test "init returns initial value at any time" {
    const v = Interp.init(5.0);
    try expectApproxEqAbs(5.0, v.get(0.0), tolerance);
    try expectApproxEqAbs(5.0, v.get(1.0), tolerance);
    try expectApproxEqAbs(5.0, v.get(100.0), tolerance);
}

test "linear interpolation start, mid, end" {
    var v = Interp.init(0.0);
    v.set(10.0, 0.0);
    v.setDuration(1.0);

    try expectApproxEqAbs(0.0, v.get(0.0), tolerance);
    try expectApproxEqAbs(5.0, v.get(0.5), tolerance);
    try expectApproxEqAbs(10.0, v.get(1.0), tolerance);
    // Past duration returns end
    try expectApproxEqAbs(10.0, v.get(2.0), tolerance);
}

test "mid-transition retarget has no discontinuity" {
    var v = Interp.init(0.0);
    v.set(10.0, 0.0);
    v.setDuration(1.0);

    // At t=0.5, value should be 5.0
    const mid_value = v.get(0.5);
    try expectApproxEqAbs(5.0, mid_value, tolerance);

    // Retarget to 20.0 at t=0.5
    v.set(20.0, 0.5);
    v.setDuration(1.0);

    // Immediately after retarget, value should still be 5.0
    try expectApproxEqAbs(5.0, v.get(0.5), tolerance);
    // Halfway through new transition
    try expectApproxEqAbs(12.5, v.get(1.0), tolerance);
    // End of new transition
    try expectApproxEqAbs(20.0, v.get(1.5), tolerance);
}

test "custom easing function" {
    var v = Interp.init(0.0);
    v.set(10.0, 0.0);
    v.setDuration(1.0);
    v.easing_fn = &easeInQuad;

    // easeInQuad(0.5) = 0.25, so value = 0 + 10 * 0.25 = 2.5
    try expectApproxEqAbs(2.5, v.get(0.5), tolerance);
}

test "f64 variant works" {
    var v = Interpd.init(0.0);
    v.set(10.0, 0.0);
    v.setDuration(1.0);

    try expectApproxEqAbs(5.0, v.get(0.5), tolerance);
    try expectApproxEqAbs(10.0, v.get(1.0), tolerance);
}

test "isDone: scalar" {
    var v = Interp.init(0.0);
    v.set(10.0, 0.0);
    v.setDuration(1.0);

    try std.testing.expect(!v.isDone(0.5));
    try std.testing.expect(v.isDone(1.0));
    try std.testing.expect(v.isDone(2.0));
}

test "progress: scalar" {
    var v = Interp.init(0.0);
    v.set(10.0, 0.0);
    v.setDuration(2.0);

    try expectApproxEqAbs(@as(f32, 0.0), v.progress(0.0), tolerance);
    try expectApproxEqAbs(@as(f32, 0.5), v.progress(1.0), tolerance);
    try expectApproxEqAbs(@as(f32, 1.0), v.progress(2.0), tolerance);
    // Clamped past end
    try expectApproxEqAbs(@as(f32, 1.0), v.progress(5.0), tolerance);
}

// -- Vector tests --

test "Interp3: linear interpolation" {
    var v = Interp3.init(.{ 0, 0, 0 });
    v.set(.{ 10, 20, 30 }, 0.0);
    v.setDuration(1.0);

    const mid = v.get(0.5);
    try expectApproxEqAbs(@as(f32, 5.0), mid[0], tolerance);
    try expectApproxEqAbs(@as(f32, 10.0), mid[1], tolerance);
    try expectApproxEqAbs(@as(f32, 15.0), mid[2], tolerance);

    const end = v.get(1.0);
    try expectApproxEqAbs(@as(f32, 10.0), end[0], tolerance);
    try expectApproxEqAbs(@as(f32, 20.0), end[1], tolerance);
    try expectApproxEqAbs(@as(f32, 30.0), end[2], tolerance);
}

test "Interp3: isDone" {
    var v = Interp3.init(.{ 0, 0, 0 });
    v.set(.{ 1, 1, 1 }, 0.0);
    v.setDuration(0.5);

    try std.testing.expect(!v.isDone(0.25));
    try std.testing.expect(v.isDone(0.5));
}

test "Interp3: retarget preserves continuity" {
    var v = Interp3.init(.{ 0, 0, 0 });
    v.set(.{ 10, 10, 10 }, 0.0);
    v.setDuration(1.0);

    // Retarget at t=0.5
    v.set(.{ 20, 20, 20 }, 0.5);
    v.setDuration(1.0);

    // Should be at (5,5,5) immediately after retarget
    const snap = v.get(0.5);
    try expectApproxEqAbs(@as(f32, 5.0), snap[0], tolerance);
    try expectApproxEqAbs(@as(f32, 5.0), snap[1], tolerance);

    // Should reach (20,20,20) at t=1.5
    const done = v.get(1.5);
    try expectApproxEqAbs(@as(f32, 20.0), done[0], tolerance);
    try expectApproxEqAbs(@as(f32, 20.0), done[1], tolerance);
}

test "Interp2: basic" {
    var v = Interp2.init(.{ 0, 0 });
    v.set(.{ 4, 8 }, 0.0);
    v.setDuration(1.0);

    const mid = v.get(0.5);
    try expectApproxEqAbs(@as(f32, 2.0), mid[0], tolerance);
    try expectApproxEqAbs(@as(f32, 4.0), mid[1], tolerance);
}

test "Interp4: basic" {
    var v = Interp4.init(.{ 0, 0, 0, 0 });
    v.set(.{ 1, 2, 3, 4 }, 0.0);
    v.setDuration(1.0);

    const end = v.get(1.0);
    try expectApproxEqAbs(@as(f32, 1.0), end[0], tolerance);
    try expectApproxEqAbs(@as(f32, 2.0), end[1], tolerance);
    try expectApproxEqAbs(@as(f32, 3.0), end[2], tolerance);
    try expectApproxEqAbs(@as(f32, 4.0), end[3], tolerance);
}

// -- Rotation tests --

test "InterpRot3: init returns identity at any time" {
    const Rot3 = linalg.Rotor3Type(f32);
    const v = InterpRot3.init(Rot3.identity);
    const r = v.get(100.0);
    try expectApproxEqAbs(@as(f32, 1.0), r.a, tolerance);
    try expectApproxEqAbs(@as(f32, 0.0), r.b01, tolerance);
    try expectApproxEqAbs(@as(f32, 0.0), r.b02, tolerance);
    try expectApproxEqAbs(@as(f32, 0.0), r.b12, tolerance);
}

test "InterpRot3: endpoints are exact" {
    const Rot3 = linalg.Rotor3Type(f32);
    const Vec3 = linalg.VecType(3, f32);
    const y_axis = Vec3.init(.{ 0, 1, 0 });
    const r0 = Rot3.identity;
    const r1 = Rot3.fromAxisAngle(math.pi / 2.0, y_axis);

    var v = InterpRot3.init(r0);
    v.set(r1, 0.0);
    v.setDuration(1.0);

    // t=0: should be r0 (identity)
    const at_start = v.get(0.0);
    try expectApproxEqAbs(@as(f32, 1.0), at_start.a, 1e-4);

    // t=1: should be r1
    const at_end = v.get(1.0);
    try expectApproxEqAbs(r1.a, at_end.a, 1e-4);
    try expectApproxEqAbs(r1.b01, at_end.b01, 1e-4);
    try expectApproxEqAbs(r1.b02, at_end.b02, 1e-4);
    try expectApproxEqAbs(r1.b12, at_end.b12, 1e-4);
}

test "InterpRot3: midpoint produces valid rotation" {
    const Rot3 = linalg.Rotor3Type(f32);
    const Vec3 = linalg.VecType(3, f32);
    const y_axis = Vec3.init(.{ 0, 1, 0 });
    const r0 = Rot3.identity;
    const r1 = Rot3.fromAxisAngle(math.pi / 2.0, y_axis);

    var v = InterpRot3.init(r0);
    v.set(r1, 0.0);
    v.setDuration(1.0);

    const mid = v.get(0.5);

    // Must be unit length (valid rotation)
    const len_sq = mid.a * mid.a + mid.b01 * mid.b01 + mid.b02 * mid.b02 + mid.b12 * mid.b12;
    try expectApproxEqAbs(@as(f32, 1.0), len_sq, 1e-4);

    // Should be a 45° rotation (half of 90°)
    // cos(45°/2) = cos(π/8) ≈ 0.9239
    try expectApproxEqAbs(@as(f32, @cos(math.pi / 8.0)), mid.a, 1e-4);
}

test "InterpRot3: rotated vector at midpoint" {
    const Rot3 = linalg.Rotor3Type(f32);
    const Vec3 = linalg.VecType(3, f32);
    const y_axis = Vec3.init(.{ 0, 1, 0 });
    const r0 = Rot3.identity;
    const r1 = Rot3.fromAxisAngle(math.pi / 2.0, y_axis); // 90° around Y

    var v = InterpRot3.init(r0);
    v.set(r1, 0.0);
    v.setDuration(1.0);

    // Rotate +X by the midpoint (45° around Y)
    const mid = v.get(0.5);
    const x_axis = Vec3.init(.{ 1, 0, 0 });
    const rotated = mid.rotate(x_axis);

    // 45° rotation of +X around Y → (cos45°, 0, -sin45°) ≈ (0.707, 0, -0.707)
    const c45: f32 = @cos(math.pi / 4.0);
    try expectApproxEqAbs(c45, rotated.v[0], 1e-4);
    try expectApproxEqAbs(@as(f32, 0.0), rotated.v[1], 1e-4);
    try expectApproxEqAbs(-c45, rotated.v[2], 1e-4);
}

test "InterpRot3: retarget preserves continuity" {
    const Rot3 = linalg.Rotor3Type(f32);
    const Vec3 = linalg.VecType(3, f32);
    const y_axis = Vec3.init(.{ 0, 1, 0 });
    const r0 = Rot3.identity;
    const r1 = Rot3.fromAxisAngle(math.pi / 2.0, y_axis);
    const r2 = Rot3.fromAxisAngle(math.pi, y_axis);

    var v = InterpRot3.init(r0);
    v.set(r1, 0.0);
    v.setDuration(1.0);

    // Sample at t=0.5 (45° rotation)
    const before = v.get(0.5);

    // Retarget to 180° at t=0.5
    v.set(r2, 0.5);
    v.setDuration(1.0);

    // Immediately after retarget: should be same rotation as before
    const after = v.get(0.5);
    try expectApproxEqAbs(before.a, after.a, 1e-4);
    try expectApproxEqAbs(before.b01, after.b01, 1e-4);
    try expectApproxEqAbs(before.b02, after.b02, 1e-4);
    try expectApproxEqAbs(before.b12, after.b12, 1e-4);

    // At t=1.5: should reach r2 (180°)
    const final = v.get(1.5);
    try expectApproxEqAbs(r2.a, final.a, 1e-4);
    try expectApproxEqAbs(r2.b01, final.b01, 1e-4);
    try expectApproxEqAbs(r2.b02, final.b02, 1e-4);
    try expectApproxEqAbs(r2.b12, final.b12, 1e-4);
}

test "InterpRot3: isDone and progress" {
    const Rot3 = linalg.Rotor3Type(f32);
    const Vec3 = linalg.VecType(3, f32);
    const y_axis = Vec3.init(.{ 0, 1, 0 });

    var v = InterpRot3.init(Rot3.identity);
    v.set(Rot3.fromAxisAngle(math.pi, y_axis), 0.0);
    v.setDuration(2.0);

    try std.testing.expect(!v.isDone(1.0));
    try std.testing.expect(v.isDone(2.0));
    try expectApproxEqAbs(@as(f32, 0.5), v.progress(1.0), tolerance);
    try expectApproxEqAbs(@as(f32, 1.0), v.progress(3.0), tolerance);
}

test "InterpRot2: 90 degree rotation" {
    const Rot2 = linalg.Rotor2Type(f32);
    const r0 = Rot2.fromAngle(0);
    const r1 = Rot2.fromAngle(math.pi / 2.0);

    var v = InterpRot2.init(r0);
    v.set(r1, 0.0);
    v.setDuration(1.0);

    // At t=1, should match r1
    const at_end = v.get(1.0);
    try expectApproxEqAbs(r1.a, at_end.a, 1e-4);
    try expectApproxEqAbs(r1.b, at_end.b, 1e-4);
}

// -- Easing tests --

test "easing boundary values: f(0)=0, f(1)=1" {
    const fns = [_]*const fn (f32) f32{
        &linear,
        &easeInQuad,
        &easeOutQuad,
        &easeInOutQuad,
        &easeInCubic,
        &easeOutCubic,
        &easeInOutCubic,
        &easeInSine,
        &easeOutSine,
        &easeInOutSine,
        &smoothstep,
        &smootherstep,
    };

    for (fns) |f| {
        try expectApproxEqAbs(0.0, f(0.0), tolerance);
        try expectApproxEqAbs(1.0, f(1.0), tolerance);
    }
}
