//! Affine mirror (reflection) across a plane or line.
//!
//! `MirrorType(n, Scalar)` provides a high-level wrapper over versor-based
//! reflections with support for an offset origin. The core linear operation
//! is `v − 2(v·n)n`; the origin shift makes it affine:
//! `p' = p − 2·dot(p − origin, normal) · normal`.
//!
//! Use `toVersor()` to obtain the underlying odd versor for the linear part,
//! which can be composed with other versors via `Versor.mul`.

const linalg = @import("linalg.zig");

// TODO: Separate MirrorType into Mirror2 and Mirror3. This impl does not benefit from comptime generic dimension, only scalar type.
// TODO: Move Mirror type code into linalg.zig

pub fn MirrorType(comptime n: comptime_int, comptime Scalar: type) type {
    if (n != 2 and n != 3) @compileError("MirrorType only supports n=2 or n=3");

    const VecS = linalg.VecType(n, Scalar);
    const VersorS = if (n == 2) linalg.Versor2Type(Scalar) else linalg.Versor3Type(Scalar);

    return struct {
        const Self = @This();

        /// Unit normal of the mirror plane (3D) or mirror line (2D).
        normal: VecS,
        /// A point on the mirror plane / line. Defaults to the origin.
        origin: VecS = VecS.zero,

        // -----------------------------------------------------------------
        // Convenience constants (through-origin planes / lines)
        // -----------------------------------------------------------------

        pub const acrossYZ = if (n == 3) Self{ .normal = VecS.init(.{ 1, 0, 0 }) } else {};
        pub const acrossXZ = if (n == 3) Self{ .normal = VecS.init(.{ 0, 1, 0 }) } else {};
        pub const acrossXY = if (n == 3) Self{ .normal = VecS.init(.{ 0, 0, 1 }) } else {};

        pub const acrossY = if (n == 2) Self{ .normal = VecS.init(.{ 1, 0 }) } else {};
        pub const acrossX = if (n == 2) Self{ .normal = VecS.init(.{ 0, 1 }) } else {};

        // -----------------------------------------------------------------
        // Construction
        // -----------------------------------------------------------------

        /// Create a mirror through the origin with the given unit normal.
        pub fn fromNormal(normal: VecS) Self {
            return .{ .normal = normal };
        }

        /// Create a mirror at `origin` with the given unit normal.
        pub fn fromPlane(normal: VecS, origin: VecS) Self {
            return .{ .normal = normal, .origin = origin };
        }

        // -----------------------------------------------------------------
        // Application
        // -----------------------------------------------------------------

        /// Reflect a point across this mirror (affine — accounts for origin offset).
        pub fn applyPoint(self: Self, point: VecS) VecS {
            // p' = p − 2·dot(p − origin, normal)·normal
            const diff = point.sub(self.origin);
            const d: Scalar = @as(Scalar, 2) * diff.dot(self.normal);
            return point.sub(self.normal.scale(d));
        }

        /// Reflect a direction vector (translation-invariant — ignores origin).
        pub fn applyDir(self: Self, dir: VecS) VecS {
            const d: Scalar = @as(Scalar, 2) * dir.dot(self.normal);
            return dir.sub(self.normal.scale(d));
        }

        // -----------------------------------------------------------------
        // Conversion
        // -----------------------------------------------------------------

        /// Convert to the underlying odd versor (linear part only, no origin offset).
        /// Compose with other versors via `Versor.mul`.
        pub fn toVersor(self: Self) VersorS {
            return VersorS.fromReflection(self.normal);
        }
    };
}

// ── Tests ──

const std = @import("std");
const math = std.math;
const eps_normal: f32 = @sqrt(math.floatEps(f32));
const Vec2 = linalg.VecType(2, f32);
const Vec3 = linalg.VecType(3, f32);
const Mirror2 = MirrorType(2, f32);
const Mirror3 = MirrorType(3, f32);

test "Mirror3: acrossYZ reflects X" {
    const m = Mirror3.acrossYZ;
    const result = m.applyPoint(Vec3.init(.{ 3, 4, 5 }));
    try std.testing.expect(result.eql(Vec3.init(.{ -3, 4, 5 }), eps_normal));
}

test "Mirror3: acrossXZ reflects Y" {
    const m = Mirror3.acrossXZ;
    const result = m.applyPoint(Vec3.init(.{ 3, 4, 5 }));
    try std.testing.expect(result.eql(Vec3.init(.{ 3, -4, 5 }), eps_normal));
}

test "Mirror3: acrossXY reflects Z" {
    const m = Mirror3.acrossXY;
    const result = m.applyPoint(Vec3.init(.{ 3, 4, 5 }));
    try std.testing.expect(result.eql(Vec3.init(.{ 3, 4, -5 }), eps_normal));
}

test "Mirror3: offset origin" {
    // Mirror at Y=5, normal = +Y
    const m = Mirror3.fromPlane(Vec3.init(.{ 0, 1, 0 }), Vec3.init(.{ 0, 5, 0 }));
    const result = m.applyPoint(Vec3.init(.{ 1, 3, 0 }));
    // 3 is 2 below Y=5, so reflected to Y=7
    try std.testing.expect(result.eql(Vec3.init(.{ 1, 7, 0 }), eps_normal));
}

test "Mirror3: applyDir ignores origin" {
    const m = Mirror3.fromPlane(Vec3.init(.{ 1, 0, 0 }), Vec3.init(.{ 100, 0, 0 }));
    const result = m.applyDir(Vec3.init(.{ 1, 0, 0 }));
    try std.testing.expect(result.eql(Vec3.init(.{ -1, 0, 0 }), eps_normal));
}

test "Mirror3: double reflection is identity" {
    const m = Mirror3.fromNormal(Vec3.init(.{ 0, 0, 1 }));
    const v = Vec3.init(.{ 3, 4, 5 });
    const result = m.applyPoint(m.applyPoint(v));
    try std.testing.expect(result.eql(v, eps_normal));
}

test "Mirror3: toVersor matches applyDir" {
    const s2: f32 = 1.0 / @sqrt(2.0);
    const m = Mirror3.fromNormal(Vec3.init(.{ s2, s2, 0 }));
    const versor = m.toVersor();
    const v = Vec3.init(.{ 1, 0, 0 });
    try std.testing.expect(m.applyDir(v).eql(versor.apply(v), eps_normal));
}

test "Mirror3: preserves length" {
    const s3: f32 = 1.0 / @sqrt(3.0);
    const m = Mirror3.fromNormal(Vec3.init(.{ s3, s3, s3 }));
    const v = Vec3.init(.{ 3, 4, 5 });
    const reflected = m.applyDir(v);
    try std.testing.expect(@abs(v.len() - reflected.len()) <= eps_normal);
}

test "Mirror2: acrossY reflects X" {
    const m = Mirror2.acrossY;
    const result = m.applyPoint(Vec2.init(.{ 3, 4 }));
    try std.testing.expect(result.eql(Vec2.init(.{ -3, 4 }), eps_normal));
}

test "Mirror2: acrossX reflects Y" {
    const m = Mirror2.acrossX;
    const result = m.applyPoint(Vec2.init(.{ 3, 4 }));
    try std.testing.expect(result.eql(Vec2.init(.{ 3, -4 }), eps_normal));
}

test "Mirror2: offset origin" {
    const m = Mirror2.fromPlane(Vec2.init(.{ 1, 0 }), Vec2.init(.{ 5, 0 }));
    const result = m.applyPoint(Vec2.init(.{ 3, 7 }));
    // 3 is 2 left of X=5, reflected to X=7
    try std.testing.expect(result.eql(Vec2.init(.{ 7, 7 }), eps_normal));
}

test "Mirror2: toVersor matches applyDir" {
    const s2: f32 = 1.0 / @sqrt(2.0);
    const m = Mirror2.fromNormal(Vec2.init(.{ s2, s2 }));
    const versor = m.toVersor();
    const v = Vec2.init(.{ 1, 0 });
    try std.testing.expect(m.applyDir(v).eql(versor.apply(v), eps_normal));
}
