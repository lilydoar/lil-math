//! Spatial transform combining translation, rotation, and non-uniform scale.
//!
//! A `Transform` decomposes a 4×4 affine matrix into its three intuitive
//! components. Conversion to `Mat4` applies them in TRS order — scale first,
//! then rotation, then translation — matching the convention used by most
//! real-time renderers and game engines.

const std = @import("std");
const root = @import("root.zig");

const Vec3 = root.Vec3;
const Vec4 = root.Vec4;
const Mat4 = root.Mat4;
const Rot3 = root.Rot3;
const Approx = root.Approx;

/// A spatial transform: translation, rotation, and scale.
/// Composes to a Mat4 as T * R * S (scale applied first, then rotation, then translation).
pub const Transform = struct {
    /// World-space offset applied after rotation and scale.
    translation: Vec3 = Vec3.init(.{ 0, 0, 0 }),
    /// Orientation encoded as a unit rotor (geometric-algebra quaternion equivalent).
    rotation: Rot3 = Rot3.identity,
    /// Per-axis scale factors. Values of 1.0 produce no scaling.
    scaling: Vec3 = Vec3.init(.{ 1, 1, 1 }),

    /// The identity transform: no translation, no rotation, unit scale.
    pub const identity: Transform = .{};

    /// Converts the transform to a column-major 4×4 matrix in TRS order.
    ///
    /// The resulting matrix is equivalent to `T * R * S`, so a point is first
    /// scaled, then rotated, then translated when multiplied on the right.
    pub fn toMat4(self: Transform) Mat4 {
        const t = Mat4.translation(self.translation);
        const r = self.rotation.toMat4();
        const s = Mat4.scaling(self.scaling);
        return t.mul(r.mul(s));
    }

    /// Returns a transform with `v` as translation and identity rotation and scale.
    pub fn fromTranslation(v: Vec3) Transform {
        return .{ .translation = v };
    }

    /// Returns a transform with `r` as rotation and zero translation and unit scale.
    pub fn fromRotation(r: Rot3) Transform {
        return .{ .rotation = r };
    }
};

// Tests

const approx = Approx(.normal);

test "identity transform produces identity matrix" {
    const t = Transform.identity;
    const m = t.toMat4();
    inline for (0..4) |col| {
        inline for (0..4) |row| {
            const expected: f32 = if (col == row) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, m.m[col][row], 1e-6);
        }
    }
}

test "translation-only transform" {
    const t = Transform.fromTranslation(Vec3.init(.{ 3, 4, 5 }));
    const m = t.toMat4();
    const expected = Mat4.translation(Vec3.init(.{ 3, 4, 5 }));
    try std.testing.expect(approx.eql(m, expected));
}

test "rotation-only transform" {
    const axis = Vec3.init(.{ 0, 1, 0 });
    const angle: f32 = std.math.pi / 4.0;
    const rot = Rot3.fromAxisAngle(angle, axis);
    const t = Transform.fromRotation(rot);
    const m = t.toMat4();
    const expected = rot.toMat4();
    try std.testing.expect(approx.eql(m, expected));
}

test "combined TRS applies scale then rotation then translation" {
    const t = Transform{
        .translation = Vec3.init(.{ 10, 0, 0 }),
        .rotation = Rot3.identity,
        .scaling = Vec3.init(.{ 2, 2, 2 }),
    };
    const m = t.toMat4();

    // A point at (1, 0, 0) should be scaled to (2, 0, 0) then translated to (12, 0, 0)
    const point = Vec3.init(.{ 1, 0, 0 });
    const result = m.mulVec(Vec4.init(.{ point.v[0], point.v[1], point.v[2], 1.0 }));
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), result.v[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.v[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.v[2], 1e-6);
}
