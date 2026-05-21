//! Geometric primitives: rays, segments, and bounding volumes.
//!
//! Provides generic types parameterized by dimension `n` and scalar type.
//! Most types support both integer and float scalars, with some operations
//! (e.g. length, closest point) restricted to float types via conditional
//! compilation.

const std = @import("std");
const math = std.math;
const linalg = @import("linalg.zig");

/// A ray in n-dimensional space with an origin and direction.
/// The direction vector is not normalized by default.
pub fn RayType(comptime n: comptime_int, comptime Scalar: type) type {
    const Vec = linalg.VecType(n, Scalar);

    return struct {
        const Self = @This();

        /// Starting point of the ray.
        origin: Vec,
        /// Direction vector. Not required to be normalized.
        dir: Vec,

        pub inline fn init(origin: Vec, dir: Vec) Self {
            return .{ .origin = origin, .dir = dir };
        }

        /// Returns the point at parameter `t` along the ray: origin + t * dir.
        pub inline fn pointAt(self: Self, t: Scalar) Vec {
            return self.origin.add(self.dir.scale(t));
        }

        /// Returns the parametric t at which the ray intersects the axis-aligned
        /// plane `point[axis] == value`. Returns null if the ray is parallel to
        /// the plane (|dir[axis]| < floatEps) or if the intersection is behind
        /// the origin (t < 0).
        pub inline fn intersectAxisPlane(self: Self, comptime axis: comptime_int, value: Scalar) ?Scalar {
            const d = self.dir.v[axis];
            if (@abs(d) < math.floatEps(Scalar)) return null;
            const t = (value - self.origin.v[axis]) / d;
            if (t < 0) return null;
            return t;
        }
    };
}

/// A line segment in n-dimensional space defined by two endpoints.
pub fn SegmentType(comptime n: comptime_int, comptime Scalar: type) type {
    const Vec = linalg.VecType(n, Scalar);
    const is_float = @typeInfo(Scalar) == .float;

    return struct {
        const Self = @This();

        /// Starting endpoint of the segment.
        a: Vec,
        /// Ending endpoint of the segment.
        b: Vec,

        pub inline fn init(a: Vec, b: Vec) Self {
            return .{ .a = a, .b = b };
        }

        /// Returns the direction vector from `a` to `b`.
        pub inline fn dir(self: Self) Vec {
            return self.b.sub(self.a);
        }

        /// Returns the midpoint of the segment.
        pub inline fn midpoint(self: Self) Vec {
            const two: Vec = .{ .v = @splat(2) };
            return self.a.add(self.b).div(two);
        }

        /// Converts the segment to a ray starting at `a` with direction toward `b`.
        pub inline fn toRay(self: Self) RayType(n, Scalar) {
            return .{ .origin = self.a, .dir = self.b.sub(self.a) };
        }

        /// Returns the squared length of the segment.
        pub inline fn lenSqr(self: Self) Scalar {
            const d = self.b.sub(self.a);
            return d.dot(d);
        }

        /// Returns the length of the segment.
        /// Only available for float scalar types.
        pub const len = if (is_float) struct {
            fn f(self: Self) Scalar {
                return @sqrt(self.lenSqr());
            }
        }.f else @compileError("len() requires a float scalar type");

        /// Returns the closest point on the segment to point `p`.
        /// Clamps the projection to the segment's endpoints.
        /// Only available for float scalar types.
        pub const closestPoint = if (is_float) struct {
            fn f(self: Self, p: Vec) Vec {
                const ab = self.b.sub(self.a);
                const ap = p.sub(self.a);
                var t = ap.dot(ab) / ab.dot(ab);
                t = @max(0, @min(1, t));
                return self.a.add(ab.scale(t));
            }
        }.f else @compileError("closestPoint() requires a float scalar type");
    };
}

const Vec2 = linalg.VecType(2, f32);
const Vec3 = linalg.VecType(3, f32);

const Segment2 = SegmentType(2, f32);
const Segment3 = SegmentType(3, f32);

const Segment2d = SegmentType(2, f64);
const Segment3d = SegmentType(3, f64);

/// Axis-aligned bounding box in n-dimensional space.
pub fn AabbType(comptime n: comptime_int, comptime Scalar: type) type {
    const Vec = linalg.VecType(n, Scalar);
    const is_float = @typeInfo(Scalar) == .float;

    // TODO: Union style structure to allow for min,max repr and half extent repr

    return struct {
        const Self = @This();

        /// Minimum corner of the box (component-wise).
        min: Vec,
        /// Maximum corner of the box (component-wise).
        max: Vec,

        pub inline fn init(min_pt: Vec, max_pt: Vec) Self {
            std.debug.assert(@reduce(.And, min_pt.v <= max_pt.v));
            return .{ .min = min_pt, .max = max_pt };
        }

        /// Constructs an AABB from a center point and size vector.
        /// Only available for float scalar types.
        pub inline fn fromCenterSize(c: Vec, s: Vec) Self {
            const half = if (is_float) s.scale(0.5) else unreachable;
            return .{ .min = c.sub(half), .max = c.add(half) };
        }

        /// Returns the center point of the box.
        /// Only available for float scalar types.
        pub inline fn center(self: Self) Vec {
            if (is_float) {
                return self.min.add(self.max).scale(0.5);
            }
            unreachable;
        }

        /// Returns the size of the box (max - min).
        pub inline fn size(self: Self) Vec {
            return self.max.sub(self.min);
        }

        /// Returns half the size of the box.
        /// Only available for float scalar types.
        pub inline fn halfSize(self: Self) Vec {
            if (is_float) {
                return self.max.sub(self.min).scale(0.5);
            }
            unreachable;
        }

        /// Returns true if point `p` is inside the box (inclusive of boundaries).
        pub inline fn contains(self: Self, p: Vec) bool {
            return @reduce(.And, p.v >= self.min.v) and @reduce(.And, p.v <= self.max.v);
        }

        /// Returns true if `other` is fully contained within this box.
        pub inline fn containsAabb(self: Self, other: Self) bool {
            return @reduce(.And, other.min.v >= self.min.v) and @reduce(.And, other.max.v <= self.max.v);
        }

        /// Returns true if this box overlaps with `other`.
        pub inline fn intersects(self: Self, other: Self) bool {
            return @reduce(.And, self.max.v >= other.min.v) and @reduce(.And, self.min.v <= other.max.v);
        }

        /// Returns a new AABB that includes both this box and point `p`.
        pub inline fn expand(self: Self, p: Vec) Self {
            return .{ .min = self.min.min(p), .max = self.max.max(p) };
        }

        /// Returns a new AABB that includes both this box and `other`.
        pub inline fn merge(self: Self, other: Self) Self {
            return .{ .min = self.min.min(other.min), .max = self.max.max(other.max) };
        }

        /// Returns the closest point on the box's surface (or interior) to `p`.
        /// Clamps each component of `p` to the box's bounds.
        pub inline fn closestPoint(self: Self, p: Vec) Vec {
            return .{ .v = @min(@max(p.v, self.min.v), self.max.v) };
        }

        /// Returns the volume (area in 2D, volume in 3D, etc.) of the box.
        pub inline fn volume(self: Self) Scalar {
            const s = self.max.sub(self.min);
            var result: Scalar = s.v[0];
            inline for (1..n) |i| {
                result *= s.v[i];
            }
            return result;
        }

        /// Tests ray intersection using the slab method.
        /// Returns the entry parameter t (may be negative if origin is inside).
        /// Returns null on miss.
        /// Only available for float scalar types.
        pub const intersectsRay = if (is_float) struct {
            fn f(self: Self, ray: RayType(n, Scalar)) ?Scalar {
                var tmin: Scalar = -math.inf(Scalar);
                var tmax: Scalar = math.inf(Scalar);

                inline for (0..n) |i| {
                    const inv_d = 1.0 / ray.dir.v[i];
                    var t0 = (self.min.v[i] - ray.origin.v[i]) * inv_d;
                    var t1 = (self.max.v[i] - ray.origin.v[i]) * inv_d;
                    if (inv_d < 0) {
                        const tmp = t0;
                        t0 = t1;
                        t1 = tmp;
                    }
                    tmin = @max(tmin, t0);
                    tmax = @min(tmax, t1);
                    if (tmax < tmin) return null;
                }
                return tmin;
            }
        }.f else @compileError("intersectsRay() requires a float scalar type");
    };
}

const Aabb2 = AabbType(2, f32);
const Aabb3 = AabbType(3, f32);

const Aabb2d = AabbType(2, f64);
const Aabb3d = AabbType(3, f64);

/// A sphere (or circle in 2D) defined by center point and radius.
/// Only supports float scalar types.
pub fn SphereType(comptime n: comptime_int, comptime Scalar: type) type {
    const Vec = linalg.VecType(n, Scalar);
    const is_float = @typeInfo(Scalar) == .float;
    if (!is_float) @compileError("SphereType requires a float scalar type");

    return struct {
        const Self = @This();

        /// Center point of the sphere.
        center: Vec,
        /// Radius of the sphere. Must be non-negative.
        radius: Scalar,

        pub inline fn init(c: Vec, r: Scalar) Self {
            std.debug.assert(r >= 0);
            return .{ .center = c, .radius = r };
        }

        /// Returns true if point `p` is inside the sphere (inclusive of boundary).
        pub inline fn contains(self: Self, p: Vec) bool {
            return self.center.sub(p).lenSqr() <= self.radius * self.radius;
        }

        /// Returns true if this sphere overlaps with `other`.
        pub inline fn intersects(self: Self, other: Self) bool {
            const r = self.radius + other.radius;
            return self.center.sub(other.center).lenSqr() <= r * r;
        }

        /// Returns true if this sphere overlaps with the given AABB.
        pub inline fn intersectsAabb(self: Self, box: AabbType(n, Scalar)) bool {
            const closest = box.closestPoint(self.center);
            return self.center.sub(closest).lenSqr() <= self.radius * self.radius;
        }

        /// Returns the nearest ray parameter t at which the ray enters the sphere,
        /// or null on miss. Returns negative t if the origin is inside.
        pub inline fn intersectsRay(self: Self, ray: RayType(n, Scalar)) ?Scalar {
            const oc = ray.origin.sub(self.center);
            const a = ray.dir.dot(ray.dir);
            const b = oc.dot(ray.dir);
            const c = oc.dot(oc) - self.radius * self.radius;
            const disc = b * b - a * c;
            if (disc < 0) return null;
            return (-b - @sqrt(disc)) / a;
        }

        /// Returns the smallest sphere that contains both this sphere and `other`.
        /// If one sphere contains the other, returns the larger sphere.
        pub inline fn merge(self: Self, other: Self) Self {
            const d_vec = other.center.sub(self.center);
            const dist = d_vec.length();
            if (dist + other.radius <= self.radius) return self;
            if (dist + self.radius <= other.radius) return other;
            const new_radius = (dist + self.radius + other.radius) * 0.5;
            const t = (new_radius - self.radius) / dist;
            return .{
                .center = self.center.add(d_vec.scale(t)),
                .radius = new_radius,
            };
        }
    };
}

const Ray2 = RayType(2, f32);
const Ray3 = RayType(3, f32);

const Ray2d = RayType(2, f64);
const Ray3d = RayType(3, f64);

const Circle = SphereType(2, f32);
const Sphere = SphereType(3, f32);

const Circled = SphereType(2, f64);
const Sphered = SphereType(3, f64);

fn expectApproxScalar(actual: f32, expected: f32) !void {
    if (@abs(actual - expected) > @sqrt(math.floatEps(f32))) return error.TestUnexpectedResult;
}

test "Ray: pointAt" {
    const ray = Ray3.init(
        Vec3.init(.{ 1, 2, 3 }),
        Vec3.init(.{ 0, 1, 0 }),
    );
    try std.testing.expect(ray.pointAt(0).eql(Vec3.init(.{ 1, 2, 3 })));
    try std.testing.expect(ray.pointAt(5).eql(Vec3.init(.{ 1, 7, 3 })));
    try std.testing.expect(ray.pointAt(-1).eql(Vec3.init(.{ 1, 1, 3 })));
}

test "Ray2: pointAt diagonal" {
    const ray = Ray2.init(
        Vec2.zero,
        Vec2.init(.{ 1, 1 }),
    );
    try std.testing.expect(ray.pointAt(3).eql(Vec2.init(.{ 3, 3 })));
}

test "Segment: length and midpoint" {
    const seg = Segment3.init(
        Vec3.init(.{ 0, 0, 0 }),
        Vec3.init(.{ 3, 4, 0 }),
    );
    try std.testing.expectEqual(@as(f32, 25), seg.lenSqr());
    try std.testing.expectEqual(@as(f32, 5), seg.len());
    try std.testing.expect(seg.midpoint().eql(Vec3.init(.{ 1.5, 2, 0 })));
}

test "Segment: dir and toRay" {
    const seg = Segment2.init(
        Vec2.init(.{ 1, 1 }),
        Vec2.init(.{ 4, 5 }),
    );
    try std.testing.expect(seg.dir().eql(Vec2.init(.{ 3, 4 })));

    const ray = seg.toRay();
    try std.testing.expect(ray.origin.eql(seg.a));
    try std.testing.expect(ray.dir.eql(Vec2.init(.{ 3, 4 })));
}

test "Segment: closestPoint" {
    const seg = Segment3.init(
        Vec3.init(.{ 0, 0, 0 }),
        Vec3.init(.{ 10, 0, 0 }),
    );

    // Point projects onto middle of segment
    try std.testing.expect(seg.closestPoint(Vec3.init(.{ 5, 3, 0 })).eql(Vec3.init(.{ 5, 0, 0 })));

    // Point before start — clamps to a
    try std.testing.expect(seg.closestPoint(Vec3.init(.{ -5, 1, 0 })).eql(Vec3.init(.{ 0, 0, 0 })));

    // Point past end — clamps to b
    try std.testing.expect(seg.closestPoint(Vec3.init(.{ 15, 1, 0 })).eql(Vec3.init(.{ 10, 0, 0 })));
}

test "Aabb: contains" {
    const box = Aabb3.init(Vec3.init(.{ -1, -1, -1 }), Vec3.init(.{ 1, 1, 1 }));
    try std.testing.expect(box.contains(Vec3.zero));
    try std.testing.expect(box.contains(Vec3.init(.{ 1, 1, 1 })));
    try std.testing.expect(!box.contains(Vec3.init(.{ 2, 0, 0 })));
}

test "Aabb: containsAabb" {
    const outer = Aabb2.init(Vec2.init(.{ 0, 0 }), Vec2.init(.{ 10, 10 }));
    const inner = Aabb2.init(Vec2.init(.{ 2, 2 }), Vec2.init(.{ 5, 5 }));
    const partial = Aabb2.init(Vec2.init(.{ 5, 5 }), Vec2.init(.{ 15, 15 }));
    try std.testing.expect(outer.containsAabb(inner));
    try std.testing.expect(!outer.containsAabb(partial));
}

test "Aabb: intersects" {
    const a = Aabb2.init(Vec2.init(.{ 0, 0 }), Vec2.init(.{ 2, 2 }));
    const b = Aabb2.init(Vec2.init(.{ 1, 1 }), Vec2.init(.{ 3, 3 }));
    const c = Aabb2.init(Vec2.init(.{ 5, 5 }), Vec2.init(.{ 6, 6 }));
    try std.testing.expect(a.intersects(b));
    try std.testing.expect(!a.intersects(c));
}

test "Aabb: center, size, halfSize, volume" {
    const box = Aabb3.init(Vec3.init(.{ -1, 0, 2 }), Vec3.init(.{ 3, 4, 8 }));
    try std.testing.expect(box.center().eql(Vec3.init(.{ 1, 2, 5 })));
    try std.testing.expect(box.size().eql(Vec3.init(.{ 4, 4, 6 })));
    try std.testing.expect(box.halfSize().eql(Vec3.init(.{ 2, 2, 3 })));
    try std.testing.expectEqual(@as(f32, 96), box.volume());
}

test "Aabb: fromCenterSize" {
    const box = Aabb3.fromCenterSize(Vec3.init(.{ 0, 0, 0 }), Vec3.init(.{ 4, 4, 4 }));
    try std.testing.expect(box.min.eql(Vec3.init(.{ -2, -2, -2 })));
    try std.testing.expect(box.max.eql(Vec3.init(.{ 2, 2, 2 })));
}

test "Aabb: expand and merge" {
    const box = Aabb3.init(Vec3.init(.{ 0, 0, 0 }), Vec3.init(.{ 1, 1, 1 }));
    const expanded = box.expand(Vec3.init(.{ -1, 2, 0.5 }));
    try std.testing.expect(expanded.min.eql(Vec3.init(.{ -1, 0, 0 })));
    try std.testing.expect(expanded.max.eql(Vec3.init(.{ 1, 2, 1 })));

    const other = Aabb3.init(Vec3.init(.{ -3, -3, -3 }), Vec3.init(.{ -2, -2, -2 }));
    const merged = box.merge(other);
    try std.testing.expect(merged.min.eql(Vec3.init(.{ -3, -3, -3 })));
    try std.testing.expect(merged.max.eql(Vec3.init(.{ 1, 1, 1 })));
}

test "Aabb: closestPoint" {
    const box = Aabb2.init(Vec2.init(.{ 0, 0 }), Vec2.init(.{ 4, 4 }));
    // Inside — returns the point itself
    try std.testing.expect(box.closestPoint(Vec2.init(.{ 2, 2 })).eql(Vec2.init(.{ 2, 2 })));
    // Outside — clamps
    try std.testing.expect(box.closestPoint(Vec2.init(.{ -1, 5 })).eql(Vec2.init(.{ 0, 4 })));
}

test "Aabb: intersectsRay hit" {
    const box = Aabb3.init(Vec3.init(.{ -1, -1, -1 }), Vec3.init(.{ 1, 1, 1 }));
    const ray = Ray3.init(Vec3.init(.{ -5, 0, 0 }), Vec3.init(.{ 1, 0, 0 }));
    const t = box.intersectsRay(ray) orelse return error.TestUnexpectedResult;
    // Ray hits the box at x=-1, so t=4
    try std.testing.expectEqual(@as(f32, 4), t);
}

test "Aabb: intersectsRay miss" {
    const box = Aabb3.init(Vec3.init(.{ -1, -1, -1 }), Vec3.init(.{ 1, 1, 1 }));
    const ray = Ray3.init(Vec3.init(.{ -5, 5, 0 }), Vec3.init(.{ 1, 0, 0 }));
    try std.testing.expect(box.intersectsRay(ray) == null);
}

test "Aabb: intersectsRay origin inside" {
    const box = Aabb3.init(Vec3.init(.{ -1, -1, -1 }), Vec3.init(.{ 1, 1, 1 }));
    const ray = Ray3.init(Vec3.zero, Vec3.init(.{ 1, 0, 0 }));
    const t = box.intersectsRay(ray) orelse return error.TestUnexpectedResult;
    // tmin should be negative (origin is inside)
    try std.testing.expect(t < 0);
}

test "Sphere: contains" {
    const s = Sphere.init(Vec3.zero, 2);
    try std.testing.expect(s.contains(Vec3.zero));
    try std.testing.expect(s.contains(Vec3.init(.{ 1, 1, 0 })));
    try std.testing.expect(!s.contains(Vec3.init(.{ 2, 2, 0 })));
}

test "Sphere: intersects (sphere-sphere)" {
    const a = Sphere.init(Vec3.zero, 1);
    const b = Sphere.init(Vec3.init(.{ 1.5, 0, 0 }), 1);
    const c = Sphere.init(Vec3.init(.{ 5, 0, 0 }), 1);
    try std.testing.expect(a.intersects(b));
    try std.testing.expect(!a.intersects(c));
}

test "Sphere: intersectsAabb" {
    const s = Sphere.init(Vec3.init(.{ 3, 0, 0 }), 2);
    const hit = Aabb3.init(Vec3.init(.{ 0, 0, 0 }), Vec3.init(.{ 2, 2, 2 }));
    const miss = Aabb3.init(Vec3.init(.{ -5, -5, -5 }), Vec3.init(.{ -3, -3, -3 }));
    try std.testing.expect(s.intersectsAabb(hit));
    try std.testing.expect(!s.intersectsAabb(miss));
}

test "Sphere: intersectsRay hit" {
    const s = Sphere.init(Vec3.zero, 1);
    const ray = Ray3.init(Vec3.init(.{ -5, 0, 0 }), Vec3.init(.{ 1, 0, 0 }));
    const t = s.intersectsRay(ray) orelse return error.TestUnexpectedResult;
    try expectApproxScalar(t, 4);
}

test "Sphere: intersectsRay miss" {
    const s = Sphere.init(Vec3.zero, 1);
    const ray = Ray3.init(Vec3.init(.{ -5, 5, 0 }), Vec3.init(.{ 1, 0, 0 }));
    try std.testing.expect(s.intersectsRay(ray) == null);
}

test "Sphere: merge disjoint" {
    const a = Sphere.init(Vec3.init(.{ 0, 0, 0 }), 1);
    const b = Sphere.init(Vec3.init(.{ 10, 0, 0 }), 1);
    const m = a.merge(b);
    // Merged sphere should contain both
    try std.testing.expect(m.contains(Vec3.init(.{ -1, 0, 0 })));
    try std.testing.expect(m.contains(Vec3.init(.{ 11, 0, 0 })));
    try expectApproxScalar(m.radius, 6);
    try expectApproxScalar(m.center.v[0], 5);
}

test "Sphere: merge contained" {
    const outer = Sphere.init(Vec3.zero, 10);
    const inner = Sphere.init(Vec3.init(.{ 1, 0, 0 }), 2);
    const m = outer.merge(inner);
    // Should return the outer sphere unchanged
    try std.testing.expect(m.center.eql(outer.center));
    try std.testing.expectEqual(m.radius, outer.radius);
}

test "Circle: intersectsRay 2D" {
    const c = Circle.init(Vec2.zero, 1);
    const ray = Ray2.init(Vec2.init(.{ -3, 0 }), Vec2.init(.{ 1, 0 }));
    const t = c.intersectsRay(ray) orelse return error.TestUnexpectedResult;
    try expectApproxScalar(t, 2);
}

test "Ray3: intersectAxisPlane hit" {
    // Ray traveling in +Y from below the Y=5 plane
    const ray = Ray3.init(
        Vec3.init(.{ 1, 0, 2 }),
        Vec3.init(.{ 0, 1, 0 }),
    );
    const t = ray.intersectAxisPlane(1, 5.0) orelse return error.TestUnexpectedResult;
    try expectApproxScalar(t, 5.0);
    const hit = ray.pointAt(t);
    try expectApproxScalar(hit.v[1], 5.0);
}

test "Ray3: intersectAxisPlane behind origin" {
    // Ray traveling in +Y but plane is at Y=-1 (behind)
    const ray = Ray3.init(
        Vec3.init(.{ 0, 0, 0 }),
        Vec3.init(.{ 0, 1, 0 }),
    );
    try std.testing.expect(ray.intersectAxisPlane(1, -1.0) == null);
}

test "Ray3: intersectAxisPlane parallel" {
    // Ray traveling in +X, parallel to Y plane
    const ray = Ray3.init(
        Vec3.init(.{ 0, 0, 0 }),
        Vec3.init(.{ 1, 0, 0 }),
    );
    try std.testing.expect(ray.intersectAxisPlane(1, 5.0) == null);
}

test "Ray3: intersectAxisPlane X axis" {
    const ray = Ray3.init(
        Vec3.init(.{ -3, 0, 0 }),
        Vec3.init(.{ 1, 0, 0 }),
    );
    const t = ray.intersectAxisPlane(0, 2.0) orelse return error.TestUnexpectedResult;
    try expectApproxScalar(t, 5.0);
}
