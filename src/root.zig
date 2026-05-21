//! Public API surface for lil-math.
//!
//! Re-exports all math types under a single namespace so callers can write:
//!
//! ```zig
//! const math = @import("lil-math");
//! const v: math.Vec3 = ...;
//! const r: math.Rot3 = ...;
//! ```
//!
//! ## Type families
//!
//! - **Vectors** — `Vec2/3/4` (f32), `Vec2d/3d/4d` (f64), `Vec2i/3i/4i` (i32), `Vec2u/3u/4u` (u32)
//! - **Matrices** — `Mat2/3/4` (f32), `Mat2d/3d/4d` (f64)
//! - **Rotors** — `Rot2/3` (f32), `Rot2d/3d` (f64); geometric-algebra representation of rotations
//! - **Versors** — `Versor2/3` (f32), `Versor2d/3d` (f64); Pin group (rotations + reflections)
//! - **Mirrors** — `Mirror2/3` (f32), `Mirror2d/3d` (f64); affine reflection across a plane/line
//! - **Geometry** — rays, segments, AABBs, circles/spheres
//! - **Directions** — discrete compass/grid direction enums and bitmask sets
//! - **Interpolation** — `Interp*` animated value types with pluggable easing
//! - **Transforms** — `Transform` (TRS decomposition → Mat4)
//! - **Fixed-point** — `FixedPoint(I, F)`, plus common aliases (`Q16_16`, etc.)
//! - **Grid coordinates** — `GridCoord(n, Scalar)` comptime-generic N-dimensional grid coordinate
//! - **Generic constructors** — `VecType`, `MatType`, `Rotor2Type`, `Rotor3Type`, `Versor2Type`, `Versor3Type`, `MirrorType`, `InterpType`

const linalg = @import("linalg.zig");
const direction = @import("direction.zig");
const halfedge = @import("halfedge.zig");
pub const interp = @import("interp.zig");
const fixed_point = @import("fixed_point.zig");
pub const fixed_trig = @import("fixed_trig.zig");

pub const Vec2 = linalg.VecType(2, f32);
pub const Vec3 = linalg.VecType(3, f32);
pub const Vec4 = linalg.VecType(4, f32);

pub const Vec2d = linalg.VecType(2, f64);
pub const Vec3d = linalg.VecType(3, f64);
pub const Vec4d = linalg.VecType(4, f64);

pub const Vec2i = linalg.VecType(2, i32);
pub const Vec3i = linalg.VecType(3, i32);
pub const Vec4i = linalg.VecType(4, i32);

pub const Vec2u = linalg.VecType(2, u32);
pub const Vec3u = linalg.VecType(3, u32);
pub const Vec4u = linalg.VecType(4, u32);

pub const Mat2 = linalg.MatType(2, f32);
pub const Mat3 = linalg.MatType(3, f32);
pub const Mat4 = linalg.MatType(4, f32);

pub const Mat2d = linalg.MatType(2, f64);
pub const Mat3d = linalg.MatType(3, f64);
pub const Mat4d = linalg.MatType(4, f64);

pub const Rot2 = linalg.Rotor2Type(f32);
pub const Rot3 = linalg.Rotor3Type(f32);

pub const Rot2d = linalg.Rotor2Type(f64);
pub const Rot3d = linalg.Rotor3Type(f64);

const geometry = @import("geometry.zig");

pub const Ray2 = geometry.RayType(2, f32);
pub const Ray3 = geometry.RayType(3, f32);

pub const Ray2d = geometry.RayType(2, f64);
pub const Ray3d = geometry.RayType(3, f64);

pub const Segment2 = geometry.SegmentType(2, f32);
pub const Segment3 = geometry.SegmentType(3, f32);

pub const Segment2d = geometry.SegmentType(2, f64);
pub const Segment3d = geometry.SegmentType(3, f64);

pub const Aabb2 = geometry.AabbType(2, f32);
pub const Aabb3 = geometry.AabbType(3, f32);

pub const Aabb2d = geometry.AabbType(2, f64);
pub const Aabb3d = geometry.AabbType(3, f64);

pub const Circle = geometry.SphereType(2, f32);
pub const Sphere = geometry.SphereType(3, f32);

pub const Circled = geometry.SphereType(2, f64);
pub const Sphered = geometry.SphereType(3, f64);

pub const Dir4 = direction.Dir4;
pub const Dir8 = direction.Dir8;
pub const Dir6 = direction.Dir6;
pub const Dir26 = direction.Dir26;

pub const DirSet4 = direction.DirSetType(direction.Dir4);
pub const DirSet8 = direction.DirSetType(direction.Dir8);
pub const DirSet6 = direction.DirSetType(direction.Dir6);
pub const DirSet26 = direction.DirSetType(direction.Dir26);

pub const Ori24 = direction.Ori24;

pub const DirSetType = direction.DirSetType;

pub const HalfEdgeMesh = halfedge.HalfEdgeMesh;
pub const HalfEdgeId = halfedge.HalfEdgeId;
pub const VertexId = halfedge.VertexId;
pub const FaceId = halfedge.FaceId;

pub const Interp = interp.Interp;
pub const Interpd = interp.Interpd;
pub const Interp2 = interp.Interp2;
pub const Interp3 = interp.Interp3;
pub const Interp4 = interp.Interp4;
pub const Interp2d = interp.Interp2d;
pub const Interp3d = interp.Interp3d;
pub const Interp4d = interp.Interp4d;
pub const InterpRot2 = interp.InterpRot2;
pub const InterpRot3 = interp.InterpRot3;
pub const InterpRot2d = interp.InterpRot2d;
pub const InterpRot3d = interp.InterpRot3d;
pub const InterpVersor2 = interp.InterpVersor2;
pub const InterpVersor3 = interp.InterpVersor3;
pub const InterpVersor2d = interp.InterpVersor2d;
pub const InterpVersor3d = interp.InterpVersor3d;
pub const Blend = interp.Blend;

const mirror = @import("mirror.zig");

pub const Mirror2 = mirror.MirrorType(2, f32);
pub const Mirror3 = mirror.MirrorType(3, f32);

pub const Mirror2d = mirror.MirrorType(2, f64);
pub const Mirror3d = mirror.MirrorType(3, f64);

pub const MirrorType = mirror.MirrorType;

const transform_mod = @import("transform.zig");
pub const Transform = transform_mod.Transform;

pub const VecType = linalg.VecType;
pub const MatType = linalg.MatType;
pub const Rotor2Type = linalg.Rotor2Type;
pub const Rotor3Type = linalg.Rotor3Type;

pub const Versor2 = linalg.Versor2Type(f32);
pub const Versor3 = linalg.Versor3Type(f32);

pub const Versor2d = linalg.Versor2Type(f64);
pub const Versor3d = linalg.Versor3Type(f64);

pub const Versor2Type = linalg.Versor2Type;
pub const Versor3Type = linalg.Versor3Type;
pub const Tolerance = linalg.Tolerance;
pub const Approx = linalg.Approx;
pub const RayType = geometry.RayType;
pub const SegmentType = geometry.SegmentType;
pub const AabbType = geometry.AabbType;
pub const SphereType = geometry.SphereType;
pub const InterpType = interp.InterpType;

pub const FixedPoint = fixed_point.FixedPoint;
pub const Q8_8 = FixedPoint(8, 8);
pub const Q16_16 = FixedPoint(16, 16);
pub const Q24_8 = FixedPoint(24, 8);
pub const Q8_24 = FixedPoint(8, 24);
pub const Q32_32 = FixedPoint(32, 32);

/// Noise algorithms and combinators.
pub const noise = @import("noise.zig");
pub const NoiseType = noise.NoiseType;
pub const ValueNoise = noise.ValueNoise;
pub const PerlinNoise = noise.PerlinNoise;
pub const SimplexNoise = noise.SimplexNoise;
pub const WhiteNoise = noise.WhiteNoise;
pub const BlueNoise = noise.BlueNoise;
pub const FbmOptions = noise.FbmOptions;
pub const GridFillOptions = noise.GridFillOptions;

test {
    _ = @import("direction.zig");
    _ = @import("fixed_point.zig");
    _ = @import("fixed_trig.zig");
    _ = @import("geometry.zig");
    _ = @import("halfedge.zig");
    _ = @import("interp.zig");
    _ = @import("linalg.zig");
    _ = @import("mirror.zig");
    _ = @import("noise.zig");
    _ = @import("transform.zig");
}
