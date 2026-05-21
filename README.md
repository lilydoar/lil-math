# lil-math

A Zig math library for games and simulations.

## Contents

- **Vectors** — `Vec2/3/4` (f32), `Vec2d/3d/4d` (f64), `Vec2i/3i/4i` (i32), `Vec2u/3u/4u` (u32)
- **Matrices** — `Mat2/3/4` (f32), `Mat2d/3d/4d` (f64)
- **Rotors** — `Rot2/3` (f32/f64); geometric-algebra representation of rotations
- **Versors** — `Versor2/3` (f32/f64); Pin group (rotations + reflections)
- **Mirrors** — `Mirror2/3` (f32/f64); affine reflection across a plane/line
- **Geometry** — rays, segments, AABBs, circles/spheres
- **Directions** — discrete compass/grid direction enums and bitmask sets
- **Interpolation** — animated value types with pluggable easing
- **Transforms** — TRS decomposition to Mat4
- **Fixed-point** — `FixedPoint(I, F)` with common aliases (`Q16_16`, etc.)
- **Noise** — Value, Perlin, Simplex, White, Blue noise with fBm and domain warping
- **Half-edge mesh** — connectivity structure for manifold surfaces

## Usage

Add to your `build.zig.zon`:

```zig
.lil_math = .{
    .url = "https://github.com/lilydoar/lil-math/archive/<commit>.tar.gz",
    .hash = "<hash>",
},
```

In `build.zig`:

```zig
const lil_math = b.dependency("lil_math", .{});
exe.root_module.addImport("lil-math", lil_math.module("lil-math"));
```

Then import:

```zig
const math = @import("lil-math");
const v = math.Vec3.init(.{ 1, 2, 3 });
```

## Requirements

Zig 0.16.0+
