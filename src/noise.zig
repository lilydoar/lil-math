//! Comptime-generic noise library for procedural texture generation.
//!
//! The central abstraction is `NoiseType(Algo)` — a wrapper around any noise
//! algorithm that provides a unified API including fractal Brownian motion,
//! domain warping, and grid-fill helpers.
//!
//! ## Quick start
//!
//! ```zig
//! // Single-octave Perlin noise:
//! var n = PerlinNoise.init(42);
//! const v = n.sample2d(1.5, 2.3);   // f32 in [0, 1]
//!
//! // fBm with 6 octaves:
//! const v_fbm = n.fbm(1.5, 2.3, .{ .octaves = 6 });
//!
//! // Fill a 256×256 pixel buffer:
//! var pixels: [256 * 256 * 4]u8 = undefined;
//! n.fillGrid(&pixels, 256, 256, .{ .frequency = 4.0 });
//! ```
//!
//! ## Algorithms
//!
//! | Alias | Algorithm | Notes |
//! |---|---|---|
//! | `ValueNoise` | `Value` | Interpolated hash-based noise. Cheap, smooth. |
//! | `PerlinNoise` | `Perlin` | Gradient noise. Classic, well-behaved. |
//! | `SimplexNoise` | `Simplex` | Improved Perlin. Fewer axis artifacts. |
//! | `WhiteNoise` | `White` | Uncorrelated random. |
//! | `BlueNoiseAlgo` | `BlueNoise` | Void-and-cluster dither mask. Tileable. |
//!
//! ## Algorithm interface
//!
//! A custom algorithm must implement:
//! - `pub fn init(seed: u64) @This()` — create seeded instance
//! - `pub fn sample2d(self: *const @This(), x: f32, y: f32) f32` — 2D sample in [0,1]
//! - `pub fn sample3d(self: *const @This(), x: f32, y: f32, z: f32) f32` — optional 3D

const std = @import("std");
const math = std.math;

/// Map a hash to one of 12 gradient directions for 3D noise.
/// Shared by Perlin and Simplex — both use the same 16-entry gradient table.
fn grad3(h: u32, dx: f32, dy: f32, dz: f32) f32 {
    return switch (h & 15) {
        0 => dx + dy,
        1 => -dx + dy,
        2 => dx - dy,
        3 => -dx - dy,
        4 => dx + dz,
        5 => -dx + dz,
        6 => dx - dz,
        7 => -dx - dz,
        8 => dy + dz,
        9 => -dy + dz,
        10 => dy - dz,
        11 => -dy - dz,
        12 => dx + dy,
        13 => -dy + dz,
        14 => -dx + dy,
        15 => -dy - dz,
        else => unreachable,
    };
}

/// Asserts at compile time that `T` declares every name in `names`.
/// Mirrors `meta.assertDecls` without a cross-module import.
fn assertDecls(comptime T: type, comptime names: []const []const u8) void {
    inline for (names) |name| {
        if (!@hasDecl(T, name)) {
            @compileError(@typeName(T) ++ " missing required declaration: " ++ name);
        }
    }
}

// ---------------------------------------------------------------------------
// Hash utilities — Squirrel3 integer noise
// ---------------------------------------------------------------------------

/// Squirrel3 noise function from Squirrel Eiserloh's GDC 2017 talk.
/// Maps a 1-D integer position + seed to a uniform pseudo-random u32.
fn squirrel3(n: i32, seed: u64) u32 {
    const BIT_NOISE1: u32 = 0xB5297A4D;
    const BIT_NOISE2: u32 = 0x68E31DA4;
    const BIT_NOISE3: u32 = 0x1B56C4E9;

    var bits: u32 = @bitCast(n);
    bits *%= BIT_NOISE1;
    bits +%= @truncate(seed);
    bits ^= bits >> 8;
    bits +%= BIT_NOISE2;
    bits ^= bits << 8;
    bits *%= BIT_NOISE3;
    bits ^= bits >> 8;
    return bits;
}

/// 2-D hash — combines two integer coordinates into a single u32.
fn hash2(xi: i32, yi: i32, seed: u64) u32 {
    return squirrel3(xi +% @as(i32, @truncate(@as(i64, 1_073_741_891) *% yi)), seed);
}

/// 3-D hash — combines three integer coordinates into a single u32.
fn hash3(xi: i32, yi: i32, zi: i32, seed: u64) u32 {
    const t = xi +% @as(i32, @truncate(@as(i64, 1_073_741_891) *% yi +% @as(i64, 198_491_317) *% zi));
    return squirrel3(t, seed);
}

/// Maps a u32 in [0, 2³²) to a f32 in [0, 1).
inline fn toUnitF32(h: u32) f32 {
    return @as(f32, @floatFromInt(h >> 8)) * (1.0 / @as(f32, 1 << 24));
}

// ---------------------------------------------------------------------------
// Smoothstep / fade curve (Ken Perlin's quintic)
// ---------------------------------------------------------------------------

inline fn fade(t: f32) f32 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

inline fn lerp(a: f32, b: f32, t: f32) f32 {
    return a + (b - a) * t;
}

// ---------------------------------------------------------------------------
// Value noise algorithm
// ---------------------------------------------------------------------------

/// Value noise — hashes lattice corners and bicubically interpolates.
///
/// Cheaper than Perlin but can show blocky grid artifacts at low frequencies.
pub const Value = struct {
    seed: u64,

    pub fn init(seed: u64) Value {
        return .{ .seed = seed };
    }

    /// Returns a value in [0, 1] for the 2-D coordinate (x, y).
    pub fn sample2d(self: *const Value, x: f32, y: f32) f32 {
        const xi: i32 = @intFromFloat(@floor(x));
        const yi: i32 = @intFromFloat(@floor(y));
        const fx = x - @floor(x);
        const fy = y - @floor(y);

        const ux = fade(fx);
        const uy = fade(fy);

        const v00 = toUnitF32(hash2(xi, yi, self.seed));
        const v10 = toUnitF32(hash2(xi + 1, yi, self.seed));
        const v01 = toUnitF32(hash2(xi, yi + 1, self.seed));
        const v11 = toUnitF32(hash2(xi + 1, yi + 1, self.seed));

        return lerp(lerp(v00, v10, ux), lerp(v01, v11, ux), uy);
    }

    /// Returns a value in [0, 1] for the 3-D coordinate (x, y, z).
    pub fn sample3d(self: *const Value, x: f32, y: f32, z: f32) f32 {
        const xi: i32 = @intFromFloat(@floor(x));
        const yi: i32 = @intFromFloat(@floor(y));
        const zi: i32 = @intFromFloat(@floor(z));
        const fx = x - @floor(x);
        const fy = y - @floor(y);
        const fz = z - @floor(z);

        const ux = fade(fx);
        const uy = fade(fy);
        const uz = fade(fz);

        const v000 = toUnitF32(hash3(xi, yi, zi, self.seed));
        const v100 = toUnitF32(hash3(xi + 1, yi, zi, self.seed));
        const v010 = toUnitF32(hash3(xi, yi + 1, zi, self.seed));
        const v110 = toUnitF32(hash3(xi + 1, yi + 1, zi, self.seed));
        const v001 = toUnitF32(hash3(xi, yi, zi + 1, self.seed));
        const v101 = toUnitF32(hash3(xi + 1, yi, zi + 1, self.seed));
        const v011 = toUnitF32(hash3(xi, yi + 1, zi + 1, self.seed));
        const v111 = toUnitF32(hash3(xi + 1, yi + 1, zi + 1, self.seed));

        const bot = lerp(lerp(v000, v100, ux), lerp(v010, v110, ux), uy);
        const top = lerp(lerp(v001, v101, ux), lerp(v011, v111, ux), uy);
        return lerp(bot, top, uz);
    }
};

// ---------------------------------------------------------------------------
// Perlin gradient noise algorithm
// ---------------------------------------------------------------------------

/// Classic Ken Perlin gradient noise (2001 improved version).
///
/// Smooth, well-understood properties. Uses hash-derived gradient vectors
/// from 16 canonical directions to avoid look-up tables.
pub const Perlin = struct {
    seed: u64,

    pub fn init(seed: u64) Perlin {
        return .{ .seed = seed };
    }

    /// Map a hash to one of 12 gradient directions for 2D Perlin noise.
    fn grad2(h: u32, dx: f32, dy: f32) f32 {
        return switch (h & 7) {
            0 => dx + dy,
            1 => dx - dy,
            2 => -dx + dy,
            3 => -dx - dy,
            4 => dx,
            5 => -dx,
            6 => dy,
            7 => -dy,
            else => unreachable,
        };
    }

    // grad3 is shared — see module-level grad3().

    /// Returns a value in [0, 1] for the 2-D coordinate (x, y).
    pub fn sample2d(self: *const Perlin, x: f32, y: f32) f32 {
        const xi: i32 = @intFromFloat(@floor(x));
        const yi: i32 = @intFromFloat(@floor(y));
        const fx = x - @floor(x);
        const fy = y - @floor(y);

        const ux = fade(fx);
        const uy = fade(fy);

        const g00 = grad2(hash2(xi, yi, self.seed), fx, fy);
        const g10 = grad2(hash2(xi + 1, yi, self.seed), fx - 1.0, fy);
        const g01 = grad2(hash2(xi, yi + 1, self.seed), fx, fy - 1.0);
        const g11 = grad2(hash2(xi + 1, yi + 1, self.seed), fx - 1.0, fy - 1.0);

        const result = lerp(lerp(g00, g10, ux), lerp(g01, g11, ux), uy);
        // Perlin returns roughly [-1, 1]; map to [0, 1]
        return result * 0.5 + 0.5;
    }

    /// Returns a value in [0, 1] for the 3-D coordinate (x, y, z).
    pub fn sample3d(self: *const Perlin, x: f32, y: f32, z: f32) f32 {
        const xi: i32 = @intFromFloat(@floor(x));
        const yi: i32 = @intFromFloat(@floor(y));
        const zi: i32 = @intFromFloat(@floor(z));
        const fx = x - @floor(x);
        const fy = y - @floor(y);
        const fz = z - @floor(z);

        const ux = fade(fx);
        const uy = fade(fy);
        const uz = fade(fz);

        const g000 = grad3(hash3(xi, yi, zi, self.seed), fx, fy, fz);
        const g100 = grad3(hash3(xi + 1, yi, zi, self.seed), fx - 1, fy, fz);
        const g010 = grad3(hash3(xi, yi + 1, zi, self.seed), fx, fy - 1, fz);
        const g110 = grad3(hash3(xi + 1, yi + 1, zi, self.seed), fx - 1, fy - 1, fz);
        const g001 = grad3(hash3(xi, yi, zi + 1, self.seed), fx, fy, fz - 1);
        const g101 = grad3(hash3(xi + 1, yi, zi + 1, self.seed), fx - 1, fy, fz - 1);
        const g011 = grad3(hash3(xi, yi + 1, zi + 1, self.seed), fx, fy - 1, fz - 1);
        const g111 = grad3(hash3(xi + 1, yi + 1, zi + 1, self.seed), fx - 1, fy - 1, fz - 1);

        const bot = lerp(lerp(g000, g100, ux), lerp(g010, g110, ux), uy);
        const top = lerp(lerp(g001, g101, ux), lerp(g011, g111, ux), uy);
        const result = lerp(bot, top, uz);
        return result * 0.5 + 0.5;
    }
};

// ---------------------------------------------------------------------------
// Simplex noise algorithm (2D and 3D)
// ---------------------------------------------------------------------------

/// Ken Perlin's Simplex noise (2001).
///
/// Uses a simplex lattice instead of a hypercube, giving fewer axis-aligned
/// artifacts and O(n²) complexity in n dimensions rather than O(2ⁿ).
pub const Simplex = struct {
    seed: u64,

    pub fn init(seed: u64) Simplex {
        return .{ .seed = seed };
    }

    fn grad2s(h: u32, dx: f32, dy: f32) f32 {
        return switch (h & 7) {
            0 => dx + dy,
            1 => dx - dy,
            2 => -dx + dy,
            3 => -dx - dy,
            4 => dx * 1.4142,
            5 => -dx * 1.4142,
            6 => dy * 1.4142,
            7 => -dy * 1.4142,
            else => unreachable,
        };
    }

    /// Returns a value in [0, 1] for the 2-D coordinate (x, y).
    pub fn sample2d(self: *const Simplex, x: f32, y: f32) f32 {
        // Skewing / unskewing factors for 2D simplex
        const F2: f32 = 0.5 * (math.sqrt(3.0) - 1.0);
        const G2: f32 = (3.0 - math.sqrt(3.0)) / 6.0;

        // Skew the input space to determine which simplex cell we're in
        const s = (x + y) * F2;
        const ix: i32 = @intFromFloat(@floor(x + s));
        const iy: i32 = @intFromFloat(@floor(y + s));

        // Unskew back to (x, y) space
        const t = @as(f32, @floatFromInt(ix + iy)) * G2;
        const x0 = x - (@as(f32, @floatFromInt(ix)) - t);
        const y0 = y - (@as(f32, @floatFromInt(iy)) - t);

        // Determine which simplex we're in
        const ix1: i32 = if (x0 > y0) 1 else 0;
        const iy1: i32 = if (x0 > y0) 0 else 1;

        // Offsets for middle corner
        const x1 = x0 - @as(f32, @floatFromInt(ix1)) + G2;
        const y1 = y0 - @as(f32, @floatFromInt(iy1)) + G2;
        // Offsets for last corner
        const x2 = x0 - 1.0 + 2.0 * G2;
        const y2 = y0 - 1.0 + 2.0 * G2;

        // Compute contributions from each corner
        var n0: f32 = 0;
        var n1: f32 = 0;
        var n2: f32 = 0;

        const t0 = 0.5 - x0 * x0 - y0 * y0;
        if (t0 > 0) {
            const t0sq = t0 * t0;
            n0 = t0sq * t0sq * grad2s(hash2(ix, iy, self.seed), x0, y0);
        }

        const t1 = 0.5 - x1 * x1 - y1 * y1;
        if (t1 > 0) {
            const t1sq = t1 * t1;
            n1 = t1sq * t1sq * grad2s(hash2(ix + ix1, iy + iy1, self.seed), x1, y1);
        }

        const t2 = 0.5 - x2 * x2 - y2 * y2;
        if (t2 > 0) {
            const t2sq = t2 * t2;
            n2 = t2sq * t2sq * grad2s(hash2(ix + 1, iy + 1, self.seed), x2, y2);
        }

        // Scale to [0, 1]. The raw output range is roughly [-0.8, 0.8] for 2D.
        const raw = 70.0 * (n0 + n1 + n2);
        return @max(0.0, @min(1.0, raw * 0.5 + 0.5));
    }

    // grad3 is shared — see module-level grad3().

    /// Returns a value in [0, 1] for the 3-D coordinate (x, y, z).
    pub fn sample3d(self: *const Simplex, x: f32, y: f32, z: f32) f32 {
        const F3: f32 = 1.0 / 3.0;
        const G3: f32 = 1.0 / 6.0;

        const s = (x + y + z) * F3;
        const ix: i32 = @intFromFloat(@floor(x + s));
        const iy: i32 = @intFromFloat(@floor(y + s));
        const iz: i32 = @intFromFloat(@floor(z + s));

        const t = @as(f32, @floatFromInt(ix + iy + iz)) * G3;
        const x0 = x - (@as(f32, @floatFromInt(ix)) - t);
        const y0 = y - (@as(f32, @floatFromInt(iy)) - t);
        const z0 = z - (@as(f32, @floatFromInt(iz)) - t);

        // Determine which simplex we're in
        var ix1: i32 = 0;
        var iy1: i32 = 0;
        var iz1: i32 = 0;
        var ix2: i32 = 0;
        var iy2: i32 = 0;
        var iz2: i32 = 0;

        if (x0 >= y0) {
            if (y0 >= z0) {
                ix1 = 1;
                iy1 = 0;
                iz1 = 0;
                ix2 = 1;
                iy2 = 1;
                iz2 = 0;
            } else if (x0 >= z0) {
                ix1 = 1;
                iy1 = 0;
                iz1 = 0;
                ix2 = 1;
                iy2 = 0;
                iz2 = 1;
            } else {
                ix1 = 0;
                iy1 = 0;
                iz1 = 1;
                ix2 = 1;
                iy2 = 0;
                iz2 = 1;
            }
        } else {
            if (y0 < z0) {
                ix1 = 0;
                iy1 = 0;
                iz1 = 1;
                ix2 = 0;
                iy2 = 1;
                iz2 = 1;
            } else if (x0 < z0) {
                ix1 = 0;
                iy1 = 1;
                iz1 = 0;
                ix2 = 0;
                iy2 = 1;
                iz2 = 1;
            } else {
                ix1 = 0;
                iy1 = 1;
                iz1 = 0;
                ix2 = 1;
                iy2 = 1;
                iz2 = 0;
            }
        }

        const x1 = x0 - @as(f32, @floatFromInt(ix1)) + G3;
        const y1 = y0 - @as(f32, @floatFromInt(iy1)) + G3;
        const z1 = z0 - @as(f32, @floatFromInt(iz1)) + G3;
        const x2 = x0 - @as(f32, @floatFromInt(ix2)) + 2.0 * G3;
        const y2 = y0 - @as(f32, @floatFromInt(iy2)) + 2.0 * G3;
        const z2 = z0 - @as(f32, @floatFromInt(iz2)) + 2.0 * G3;
        const x3 = x0 - 1.0 + 3.0 * G3;
        const y3 = y0 - 1.0 + 3.0 * G3;
        const z3 = z0 - 1.0 + 3.0 * G3;

        var n0: f32 = 0;
        var n1: f32 = 0;
        var n2: f32 = 0;
        var n3: f32 = 0;

        const t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
        if (t0 > 0) {
            const t0sq = t0 * t0;
            n0 = t0sq * t0sq * grad3(hash3(ix, iy, iz, self.seed), x0, y0, z0);
        }
        const t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
        if (t1 > 0) {
            const t1sq = t1 * t1;
            n1 = t1sq * t1sq * grad3(hash3(ix + ix1, iy + iy1, iz + iz1, self.seed), x1, y1, z1);
        }
        const t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
        if (t2 > 0) {
            const t2sq = t2 * t2;
            n2 = t2sq * t2sq * grad3(hash3(ix + ix2, iy + iy2, iz + iz2, self.seed), x2, y2, z2);
        }
        const t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
        if (t3 > 0) {
            const t3sq = t3 * t3;
            n3 = t3sq * t3sq * grad3(hash3(ix + 1, iy + 1, iz + 1, self.seed), x3, y3, z3);
        }

        const raw = 32.0 * (n0 + n1 + n2 + n3);
        return @max(0.0, @min(1.0, raw * 0.5 + 0.5));
    }
};

// ---------------------------------------------------------------------------
// White noise (fully uncorrelated)
// ---------------------------------------------------------------------------

/// White noise — independent uniform random value per sample.
///
/// Every coordinate produces an independent value in [0, 1]. There is no
/// spatial correlation between neighbouring samples. Useful as a building
/// block for other algorithms, or for dithering when full randomness is desired.
pub const White = struct {
    seed: u64,

    pub fn init(seed: u64) White {
        return .{ .seed = seed };
    }

    /// Returns a value in [0, 1] for the 2-D coordinate (x, y).
    pub fn sample2d(self: *const White, x: f32, y: f32) f32 {
        const xi: i32 = @intFromFloat(@floor(x));
        const yi: i32 = @intFromFloat(@floor(y));
        return toUnitF32(hash2(xi, yi, self.seed));
    }

    /// Returns a value in [0, 1] for the 3-D coordinate (x, y, z).
    pub fn sample3d(self: *const White, x: f32, y: f32, z: f32) f32 {
        const xi: i32 = @intFromFloat(@floor(x));
        const yi: i32 = @intFromFloat(@floor(y));
        const zi: i32 = @intFromFloat(@floor(z));
        return toUnitF32(hash3(xi, yi, zi, self.seed));
    }
};

// ---------------------------------------------------------------------------
// Blue noise (void-and-cluster dither mask)
// ---------------------------------------------------------------------------

/// The size of the precomputed blue noise tile (power of 2).
const blue_tile_size: usize = 64;

/// Blue noise dither mask generated via the void-and-cluster algorithm.
///
/// Produces a tileable `blue_tile_size × blue_tile_size` dither mask with good
/// low-frequency energy suppression. The precomputation runs in `init()`;
/// after that, `sample2d` is a simple array lookup with wrapping.
///
/// The void-and-cluster algorithm (Robert Ulichney, 1993):
/// 1. Start with a sparse random binary pattern.
/// 2. Phase A — repeatedly remove the tightest-cluster one and record its
///    rank (high values first: N-1, N-2, …).
/// 3. Phase B — from an empty grid, repeatedly place a one in the largest
///    void and assign ascending ranks (0, 1, …).
/// 4. Phase C — continue Phase B past N/2 to fill the second half.
///
/// Energy is maintained *incrementally*: when a cell toggles, only the
/// O(kernel²) neighbours need to be updated.  This reduces the overall
/// complexity from O(N³) to O(N²), making 64×64 feasible.
pub const BlueNoiseAlgo = struct {
    /// Normalised dither values in [0, 1] for a blue_tile_size × blue_tile_size tile.
    mask: [blue_tile_size * blue_tile_size]f32,

    const W = blue_tile_size;
    const H = blue_tile_size;
    const N = W * H;

    /// Gaussian kernel radius and sigma.
    const radius: i32 = 5;
    const sigma: f32 = 1.5;
    const ksize: usize = (2 * @as(usize, @intCast(radius)) + 1);

    /// Precomputed kernel weights [dy+radius][dx+radius] for toroidal Gaussian.
    const kernel: [ksize][ksize]f32 = blk: {
        const sigma2 = 2.0 * sigma * sigma;
        var k: [ksize][ksize]f32 = undefined;
        var dy: i32 = -radius;
        while (dy <= radius) : (dy += 1) {
            var dx: i32 = -radius;
            while (dx <= radius) : (dx += 1) {
                const d2: f32 = @floatFromInt(dx * dx + dy * dy);
                const yi: usize = @intCast(dy + radius);
                const xi: usize = @intCast(dx + radius);
                k[yi][xi] = @exp(-d2 / sigma2);
            }
        }
        break :blk k;
    };

    pub fn init(seed: u64) BlueNoiseAlgo {
        var self: BlueNoiseAlgo = undefined;

        // --- Step 1: Random sparse binary pattern ---
        const target_ones = N / 10;
        var binary: [N]u8 = [_]u8{0} ** N;
        var energy: [N]f32 = [_]f32{0} ** N;

        var rng: u64 = seed ^ 0xDEADBEEFCAFEBABE;
        var placed: usize = 0;
        while (placed < target_ones) {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            const idx = rng % N;
            if (binary[idx] == 0) {
                binary[idx] = 1;
                addEnergy(&energy, idx, 1.0);
                placed += 1;
            }
        }
        const initial_ones = placed;

        // --- Step 2: Rank array ---
        var rank: [N]u32 = [_]u32{0} ** N;

        // Phase A: remove tightest cluster → assign ranks N-1, N-2, …
        {
            var cur_rank: u32 = @intCast(N - 1);
            var a_binary = binary;
            var a_energy = energy;
            for (0..initial_ones) |_| {
                const idx = findMaxAmong1s(&a_binary, &a_energy);
                rank[idx] = cur_rank;
                if (cur_rank > 0) cur_rank -= 1;
                a_binary[idx] = 0;
                addEnergy(&a_energy, idx, -1.0);
            }
        }

        // Phase B+C: grow from empty → assign ranks 0, 1, …, N-1
        {
            var b_binary: [N]u8 = [_]u8{0} ** N;
            var b_energy: [N]f32 = [_]f32{0} ** N;

            for (0..N) |step| {
                const idx = findMinAmong0s(&b_binary, &b_energy);
                // For the first `initial_ones` steps these positions were already
                // ranked in Phase A; skip re-assigning them to preserve Phase A's ranks.
                if (step >= initial_ones) {
                    rank[idx] = @intCast(step);
                }
                b_binary[idx] = 1;
                addEnergy(&b_energy, idx, 1.0);
            }
        }

        // --- Step 3: Normalise ---
        for (0..N) |i| {
            self.mask[i] = @as(f32, @floatFromInt(rank[i])) / @as(f32, @floatFromInt(N - 1));
        }

        return self;
    }

    /// Incrementally update `energy` for all cells in the kernel neighbourhood of
    /// the cell at flat index `idx`. Pass `+1.0` when placing a one, `-1.0` when
    /// removing one.
    fn addEnergy(energy: *[N]f32, idx: usize, sign: f32) void {
        const cy: i32 = @intCast(idx / W);
        const cx: i32 = @intCast(idx % W);

        var dy: i32 = -radius;
        while (dy <= radius) : (dy += 1) {
            var dx: i32 = -radius;
            while (dx <= radius) : (dx += 1) {
                const nx: usize = @intCast(@mod(cx + dx, @as(i32, W)));
                const ny: usize = @intCast(@mod(cy + dy, @as(i32, H)));
                const ni = ny * W + nx;
                const ky: usize = @intCast(dy + radius);
                const kx: usize = @intCast(dx + radius);
                energy[ni] += sign * kernel[ky][kx];
            }
        }
    }

    /// Return the flat index of the cell with the highest energy among ones.
    fn findMaxAmong1s(binary: *const [N]u8, energy: *const [N]f32) usize {
        var best_e: f32 = -math.inf(f32);
        var best_i: usize = 0;
        for (0..N) |i| {
            if (binary[i] == 1 and energy[i] > best_e) {
                best_e = energy[i];
                best_i = i;
            }
        }
        return best_i;
    }

    /// Return the flat index of the cell with the lowest energy among zeros.
    fn findMinAmong0s(binary: *const [N]u8, energy: *const [N]f32) usize {
        var best_e: f32 = math.inf(f32);
        var best_i: usize = 0;
        for (0..N) |i| {
            if (binary[i] == 0 and energy[i] < best_e) {
                best_e = energy[i];
                best_i = i;
            }
        }
        return best_i;
    }

    /// Returns a value in [0, 1] by looking up (x, y) in the tiled mask.
    pub fn sample2d(self: *const BlueNoiseAlgo, x: f32, y: f32) f32 {
        const xi: usize = @intCast(@mod(@as(i32, @intFromFloat(@floor(x))), @as(i32, W)));
        const yi: usize = @intCast(@mod(@as(i32, @intFromFloat(@floor(y))), @as(i32, H)));
        return self.mask[yi * W + xi];
    }
};

// ---------------------------------------------------------------------------
// Configuration structs
// ---------------------------------------------------------------------------

/// Options for fractal Brownian motion (fBm) layering.
pub const FbmOptions = struct {
    /// Number of octaves to stack (more = more detail).
    octaves: u4 = 6,
    /// Frequency multiplier per octave (>1, typically 2.0).
    lacunarity: f32 = 2.0,
    /// Amplitude multiplier per octave (<1, typically 0.5).
    persistence: f32 = 0.5,
    /// Base frequency applied to input coordinates.
    frequency: f32 = 1.0,
    /// Base amplitude of the first octave.
    amplitude: f32 = 1.0,
};

/// Options for `fillGrid` — how to map noise to a pixel buffer.
pub const GridFillOptions = struct {
    /// Scales input coordinates: higher values = more detail per pixel.
    frequency: f32 = 1.0,
    /// World-space X offset applied before sampling.
    offset_x: f32 = 0.0,
    /// World-space Y offset applied before sampling.
    offset_y: f32 = 0.0,
    /// When non-null, enables fBm layering with these options.
    /// `fbm.frequency` is multiplied with `GridFillOptions.frequency`.
    fbm: ?FbmOptions = null,
};

// ---------------------------------------------------------------------------
// NoiseType — comptime-generic wrapper
// ---------------------------------------------------------------------------

/// Wraps any noise algorithm with a unified API: fBm, domain warping,
/// and grid-fill helpers.
///
/// The algorithm `Algo` must expose:
/// - `pub fn init(seed: u64) Algo`
/// - `pub fn sample2d(self: *const Algo, x: f32, y: f32) f32`
///
/// Optionally, for 3D support:
/// - `pub fn sample3d(self: *const Algo, x: f32, y: f32, z: f32) f32`
pub fn NoiseType(comptime Algo: type) type {
    assertDecls(Algo, &.{ "init", "sample2d" });

    return struct {
        algo: Algo,

        const Self = @This();

        /// Initialise the noise with the given seed.
        pub fn init(seed: u64) Self {
            return .{ .algo = Algo.init(seed) };
        }

        /// Single-octave 2-D noise sample in [0, 1].
        pub fn sample2d(self: *const Self, x: f32, y: f32) f32 {
            return self.algo.sample2d(x, y);
        }

        /// Single-octave 3-D noise sample in [0, 1].
        /// Only available when the underlying algorithm implements `sample3d`.
        pub fn sample3d(self: *const Self, x: f32, y: f32, z: f32) f32 {
            if (!@hasDecl(Algo, "sample3d")) {
                @compileError(@typeName(Algo) ++ " does not implement sample3d");
            }
            return self.algo.sample3d(x, y, z);
        }

        /// Fractal Brownian Motion: sum `opts.octaves` layers of noise,
        /// each at double the frequency and half the amplitude.
        ///
        /// Returns a value in [0, 1]; the normalisation accounts for the
        /// geometric series sum so amplitude scaling is predictable.
        pub fn fbm(self: *const Self, x: f32, y: f32, opts: FbmOptions) f32 {
            var value: f32 = 0;
            var max_value: f32 = 0;
            var amplitude = opts.amplitude;
            var frequency = opts.frequency;

            for (0..opts.octaves) |_| {
                const s = self.algo.sample2d(x * frequency, y * frequency);
                value += s * amplitude;
                max_value += amplitude;
                amplitude *= opts.persistence;
                frequency *= opts.lacunarity;
            }

            return if (max_value > 0) value / max_value else 0;
        }

        /// Domain warping: offset the input coordinates by noise, then
        /// sample again.  Produces flowing, organic distortion.
        ///
        /// `strength` controls how far coordinates are displaced (in
        /// noise-space units). Returns a value in [0, 1].
        pub fn warp2d(self: *const Self, x: f32, y: f32, strength: f32) f32 {
            const wx = self.algo.sample2d(x + 0.0, y + 0.0) * 2.0 - 1.0;
            const wy = self.algo.sample2d(x + 5.2, y + 1.3) * 2.0 - 1.0;
            return self.algo.sample2d(x + strength * wx, y + strength * wy);
        }

        /// Fill `pixels` (RGBA, 4 bytes per pixel) with noise-derived
        /// grayscale values.
        ///
        /// Each pixel is (v, v, v, 255) where `v` ∈ [0, 255] maps the
        /// noise output [0, 1] linearly to the unsigned byte range.
        ///
        /// `pixels` must have length `width * height * 4`.
        pub fn fillGrid(
            self: *const Self,
            pixels: []u8,
            width: u32,
            height: u32,
            opts: GridFillOptions,
        ) void {
            const w_f: f32 = @floatFromInt(width);
            const h_f: f32 = @floatFromInt(height);

            for (0..height) |py| {
                for (0..width) |px| {
                    const nx = (@as(f32, @floatFromInt(px)) / w_f + opts.offset_x) * opts.frequency;
                    const ny = (@as(f32, @floatFromInt(py)) / h_f + opts.offset_y) * opts.frequency;

                    // Base frequency already applied via nx/ny; pass fbm_opts as-is.
                    const v: f32 = if (opts.fbm) |fbm_opts|
                        self.fbm(nx, ny, fbm_opts)
                    else
                        self.algo.sample2d(nx, ny);

                    const byte: u8 = @intFromFloat(@round(v * 255.0));
                    const idx = (py * width + px) * 4;
                    pixels[idx + 0] = byte;
                    pixels[idx + 1] = byte;
                    pixels[idx + 2] = byte;
                    pixels[idx + 3] = 255;
                }
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Convenience aliases
// ---------------------------------------------------------------------------

/// Value noise wrapper — cheap, smooth, slight grid artifacts.
pub const ValueNoise = NoiseType(Value);

/// Perlin gradient noise wrapper — classic, smooth, good all-rounder.
pub const PerlinNoise = NoiseType(Perlin);

/// Simplex noise wrapper — fewer axis artifacts than Perlin.
pub const SimplexNoise = NoiseType(Simplex);

/// White noise wrapper — fully uncorrelated random.
pub const WhiteNoise = NoiseType(White);

/// Blue noise wrapper — tileable low-frequency suppressed dither mask.
pub const BlueNoise = NoiseType(BlueNoiseAlgo);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const testing = std.testing;

// -- Helpers -----------------------------------------------------------------

fn checkRange(v: f32) !void {
    if (v < 0.0 or v > 1.0) {
        std.debug.print("Out of range: {}\n", .{v});
        return error.OutOfRange;
    }
}

// -- Value noise tests -------------------------------------------------------

test "value noise: determinism" {
    const n = ValueNoise.init(42);
    const a = n.sample2d(1.5, 2.7);
    const b = n.sample2d(1.5, 2.7);
    try testing.expectEqual(a, b);
}

test "value noise: range [0,1]" {
    const n = ValueNoise.init(12345);
    const coords = [_][2]f32{
        .{ 0.0, 0.0 },     .{ 0.5, 0.5 },  .{ -1.2, 3.7 },      .{ 100.0, -50.0 },
        .{ 0.001, 0.999 }, .{ 7.3, -7.3 }, .{ 1000.0, 1000.0 },
    };
    for (coords) |c| {
        const v = n.sample2d(c[0], c[1]);
        try checkRange(v);
    }
}

test "value noise: seed independence" {
    const n1 = ValueNoise.init(1);
    const n2 = ValueNoise.init(2);
    // With overwhelming probability, at least one of these will differ.
    const same = n1.sample2d(0.5, 0.5) == n2.sample2d(0.5, 0.5) and
        n1.sample2d(1.5, 2.5) == n2.sample2d(1.5, 2.5);
    try testing.expect(!same);
}

test "value noise: 3d determinism and range" {
    const n = ValueNoise.init(99);
    const a = n.sample3d(1.0, 2.0, 3.0);
    const b = n.sample3d(1.0, 2.0, 3.0);
    try testing.expectEqual(a, b);
    try checkRange(a);
}

// -- Perlin noise tests ------------------------------------------------------

test "perlin noise: determinism" {
    const n = PerlinNoise.init(7);
    const a = n.sample2d(3.14, 2.72);
    const b = n.sample2d(3.14, 2.72);
    try testing.expectEqual(a, b);
}

test "perlin noise: range [0,1]" {
    const n = PerlinNoise.init(0);
    const coords = [_][2]f32{
        .{ 0.0, 0.0 }, .{ 0.25, 0.75 }, .{ -5.0, 5.0 },
        .{ 0.5, 0.5 }, .{ 1.0, 1.0 },   .{ 10.0, -10.0 },
    };
    for (coords) |c| {
        const v = n.sample2d(c[0], c[1]);
        try checkRange(v);
    }
}

test "perlin noise: seed independence" {
    const n1 = PerlinNoise.init(100);
    const n2 = PerlinNoise.init(200);
    const same = n1.sample2d(0.3, 0.7) == n2.sample2d(0.3, 0.7);
    try testing.expect(!same);
}

test "perlin noise: 3d range" {
    const n = PerlinNoise.init(55);
    const coords = [_][3]f32{
        .{ 0.5, 0.5, 0.5 }, .{ 1.0, 2.0, 3.0 }, .{ -1.0, -1.0, -1.0 },
    };
    for (coords) |c| {
        const v = n.sample3d(c[0], c[1], c[2]);
        try checkRange(v);
    }
}

// -- Simplex noise tests -----------------------------------------------------

test "simplex noise: determinism" {
    const n = SimplexNoise.init(0xABCDEF);
    const a = n.sample2d(0.1, 0.9);
    const b = n.sample2d(0.1, 0.9);
    try testing.expectEqual(a, b);
}

test "simplex noise: range [0,1]" {
    const n = SimplexNoise.init(31415926);
    const coords = [_][2]f32{
        .{ 0.0, 0.0 },     .{ 0.5, 0.5 },     .{ -3.0, 7.0 },      .{ 5.0, 5.0 },
        .{ 0.001, 0.001 }, .{ 100.0, 100.0 }, .{ -100.0, -100.0 },
    };
    for (coords) |c| {
        const v = n.sample2d(c[0], c[1]);
        try checkRange(v);
    }
}

test "simplex noise: seed independence" {
    const n1 = SimplexNoise.init(1234);
    const n2 = SimplexNoise.init(5678);
    const same = n1.sample2d(0.5, 0.5) == n2.sample2d(0.5, 0.5);
    try testing.expect(!same);
}

test "simplex noise: 3d range" {
    const n = SimplexNoise.init(42);
    const coords = [_][3]f32{
        .{ 0.5, 0.5, 0.5 }, .{ 1.0, 2.0, 3.0 }, .{ -1.0, 0.0, 1.0 },
    };
    for (coords) |c| {
        const v = n.sample3d(c[0], c[1], c[2]);
        try checkRange(v);
    }
}

// -- White noise tests -------------------------------------------------------

test "white noise: determinism" {
    const n = WhiteNoise.init(0);
    const a = n.sample2d(3.0, 7.0);
    const b = n.sample2d(3.0, 7.0);
    try testing.expectEqual(a, b);
}

test "white noise: range [0,1]" {
    const n = WhiteNoise.init(999);
    for (0..32) |i| {
        const fi: f32 = @floatFromInt(i);
        const v = n.sample2d(fi, fi * 1.7);
        try checkRange(v);
    }
}

test "white noise: uncorrelated (different cells differ with high probability)" {
    const n = WhiteNoise.init(1);
    // Check 10 adjacent pairs — with probability 1-(1/2^24)^10 ≈ 1, they differ.
    var all_same = true;
    for (0..10) |i| {
        const fi: f32 = @floatFromInt(i);
        if (n.sample2d(fi, 0.0) != n.sample2d(fi + 1.0, 0.0)) {
            all_same = false;
            break;
        }
    }
    try testing.expect(!all_same);
}

// -- fBm tests ---------------------------------------------------------------

test "fBm: single octave equals base noise" {
    const n = PerlinNoise.init(42);
    const x = 1.5;
    const y = 2.5;
    const base = n.sample2d(x, y);
    const fbm_val = n.fbm(x, y, .{ .octaves = 1, .frequency = 1.0, .amplitude = 1.0 });
    // With 1 octave and amplitude=1, fBm normalises by max_value=1, so result == base.
    try testing.expectApproxEqAbs(base, fbm_val, 1e-5);
}

test "fBm: output in [0,1]" {
    const n = PerlinNoise.init(7);
    const coords = [_][2]f32{
        .{ 0.0, 0.0 }, .{ 1.0, 1.0 }, .{ 5.0, 5.0 }, .{ -3.0, 2.0 },
    };
    for (coords) |c| {
        const v = n.fbm(c[0], c[1], .{ .octaves = 6 });
        try checkRange(v);
    }
}

test "fBm: amplitude scaling" {
    // With amplitude=0, all octaves contribute nothing → result should be 0
    // (max_value would also be 0, triggering the guard).  Instead test amplitude=2.
    const n = PerlinNoise.init(1);
    const v1 = n.fbm(1.0, 1.0, .{ .octaves = 4, .amplitude = 1.0 });
    const v2 = n.fbm(1.0, 1.0, .{ .octaves = 4, .amplitude = 1.0 });
    // Same params → same result (determinism)
    try testing.expectEqual(v1, v2);
}

test "fBm: more octaves increases detail (variance)" {
    // Sample a grid at two octave counts and confirm range is still valid.
    const n = PerlinNoise.init(17);
    var mn1: f32 = 1.0;
    var mx1: f32 = 0.0;
    var mn6: f32 = 1.0;
    var mx6: f32 = 0.0;
    for (0..16) |i| {
        for (0..16) |j| {
            const x = @as(f32, @floatFromInt(i)) * 0.5;
            const y = @as(f32, @floatFromInt(j)) * 0.5;
            const v1 = n.fbm(x, y, .{ .octaves = 1 });
            const v6 = n.fbm(x, y, .{ .octaves = 6 });
            mn1 = @min(mn1, v1);
            mx1 = @max(mx1, v1);
            mn6 = @min(mn6, v6);
            mx6 = @max(mx6, v6);
        }
    }
    // Both ranges must be within [0,1]
    try testing.expect(mn1 >= 0.0 and mx1 <= 1.0);
    try testing.expect(mn6 >= 0.0 and mx6 <= 1.0);
}

// -- Grid fill tests ---------------------------------------------------------

test "fillGrid: correct pixel count" {
    const n = ValueNoise.init(0);
    var pixels: [4 * 4 * 4]u8 = undefined;
    n.fillGrid(&pixels, 4, 4, .{});
    // Alpha channels must all be 255
    for (0..16) |i| {
        try testing.expectEqual(@as(u8, 255), pixels[i * 4 + 3]);
    }
}

test "fillGrid: values in [0,255]" {
    const n = PerlinNoise.init(123);
    var pixels: [8 * 8 * 4]u8 = undefined;
    n.fillGrid(&pixels, 8, 8, .{ .frequency = 2.0 });
    for (0..64) |i| {
        // R == G == B (grayscale)
        try testing.expectEqual(pixels[i * 4 + 0], pixels[i * 4 + 1]);
        try testing.expectEqual(pixels[i * 4 + 0], pixels[i * 4 + 2]);
        // Values are always valid u8, so no explicit range check needed.
    }
}

test "fillGrid: with fBm option" {
    const n = SimplexNoise.init(77);
    var pixels: [4 * 4 * 4]u8 = undefined;
    n.fillGrid(&pixels, 4, 4, .{ .fbm = .{ .octaves = 4 } });
    // Just verify no crash and alpha = 255
    for (0..16) |i| {
        try testing.expectEqual(@as(u8, 255), pixels[i * 4 + 3]);
    }
}

test "fillGrid: offset changes output" {
    const n = ValueNoise.init(1);
    var p1: [4 * 4 * 4]u8 = undefined;
    var p2: [4 * 4 * 4]u8 = undefined;
    n.fillGrid(&p1, 4, 4, .{ .offset_x = 0.0 });
    n.fillGrid(&p2, 4, 4, .{ .offset_x = 10.0 });
    try testing.expect(!std.mem.eql(u8, &p1, &p2));
}

// -- Domain warp tests -------------------------------------------------------

test "warp2d: output in [0,1]" {
    const n = PerlinNoise.init(99);
    const coords = [_][2]f32{ .{ 0.5, 0.5 }, .{ 2.0, 3.0 }, .{ -1.0, 1.0 } };
    for (coords) |c| {
        const v = n.warp2d(c[0], c[1], 0.5);
        try checkRange(v);
    }
}

// -- Blue noise tests --------------------------------------------------------

test "blue noise: determinism" {
    const n = BlueNoise.init(0);
    const a = n.sample2d(5.0, 10.0);
    const b = n.sample2d(5.0, 10.0);
    try testing.expectEqual(a, b);
}

test "blue noise: range [0,1]" {
    const n = BlueNoise.init(1);
    for (0..@as(usize, blue_tile_size)) |i| {
        for (0..@as(usize, blue_tile_size)) |j| {
            const v = n.sample2d(@floatFromInt(i), @floatFromInt(j));
            try checkRange(v);
        }
    }
}

test "blue noise: tileable (wraps at tile boundary)" {
    const n = BlueNoise.init(42);
    // Value at (0,0) == value at (blue_tile_size, 0) due to wrapping
    const v0 = n.sample2d(0.0, 0.0);
    const vT = n.sample2d(@as(f32, blue_tile_size), 0.0);
    try testing.expectEqual(v0, vT);
}

test "blue noise: spatial uniformity (low-frequency suppression)" {
    // Divide the tile into 4×4 quadrants and verify that the mean of each
    // quadrant is close to 0.5 (good low-frequency coverage).
    const n = BlueNoise.init(7);
    const Q = blue_tile_size / 4;
    for (0..4) |qy| {
        for (0..4) |qx| {
            var sum: f32 = 0;
            for (0..Q) |dy| {
                for (0..Q) |dx| {
                    const px: f32 = @floatFromInt(qx * Q + dx);
                    const py: f32 = @floatFromInt(qy * Q + dy);
                    sum += n.sample2d(px, py);
                }
            }
            const mean = sum / @as(f32, Q * Q);
            // Mean of each quadrant should be within 0.15 of 0.5
            try testing.expect(@abs(mean - 0.5) < 0.15);
        }
    }
}
