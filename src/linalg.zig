//! Linear algebra primitives: vectors, matrices, and rotors.
//!
//! Conventions:
//!   - Right-handed coordinate system
//!   - Column vectors, column-major matrix storage: `m[col][row]`
//!   - Matrix-vector multiply: `M * v` (column on the right)
//!   - Projection depth maps to [0, 1] (Vulkan / Metal / SDL3 GPU)
//!   - Camera looks along -Z in view space; near/far are positive distances
//!
const std = @import("std");
const math = std.math;

/// Generic N-dimensional vector backed by `@Vector(n, Scalar)`.
/// Float-only operations (normalize, lerp, etc.) are gated at comptime.
pub fn VecType(comptime n: comptime_int, comptime Scalar: type) type {
    const is_float = @typeInfo(Scalar) == .float;

    return struct {
        const Self = @This();

        /// Number of components.
        pub const dim = n;

        /// Underlying SIMD storage.
        v: @Vector(n, Scalar),

        /// The zero vector (all components zero).
        pub const zero = Self{ .v = @splat(0) };
        /// The one vector (all components one).
        pub const one = Self{ .v = @splat(1) };

        /// Constructs a vector from a fixed-size array of `n` scalars.
        pub inline fn init(vals: [n]Scalar) Self {
            return .{ .v = vals };
        }

        /// Constructs a vector with every component set to `val`.
        pub inline fn splat(val: Scalar) Self {
            return .{ .v = @splat(val) };
        }

        /// Component-wise addition.
        pub inline fn add(a: Self, b: Self) Self {
            return .{ .v = a.v + b.v };
        }

        /// Component-wise subtraction.
        pub inline fn sub(a: Self, b: Self) Self {
            return .{ .v = a.v - b.v };
        }

        /// Component-wise multiplication (Hadamard product).
        pub inline fn mul(a: Self, b: Self) Self {
            return .{ .v = a.v * b.v };
        }

        /// Component-wise division.
        pub inline fn div(a: Self, b: Self) Self {
            return .{ .v = a.v / b.v };
        }

        /// Scales all components by scalar `s`.
        pub inline fn scale(a: Self, s: Scalar) Self {
            return .{ .v = a.v * @as(@Vector(n, Scalar), @splat(s)) };
        }

        /// Component-wise negation.
        pub inline fn neg(a: Self) Self {
            return .{ .v = -a.v };
        }

        /// For float scalars: returns true if all components differ by at most `e`.
        /// For integer scalars: exact equality (no `e` parameter).
        pub const eql = if (is_float) struct {
            pub fn f(a: Self, b: Self, e: Scalar) bool {
                const eps: @Vector(n, Scalar) = @splat(e);
                return @reduce(.And, @abs(a.v - b.v) <= eps);
            }
        }.f else struct {
            pub fn f(a: Self, b: Self) bool {
                return @reduce(.And, a.v == b.v);
            }
        }.f;

        /// Component-wise absolute value.
        pub inline fn abs(a: Self) Self {
            return .{ .v = @abs(a.v) };
        }

        /// Returns the component-wise sign: `-1`, `0`, or `1` per element.
        pub inline fn sign(a: Self) Self {
            const pos: @Vector(n, Scalar) = @splat(1);
            const neg_one: @Vector(n, Scalar) = @splat(-1);
            const zeros: @Vector(n, Scalar) = @splat(0);
            return .{ .v = @select(Scalar, a.v > zeros, pos, @select(Scalar, a.v < zeros, neg_one, zeros)) };
        }

        /// Component-wise minimum.
        pub inline fn min(a: Self, b: Self) Self {
            return .{ .v = @min(a.v, b.v) };
        }

        /// Component-wise maximum.
        pub inline fn max(a: Self, b: Self) Self {
            return .{ .v = @max(a.v, b.v) };
        }

        /// Rounds each component toward negative infinity.
        /// Only available for float scalar types.
        pub const floor = if (is_float) struct {
            fn f(a: Self) Self {
                return .{ .v = @floor(a.v) };
            }
        }.f else @compileError("floor() requires a float scalar type");

        /// Rounds each component toward positive infinity.
        /// Only available for float scalar types.
        pub const ceil = if (is_float) struct {
            fn f(a: Self) Self {
                return .{ .v = @ceil(a.v) };
            }
        }.f else @compileError("ceil() requires a float scalar type");

        /// Rounds each component to the nearest integer (half away from zero).
        /// Only available for float scalar types.
        pub const round = if (is_float) struct {
            fn f(a: Self) Self {
                return .{ .v = @round(a.v) };
            }
        }.f else @compileError("round() requires a float scalar type");

        /// Returns the dot product (sum of component-wise products).
        pub inline fn dot(a: Self, b: Self) Scalar {
            return @reduce(.Add, a.v * b.v);
        }

        /// Returns the squared Euclidean length. Prefer over `length` when
        /// only comparing magnitudes.
        pub inline fn lenSqr(a: Self) Scalar {
            return @reduce(.Add, a.v * a.v);
        }

        /// Returns the smallest component.
        pub inline fn minElement(a: Self) Scalar {
            return @reduce(.Min, a.v);
        }

        /// Returns the largest component.
        pub inline fn maxElement(a: Self) Scalar {
            return @reduce(.Max, a.v);
        }

        /// Returns the Euclidean length.
        /// Only available for float scalar types.
        pub const len = if (is_float) struct {
            fn f(a: Self) Scalar {
                return @sqrt(@reduce(.Add, a.v * a.v));
            }
        }.f else @compileError("len() requires a float scalar type");

        /// Returns a unit vector in the same direction as `a`.
        /// Returns the zero vector if `a` has zero length.
        /// Only available for float scalar types.
        pub const normalize = if (is_float) struct {
            fn f(a: Self) Self {
                const l = a.len();
                if (l == 0) return .{ .v = @splat(0) };
                return .{ .v = a.v / @as(@Vector(n, Scalar), @splat(l)) };
            }
        }.f else @compileError("normalize() requires a float scalar type");

        /// Linearly interpolates from `a` to `b` at parameter `t`.
        /// `t = 0` returns `a`, `t = 1` returns `b`; values outside `[0, 1]`
        /// extrapolate.
        /// Only available for float scalar types.
        pub const lerp = if (is_float) struct {
            fn f(a: Self, b: Self, t: Scalar) Self {
                const tv: @Vector(n, Scalar) = @splat(t);
                return .{ .v = a.v + (b.v - a.v) * tv };
            }
        }.f else @compileError("lerp() requires a float scalar type");

        /// Projects `a` onto `onto`. Returns the zero vector if `onto` is zero.
        /// Only available for float scalar types.
        pub const project = if (is_float) struct {
            fn f(a: Self, onto: Self) Self {
                const onto_len_sq = @reduce(.Add, onto.v * onto.v);
                if (onto_len_sq == 0) return .{ .v = @splat(0) };
                const s: @Vector(n, Scalar) = @splat(@reduce(.Add, a.v * onto.v) / onto_len_sq);
                return .{ .v = onto.v * s };
            }
        }.f else @compileError("project() requires a float scalar type");

        /// Returns the counter-clockwise perpendicular vector: `(-y, x)`.
        /// Only available for 2D vectors.
        pub const perp = if (n == 2) struct {
            fn f(a: Self) Self {
                return .{ .v = .{ -a.v[1], a.v[0] } };
            }
        }.f else @compileError("perp() is only defined for 2D vectors");

        /// Returns the angle of the vector in radians, in the range `(-π, π]`.
        /// Only available for 2D float vectors.
        pub const angle = if (n == 2 and is_float) struct {
            fn f(a: Self) Scalar {
                return math.atan2(a.v[1], a.v[0]);
            }
        }.f else @compileError("angle() is only defined for 2D float vectors");

        /// 2D: returns the scalar z-component of the 3D cross product (`a.x*b.y - a.y*b.x`).
        /// 3D: returns the cross product vector, perpendicular to both operands.
        /// Only available for 2D and 3D vectors.
        pub const cross = switch (n) {
            2 => struct {
                fn f(a: Self, b: Self) Scalar {
                    return a.v[0] * b.v[1] - a.v[1] * b.v[0];
                }
            }.f,
            3 => struct {
                fn f(a: Self, b: Self) Self {
                    return .{ .v = .{
                        a.v[1] * b.v[2] - a.v[2] * b.v[1],
                        a.v[2] * b.v[0] - a.v[0] * b.v[2],
                        a.v[0] * b.v[1] - a.v[1] * b.v[0],
                    } };
                }
            }.f,
            else => @compileError("cross() is only defined for 2D and 3D vectors"),
        };
    };
}

/// Generic N×N square matrix, column-major: `m[col][row]`.
/// Transform constructors (translation, rotation, scaling, lookAt, perspective, ortho)
/// are gated by dimension and scalar type at comptime.
pub fn MatType(comptime n: comptime_int, comptime Scalar: type) type {
    const ColVec = @Vector(n, Scalar);
    const is_float = @typeInfo(Scalar) == .float;

    return struct {
        const Self = @This();

        /// Matrix dimension (`n` for an n×n matrix).
        pub const dim = n;
        /// Column vector type (`VecType(n, Scalar)`).
        pub const Col = VecType(n, Scalar);

        /// Column-major storage: `m[col][row]`.
        m: [n]ColVec,

        /// The zero matrix (all elements zero).
        pub const zero = Self{ .m = .{@as(ColVec, @splat(0))} ** n };

        /// The identity matrix.
        pub const identity = id: {
            var m: [n]ColVec = .{@as(ColVec, @splat(0))} ** n;
            for (0..n) |i| {
                m[i] = @splat(0);
                m[i][i] = 1;
            }
            break :id Self{ .m = m };
        };

        /// Constructs a matrix from an array of `n` column vectors.
        pub inline fn fromCols(cols: [n]Col) Self {
            var m: [n]ColVec = undefined;
            inline for (0..n) |c| {
                m[c] = cols[c].v;
            }
            return .{ .m = m };
        }

        /// Returns column `c` as a vector.
        pub inline fn col(self: Self, c: usize) Col {
            return .{ .v = self.m[c] };
        }

        /// Returns row `r` as a vector.
        pub inline fn row(self: Self, r: usize) VecType(n, Scalar) {
            var result: [n]Scalar = undefined;
            inline for (0..n) |c| {
                result[c] = self.m[c][r];
            }
            return .{ .v = result };
        }

        /// Component-wise addition.
        pub inline fn add(a: Self, b: Self) Self {
            var result: [n]ColVec = undefined;
            inline for (0..n) |c| {
                result[c] = a.m[c] + b.m[c];
            }
            return .{ .m = result };
        }

        /// Component-wise subtraction.
        pub inline fn sub(a: Self, b: Self) Self {
            var result: [n]ColVec = undefined;
            inline for (0..n) |c| {
                result[c] = a.m[c] - b.m[c];
            }
            return .{ .m = result };
        }

        /// Multiplies every element by scalar `s`.
        pub inline fn scale(a: Self, s: Scalar) Self {
            var result: [n]ColVec = undefined;
            inline for (0..n) |c| {
                result[c] = a.m[c] * @as(ColVec, @splat(s));
            }
            return .{ .m = result };
        }

        /// Component-wise negation.
        pub inline fn neg(a: Self) Self {
            var result: [n]ColVec = undefined;
            inline for (0..n) |c| {
                result[c] = -a.m[c];
            }
            return .{ .m = result };
        }

        /// Returns the matrix product `a * b` (standard linear-algebraic order).
        /// Applies `b`'s transform first, then `a`'s when used with column vectors.
        pub inline fn mul(a: Self, b: Self) Self {
            var result: [n]ColVec = undefined;
            inline for (0..n) |c| {
                var column: ColVec = @splat(0);
                inline for (0..n) |r| {
                    column += a.m[r] * @as(ColVec, @splat(b.m[c][r]));
                }
                result[c] = column;
            }
            return .{ .m = result };
        }

        /// Multiplies the matrix by column vector `v` on the right (`M * v`).
        pub inline fn mulVec(a: Self, v: Col) Col {
            var result: ColVec = @splat(0);
            inline for (0..n) |c| {
                result += a.m[c] * @as(ColVec, @splat(v.v[c]));
            }
            return .{ .v = result };
        }

        /// Returns the transpose (rows become columns).
        pub inline fn transpose(self: Self) Self {
            var result: [n]ColVec = undefined;
            inline for (0..n) |c| {
                inline for (0..n) |r| {
                    result[c][r] = self.m[r][c];
                }
            }
            return .{ .m = result };
        }

        /// For float scalars: returns true if all elements differ by at most `e`.
        /// For integer scalars: exact equality (no `e` parameter).
        pub const eql = if (is_float) struct {
            pub fn f(a: Self, b: Self, e: Scalar) bool {
                const eps: ColVec = @splat(e);
                inline for (0..n) |c| {
                    if (!@reduce(.And, @abs(a.m[c] - b.m[c]) <= eps)) return false;
                }
                return true;
            }
        }.f else struct {
            pub fn f(a: Self, b: Self) bool {
                inline for (0..n) |c| {
                    if (!@reduce(.And, a.m[c] == b.m[c])) return false;
                }
                return true;
            }
        }.f;

        /// Returns the sum of the diagonal elements.
        pub inline fn trace(self: Self) Scalar {
            var sum: Scalar = 0;
            inline for (0..n) |i| {
                sum += self.m[i][i];
            }
            return sum;
        }

        /// Returns the determinant. Available for 2×2, 3×3, and 4×4 matrices.
        pub const det = switch (n) {
            2 => struct {
                fn f(self: Self) Scalar {
                    return self.m[0][0] * self.m[1][1] - self.m[1][0] * self.m[0][1];
                }
            }.f,
            3 => struct {
                fn f(self: Self) Scalar {
                    const c0 = self.m[1][1] * self.m[2][2] - self.m[2][1] * self.m[1][2];
                    const c1 = self.m[2][1] * self.m[0][2] - self.m[0][1] * self.m[2][2];
                    const c2 = self.m[0][1] * self.m[1][2] - self.m[1][1] * self.m[0][2];
                    return self.m[0][0] * c0 + self.m[1][0] * c1 + self.m[2][0] * c2;
                }
            }.f,
            4 => struct {
                fn f(self: Self) Scalar {
                    // 2x2 sub-determinants from column-pairs (0,1) and (2,3)
                    const s0 = self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0];
                    const s1 = self.m[0][0] * self.m[1][2] - self.m[0][2] * self.m[1][0];
                    const s2 = self.m[0][0] * self.m[1][3] - self.m[0][3] * self.m[1][0];
                    const s3 = self.m[0][1] * self.m[1][2] - self.m[0][2] * self.m[1][1];
                    const s4 = self.m[0][1] * self.m[1][3] - self.m[0][3] * self.m[1][1];
                    const s5 = self.m[0][2] * self.m[1][3] - self.m[0][3] * self.m[1][2];

                    const c0 = self.m[2][0] * self.m[3][1] - self.m[2][1] * self.m[3][0];
                    const c1 = self.m[2][0] * self.m[3][2] - self.m[2][2] * self.m[3][0];
                    const c2 = self.m[2][0] * self.m[3][3] - self.m[2][3] * self.m[3][0];
                    const c3 = self.m[2][1] * self.m[3][2] - self.m[2][2] * self.m[3][1];
                    const c4 = self.m[2][1] * self.m[3][3] - self.m[2][3] * self.m[3][1];
                    const c5 = self.m[2][2] * self.m[3][3] - self.m[2][3] * self.m[3][2];

                    return s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0;
                }
            }.f,
            else => @compileError("det() is only defined for 2x2, 3x3, and 4x4 matrices"),
        };

        /// Returns the matrix inverse via the adjugate method.
        /// Assumes that the matrix is non-singular (`det ≠ 0`).
        /// Available for 2×2, 3×3, and 4×4 float matrices.
        pub const inverse = if (!is_float)
            @compileError("inverse() requires a float scalar type")
        else switch (n) {
            2 => struct {
                fn f(self: Self) Self {
                    const inv_det = 1.0 / (self.m[0][0] * self.m[1][1] - self.m[1][0] * self.m[0][1]);
                    return .{ .m = .{
                        .{ self.m[1][1] * inv_det, -self.m[0][1] * inv_det },
                        .{ -self.m[1][0] * inv_det, self.m[0][0] * inv_det },
                    } };
                }
            }.f,
            3 => struct {
                fn f(self: Self) Self {
                    const a = self.m;
                    // Column-0 cofactors (used for determinant)
                    const c00 = a[1][1] * a[2][2] - a[2][1] * a[1][2];
                    const c10 = a[2][0] * a[1][2] - a[1][0] * a[2][2];
                    const c20 = a[1][0] * a[2][1] - a[2][0] * a[1][1];

                    const inv_det = 1.0 / (a[0][0] * c00 + a[0][1] * c10 + a[0][2] * c20);

                    // Adjugate (cofactor transposed): result[col][row] = C(col,row) / det
                    return .{ .m = .{
                        .{
                            c00 * inv_det,
                            (a[2][1] * a[0][2] - a[0][1] * a[2][2]) * inv_det,
                            (a[0][1] * a[1][2] - a[1][1] * a[0][2]) * inv_det,
                        },
                        .{
                            c10 * inv_det,
                            (a[0][0] * a[2][2] - a[2][0] * a[0][2]) * inv_det,
                            (a[1][0] * a[0][2] - a[0][0] * a[1][2]) * inv_det,
                        },
                        .{
                            c20 * inv_det,
                            (a[2][0] * a[0][1] - a[0][0] * a[2][1]) * inv_det,
                            (a[0][0] * a[1][1] - a[1][0] * a[0][1]) * inv_det,
                        },
                    } };
                }
            }.f,
            4 => struct {
                fn f(self: Self) Self {
                    const a = self.m;
                    // 2x2 sub-determinants from column-pairs (0,1) and (2,3)
                    const s0 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
                    const s1 = a[0][0] * a[1][2] - a[0][2] * a[1][0];
                    const s2 = a[0][0] * a[1][3] - a[0][3] * a[1][0];
                    const s3 = a[0][1] * a[1][2] - a[0][2] * a[1][1];
                    const s4 = a[0][1] * a[1][3] - a[0][3] * a[1][1];
                    const s5 = a[0][2] * a[1][3] - a[0][3] * a[1][2];

                    const c0 = a[2][0] * a[3][1] - a[2][1] * a[3][0];
                    const c1 = a[2][0] * a[3][2] - a[2][2] * a[3][0];
                    const c2 = a[2][0] * a[3][3] - a[2][3] * a[3][0];
                    const c3 = a[2][1] * a[3][2] - a[2][2] * a[3][1];
                    const c4 = a[2][1] * a[3][3] - a[2][3] * a[3][1];
                    const c5 = a[2][2] * a[3][3] - a[2][3] * a[3][2];

                    const inv_det = 1.0 / (s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0);

                    return .{ .m = .{
                        .{
                            (a[1][1] * c5 - a[1][2] * c4 + a[1][3] * c3) * inv_det,
                            (-a[0][1] * c5 + a[0][2] * c4 - a[0][3] * c3) * inv_det,
                            (a[3][1] * s5 - a[3][2] * s4 + a[3][3] * s3) * inv_det,
                            (-a[2][1] * s5 + a[2][2] * s4 - a[2][3] * s3) * inv_det,
                        },
                        .{
                            (-a[1][0] * c5 + a[1][2] * c2 - a[1][3] * c1) * inv_det,
                            (a[0][0] * c5 - a[0][2] * c2 + a[0][3] * c1) * inv_det,
                            (-a[3][0] * s5 + a[3][2] * s2 - a[3][3] * s1) * inv_det,
                            (a[2][0] * s5 - a[2][2] * s2 + a[2][3] * s1) * inv_det,
                        },
                        .{
                            (a[1][0] * c4 - a[1][1] * c2 + a[1][3] * c0) * inv_det,
                            (-a[0][0] * c4 + a[0][1] * c2 - a[0][3] * c0) * inv_det,
                            (a[3][0] * s4 - a[3][1] * s2 + a[3][3] * s0) * inv_det,
                            (-a[2][0] * s4 + a[2][1] * s2 - a[2][3] * s0) * inv_det,
                        },
                        .{
                            (-a[1][0] * c3 + a[1][1] * c1 - a[1][2] * c0) * inv_det,
                            (a[0][0] * c3 - a[0][1] * c1 + a[0][2] * c0) * inv_det,
                            (-a[3][0] * s3 + a[3][1] * s1 - a[3][2] * s0) * inv_det,
                            (a[2][0] * s3 - a[2][1] * s1 + a[2][2] * s0) * inv_det,
                        },
                    } };
                }
            }.f,
            else => @compileError("inverse() is only defined for 2x2, 3x3, and 4x4 matrices"),
        };

        /// Counter-clockwise rotation matrix for the given `angle` in radians.
        ///
        /// - 2×2: pure rotation matrix.
        /// - 3×3: 2D rotation embedded in a homogeneous matrix (bottom-right element is 1).
        /// - 4×4: rotation around the given unit `axis` vector (Rodrigues' formula);
        ///   the translation column is left as identity.
        ///
        /// Assumes that `axis` is a unit vector (4×4 variant only).
        /// Only available for float matrices.
        pub const rotation = if (!is_float)
            @compileError("rotation() requires a float scalar type")
        else switch (n) {
            2 => struct {
                fn f(angle: Scalar) Self {
                    const c = @cos(angle);
                    const s = @sin(angle);
                    return .{ .m = .{
                        .{ c, s },
                        .{ -s, c },
                    } };
                }
            }.f,
            3 => struct {
                fn f(angle: Scalar) Self {
                    const c = @cos(angle);
                    const s = @sin(angle);
                    return .{ .m = .{
                        .{ c, s, 0 },
                        .{ -s, c, 0 },
                        .{ 0, 0, 1 },
                    } };
                }
            }.f,
            4 => struct {
                fn f(axis: VecType(3, Scalar), angle: Scalar) Self {
                    const c = @cos(angle);
                    const s = @sin(angle);
                    const t = 1.0 - c;
                    const x = axis.v[0];
                    const y = axis.v[1];
                    const z = axis.v[2];
                    return .{ .m = .{
                        .{ t * x * x + c, t * x * y + s * z, t * x * z - s * y, 0 },
                        .{ t * x * y - s * z, t * y * y + c, t * y * z + s * x, 0 },
                        .{ t * x * z + s * y, t * y * z - s * x, t * z * z + c, 0 },
                        .{ 0, 0, 0, 1 },
                    } };
                }
            }.f,
            else => @compileError("rotation() is only defined for 2x2, 3x3, and 4x4 matrices"),
        };

        /// Translation matrix embedding a displacement vector.
        ///
        /// - 3×3: 2D homogeneous translation; takes a `Vec2`.
        /// - 4×4: 3D homogeneous translation; takes a `Vec3`.
        ///
        /// Directions (w = 0) are unaffected; points (w = 1) are displaced.
        /// Only available for float matrices.
        pub const translation = if (!is_float)
            @compileError("translation() requires a float scalar type")
        else switch (n) {
            3 => struct {
                fn f(v: VecType(2, Scalar)) Self {
                    return .{ .m = .{
                        .{ 1, 0, 0 },
                        .{ 0, 1, 0 },
                        .{ v.v[0], v.v[1], 1 },
                    } };
                }
            }.f,
            4 => struct {
                fn f(v: VecType(3, Scalar)) Self {
                    return .{ .m = .{
                        .{ 1, 0, 0, 0 },
                        .{ 0, 1, 0, 0 },
                        .{ 0, 0, 1, 0 },
                        .{ v.v[0], v.v[1], v.v[2], 1 },
                    } };
                }
            }.f,
            else => @compileError("translation() is only defined for 3x3 and 4x4 matrices"),
        };

        /// Diagonal scaling matrix.
        ///
        /// - 2×2: pure 2D scale; takes a `Vec2`.
        /// - 3×3: 2D scale embedded in a homogeneous matrix; takes a `Vec2`.
        /// - 4×4: 3D scale embedded in a homogeneous matrix; takes a `Vec3`.
        ///
        /// Only available for float matrices.
        pub const scaling = if (!is_float)
            @compileError("scaling() requires a float scalar type")
        else switch (n) {
            2 => struct {
                fn f(v: VecType(2, Scalar)) Self {
                    return .{ .m = .{
                        .{ v.v[0], 0 },
                        .{ 0, v.v[1] },
                    } };
                }
            }.f,
            3 => struct {
                fn f(v: VecType(2, Scalar)) Self {
                    return .{ .m = .{
                        .{ v.v[0], 0, 0 },
                        .{ 0, v.v[1], 0 },
                        .{ 0, 0, 1 },
                    } };
                }
            }.f,
            4 => struct {
                fn f(v: VecType(3, Scalar)) Self {
                    return .{ .m = .{
                        .{ v.v[0], 0, 0, 0 },
                        .{ 0, v.v[1], 0, 0 },
                        .{ 0, 0, v.v[2], 0 },
                        .{ 0, 0, 0, 1 },
                    } };
                }
            }.f,
            else => @compileError("scaling() is only defined for 2x2, 3x3, and 4x4 matrices"),
        };

        /// Constructs a right-handed view matrix.
        ///
        /// In the resulting view space the camera looks along -Z, `up` maps to +Y,
        /// and `eye` maps to the origin.
        ///
        /// Returns `null` if `eye == target` or if `up` is parallel to the forward
        /// direction (degenerate configuration). The caller must handle this case.
        /// Only available for 4×4 float matrices.
        pub const lookAt = if (n == 4 and is_float) struct {
            fn f(eye: VecType(3, Scalar), target: VecType(3, Scalar), up: VecType(3, Scalar)) ?Self {
                const cam_fwd_unnorm = target.sub(eye);
                const fwd_len_sq = cam_fwd_unnorm.dot(cam_fwd_unnorm);
                if (fwd_len_sq < math.floatEps(Scalar)) return null; // eye == target
                const cam_fwd = cam_fwd_unnorm.scale(1.0 / @sqrt(fwd_len_sq));

                const cam_right_unnorm = cam_fwd.cross(up);
                const right_len_sq = cam_right_unnorm.dot(cam_right_unnorm);
                if (right_len_sq < math.floatEps(Scalar)) return null; // up parallel to forward
                const cam_right = cam_right_unnorm.scale(1.0 / @sqrt(right_len_sq));

                const cam_up = cam_right.cross(cam_fwd);

                // View matrix = transpose(rotation) * translate(-eye)
                // Combining into one matrix (column-major):
                return .{ .m = .{
                    .{ cam_right.v[0], cam_up.v[0], -cam_fwd.v[0], 0 },
                    .{ cam_right.v[1], cam_up.v[1], -cam_fwd.v[1], 0 },
                    .{ cam_right.v[2], cam_up.v[2], -cam_fwd.v[2], 0 },
                    .{
                        -cam_right.dot(eye),
                        -cam_up.dot(eye),
                        cam_fwd.dot(eye),
                        1,
                    },
                } };
            }
        }.f else @compileError("lookAt() is only defined for 4x4 float matrices");

        /// Perspective projection mapping depth to [0, 1] (Vulkan / Metal / SDL3 GPU convention).
        ///
        /// `fov_y` is the full vertical field-of-view in radians. `aspect` is
        /// width / height. `near` and `far` are positive distances along the
        /// camera -Z axis; the near plane maps to depth 0, the far plane to 1.
        /// Only available for 4×4 float matrices.
        pub const perspective = if (n == 4 and is_float) struct {
            fn f(fov_y: Scalar, aspect: Scalar, near: Scalar, far: Scalar) Self {
                const half_tan = @tan(fov_y * @as(Scalar, 0.5));
                const inv_range = 1.0 / (near - far);
                return .{ .m = .{
                    .{ 1.0 / (aspect * half_tan), 0, 0, 0 },
                    .{ 0, 1.0 / half_tan, 0, 0 },
                    .{ 0, 0, far * inv_range, -1 },
                    .{ 0, 0, near * far * inv_range, 0 },
                } };
            }
        }.f else @compileError("perspective() is only defined for 4x4 float matrices");

        /// Orthographic projection mapping depth to [0, 1] (Vulkan / Metal / SDL3 GPU convention).
        ///
        /// `left`, `right_`, `bottom`, `top` define the view volume extents in
        /// view space. `near` and `far` are positive distances along the camera
        /// -Z axis; the near plane maps to depth 0, the far plane to 1.
        /// Only available for 4×4 float matrices.
        pub const ortho = if (n == 4 and is_float) struct {
            fn f(left: Scalar, right_: Scalar, bottom: Scalar, top: Scalar, near: Scalar, far: Scalar) Self {
                const rl = right_ - left;
                const tb = top - bottom;
                const fn_ = far - near;
                return .{ .m = .{
                    .{ 2.0 / rl, 0, 0, 0 },
                    .{ 0, 2.0 / tb, 0, 0 },
                    .{ 0, 0, -1.0 / fn_, 0 },
                    .{ -(right_ + left) / rl, -(top + bottom) / tb, -near / fn_, 1 },
                } };
            }
        }.f else @compileError("ortho() is only defined for 4x4 float matrices");

        /// Extracts the upper-left 3×3 submatrix from a 4×4 matrix.
        ///
        /// Useful for stripping the translation column from a model/view matrix
        /// when transforming direction vectors or normals.
        /// Only available for 3×3 float matrices.
        pub const fromMat4 = if (n == 3 and is_float) struct {
            fn f(m4: MatType(4, Scalar)) Self {
                return .{ .m = .{
                    .{ m4.m[0][0], m4.m[0][1], m4.m[0][2] },
                    .{ m4.m[1][0], m4.m[1][1], m4.m[1][2] },
                    .{ m4.m[2][0], m4.m[2][1], m4.m[2][2] },
                } };
            }
        }.f else @compileError("fromMat4() is only defined for 3x3 float matrices");
    };
}

// TODO: Remove file wide declarations. Use non pub declarations within a struct/namespace
const Vec2 = VecType(2, f32);
const Vec3 = VecType(3, f32);
const Vec4 = VecType(4, f32);

const Vec2d = VecType(2, f64);
const Vec3d = VecType(3, f64);
const Vec4d = VecType(4, f64);

const Vec2i = VecType(2, i32);
const Vec3i = VecType(3, i32);
const Vec4i = VecType(4, i32);

const Vec2u = VecType(2, u32);
const Vec3u = VecType(3, u32);
const Vec4u = VecType(4, u32);

const Mat2 = MatType(2, f32);
const Mat3 = MatType(3, f32);
const Mat4 = MatType(4, f32);

const Mat2d = MatType(2, f64);
const Mat3d = MatType(3, f64);
const Mat4d = MatType(4, f64);

/// 2D rotor (geometric algebra equivalent of a complex unit phasor).
/// Represents a rotation as `cos(angle/2) + sin(angle/2) * e12`.
/// Positive angle = counter-clockwise.
pub fn Rotor2Type(comptime Scalar: type) type {
    const Vec2S = VecType(2, Scalar);
    const Mat2S = MatType(2, Scalar);
    const Mat3S = MatType(3, Scalar);

    return struct {
        const Self = @This();

        /// Scalar part (cos(angle/2))
        a: Scalar,
        /// Bivector e12 coefficient (sin(angle/2))
        b: Scalar,

        /// The identity rotor (no rotation).
        pub const identity = Self{ .a = 1, .b = 0 };

        /// Create a rotor from an angle (radians). Positive = counter-clockwise.
        pub inline fn fromAngle(theta: Scalar) Self {
            const half = theta * @as(Scalar, 0.5);
            return .{ .a = @cos(half), .b = @sin(half) };
        }

        /// Create a rotor that rotates unit vector `from` to unit vector `to`.
        pub inline fn fromVecs(from: Vec2S, to: Vec2S) Self {
            // R = normalize(1 + from·to + from∧to)
            const d = from.dot(to);
            const w = from.v[0] * to.v[1] - from.v[1] * to.v[0]; // 2D wedge product
            const r = Self{ .a = 1 + d, .b = w };
            const len_sq = r.a * r.a + r.b * r.b;
            if (len_sq < 1e-12) {
                // from ≈ -to: 180° rotation
                return .{ .a = 0, .b = 1 };
            }
            const inv_len = 1.0 / @sqrt(len_sq);
            return .{ .a = r.a * inv_len, .b = r.b * inv_len };
        }

        /// Compose two rotations. `a.mul(b)` applies `b` first, then `a`.
        pub inline fn mul(p: Self, q: Self) Self {
            return .{
                .a = p.a * q.a - p.b * q.b,
                .b = p.a * q.b + p.b * q.a,
            };
        }

        /// Rotate a 2D vector by this rotor.
        pub inline fn rotate(self: Self, v: Vec2S) Vec2S {
            const a2 = self.a * self.a;
            const b2 = self.b * self.b;
            const ab2 = 2 * self.a * self.b;
            return .{ .v = .{
                (a2 - b2) * v.v[0] - ab2 * v.v[1],
                ab2 * v.v[0] + (a2 - b2) * v.v[1],
            } };
        }

        /// Reverse (conjugate) — the inverse rotation for a unit rotor.
        pub inline fn reverse(self: Self) Self {
            return .{ .a = self.a, .b = -self.b };
        }

        /// Normalize to unit length.
        pub inline fn normalize(self: Self) Self {
            const inv_len = 1.0 / @sqrt(self.a * self.a + self.b * self.b);
            return .{ .a = self.a * inv_len, .b = self.b * inv_len };
        }

        /// Convert to a 2×2 rotation matrix.
        pub inline fn toMat2(self: Self) Mat2S {
            const c = self.a * self.a - self.b * self.b;
            const s = 2 * self.a * self.b;
            return .{ .m = .{
                .{ c, s },
                .{ -s, c },
            } };
        }

        /// Convert to a 3×3 homogeneous rotation matrix.
        pub inline fn toMat3(self: Self) Mat3S {
            const c = self.a * self.a - self.b * self.b;
            const s = 2 * self.a * self.b;
            return .{ .m = .{
                .{ c, s, 0 },
                .{ -s, c, 0 },
                .{ 0, 0, 1 },
            } };
        }

        /// Extract the rotation angle in radians.
        pub inline fn angle(self: Self) Scalar {
            return 2 * math.atan2(self.b, self.a);
        }

        /// Normalized linear interpolation.
        pub inline fn nlerp(from: Self, to: Self, t: Scalar) Self {
            // Ensure shortest path
            var to_adj = to;
            if (from.a * to.a + from.b * to.b < 0) {
                to_adj = .{ .a = -to.a, .b = -to.b };
            }
            return (Self{
                .a = from.a + (to_adj.a - from.a) * t,
                .b = from.b + (to_adj.b - from.b) * t,
            }).normalize();
        }

        /// Spherical linear interpolation.
        pub inline fn slerp(from: Self, to: Self, t: Scalar) Self {
            var d = from.a * to.a + from.b * to.b;
            var to_adj = to;
            if (d < 0) {
                d = -d;
                to_adj = .{ .a = -to.a, .b = -to.b };
            }
            // TODO: Use a more exact value here
            if (d > 0.9995) {
                return nlerp(from, to_adj, t);
            }
            const theta = math.acos(d);
            const sin_theta = @sin(theta);
            const s0 = @sin((1 - t) * theta) / sin_theta;
            const s1 = @sin(t * theta) / sin_theta;
            return (Self{
                .a = from.a * s0 + to_adj.a * s1,
                .b = from.b * s0 + to_adj.b * s1,
            });
        }
    };
}

/// 3D rotor (geometric algebra equivalent of a unit quaternion).
/// Represents a rotation as `cos(angle/2) + sin(angle/2) * B` where B is a unit bivector.
/// Components map to quaternion as: `w=a, i=b12, j=b02, k=b01`.
pub fn Rotor3Type(comptime Scalar: type) type {
    const Vec3S = VecType(3, Scalar);
    const Mat3S = MatType(3, Scalar);
    const Mat4S = MatType(4, Scalar);

    return struct {
        const Self = @This();

        /// Scalar part (cos(angle/2))
        a: Scalar,
        /// Bivector e12 (xy-plane) coefficient
        b01: Scalar,
        /// Bivector e13 (xz-plane) coefficient
        b02: Scalar,
        /// Bivector e23 (yz-plane) coefficient
        b12: Scalar,

        /// The identity rotor (no rotation).
        pub const identity = Self{ .a = 1, .b01 = 0, .b02 = 0, .b12 = 0 };

        /// Create a rotor from an angle (radians) and a unit axis vector.
        /// The axis is the Hodge dual of the rotation plane:
        ///   x-axis → yz-plane, y-axis → xz-plane, z-axis → xy-plane.
        pub inline fn fromAxisAngle(angle: Scalar, axis: Vec3S) Self {
            const half = angle * @as(Scalar, 0.5);
            const s = @sin(half);
            return .{
                .a = @cos(half),
                .b01 = s * axis.v[2], // z → xy-plane
                .b02 = s * axis.v[1], // y → xz-plane
                .b12 = s * axis.v[0], // x → yz-plane
            };
        }

        /// Create a rotor that rotates unit vector `from` to unit vector `to`.
        pub inline fn fromVecs(from: Vec3S, to: Vec3S) Self {
            // R = normalize(1 + dot(from,to), cross(from,to) mapped to bivector)
            const d = from.dot(to);
            // cross(from, to) components → bivector via quaternion mapping
            const cx = from.v[1] * to.v[2] - from.v[2] * to.v[1];
            const cy = from.v[2] * to.v[0] - from.v[0] * to.v[2];
            const cz = from.v[0] * to.v[1] - from.v[1] * to.v[0];

            var r = Self{ .a = 1 + d, .b01 = cz, .b02 = cy, .b12 = cx };
            const len_sq = r.a * r.a + r.b01 * r.b01 + r.b02 * r.b02 + r.b12 * r.b12;

            if (len_sq < 1e-12) {
                // from ≈ -to: pick any perpendicular axis for 180° rotation
                const ax = @abs(from.v[0]);
                const ay = @abs(from.v[1]);
                const az = @abs(from.v[2]);
                const perp = if (ax < ay and ax < az)
                    Vec3S.init(.{ 1, 0, 0 })
                else if (ay < az)
                    Vec3S.init(.{ 0, 1, 0 })
                else
                    Vec3S.init(.{ 0, 0, 1 });
                // cross(from, perp) gives a vector perpendicular to from
                const px = from.v[1] * perp.v[2] - from.v[2] * perp.v[1];
                const py = from.v[2] * perp.v[0] - from.v[0] * perp.v[2];
                const pz = from.v[0] * perp.v[1] - from.v[1] * perp.v[0];
                const plen = @sqrt(px * px + py * py + pz * pz);
                const inv = 1.0 / plen;
                // 180° rotation: a=0, bivector = unit perpendicular axis (mapped)
                return .{ .a = 0, .b01 = pz * inv, .b02 = py * inv, .b12 = px * inv };
            }

            const inv_len = 1.0 / @sqrt(len_sq);
            r.a *= inv_len;
            r.b01 *= inv_len;
            r.b02 *= inv_len;
            r.b12 *= inv_len;
            return r;
        }

        /// Create a rotor from Euler angles (radians). Convention: YXZ (yaw-pitch-roll).
        /// Applied as: roll (Z) first, then pitch (X), then yaw (Y).
        pub inline fn fromEuler(yaw: Scalar, pitch: Scalar, roll: Scalar) Self {
            const ry = Self.fromAxisAngle(yaw, Vec3S.init(.{ 0, 1, 0 }));
            const rx = Self.fromAxisAngle(pitch, Vec3S.init(.{ 1, 0, 0 }));
            const rz = Self.fromAxisAngle(roll, Vec3S.init(.{ 0, 0, 1 }));
            return ry.mul(rx.mul(rz));
        }

        /// Compose two rotations. `a.mul(b)` applies `b` first, then `a`.
        /// Equivalent to quaternion multiplication with mapping w=a, i=b12, j=b02, k=b01.
        pub inline fn mul(p: Self, q: Self) Self {
            return .{
                .a = p.a * q.a - p.b12 * q.b12 - p.b02 * q.b02 - p.b01 * q.b01,
                .b01 = p.a * q.b01 + p.b12 * q.b02 - p.b02 * q.b12 + p.b01 * q.a,
                .b02 = p.a * q.b02 - p.b12 * q.b01 + p.b02 * q.a + p.b01 * q.b12,
                .b12 = p.a * q.b12 + p.b12 * q.a + p.b02 * q.b01 - p.b01 * q.b02,
            };
        }

        /// Rotate a 3D vector by this rotor (sandwich product).
        pub inline fn rotate(self: Self, v: Vec3S) Vec3S {
            // Equivalent to: t = 2 * cross(bv, v); result = v + a*t + cross(bv, t)
            // where bv = (b12, b02, b01) is the axis-vector dual of the bivector.
            const bx = self.b12;
            const by = self.b02;
            const bz = self.b01;

            // t = 2 * cross(bv, v)
            const tx = 2 * (by * v.v[2] - bz * v.v[1]);
            const ty = 2 * (bz * v.v[0] - bx * v.v[2]);
            const tz = 2 * (bx * v.v[1] - by * v.v[0]);

            return .{ .v = .{
                v.v[0] + self.a * tx + (by * tz - bz * ty),
                v.v[1] + self.a * ty + (bz * tx - bx * tz),
                v.v[2] + self.a * tz + (bx * ty - by * tx),
            } };
        }

        /// Reverse (conjugate) — the inverse rotation for a unit rotor.
        pub inline fn reverse(self: Self) Self {
            return .{ .a = self.a, .b01 = -self.b01, .b02 = -self.b02, .b12 = -self.b12 };
        }

        /// Normalize to unit length.
        pub inline fn normalize(self: Self) Self {
            const inv_len = 1.0 / @sqrt(self.a * self.a + self.b01 * self.b01 + self.b02 * self.b02 + self.b12 * self.b12);
            return .{
                .a = self.a * inv_len,
                .b01 = self.b01 * inv_len,
                .b02 = self.b02 * inv_len,
                .b12 = self.b12 * inv_len,
            };
        }

        /// Convert to a 3×3 rotation matrix.
        pub inline fn toMat3(self: Self) Mat3S {
            const w = self.a;
            const x = self.b12;
            const y = self.b02;
            const z = self.b01;

            const x2 = x + x;
            const y2 = y + y;
            const z2 = z + z;
            const xx = x * x2;
            const xy = x * y2;
            const xz = x * z2;
            const yy = y * y2;
            const yz = y * z2;
            const zz = z * z2;
            const wx = w * x2;
            const wy = w * y2;
            const wz = w * z2;

            // Column-major: m[col][row]
            return .{ .m = .{
                .{ 1 - (yy + zz), xy + wz, xz - wy },
                .{ xy - wz, 1 - (xx + zz), yz + wx },
                .{ xz + wy, yz - wx, 1 - (xx + yy) },
            } };
        }

        /// Convert to a 4×4 rotation matrix (upper-left 3×3 rotation, rest is identity).
        pub inline fn toMat4(self: Self) Mat4S {
            const w = self.a;
            const x = self.b12;
            const y = self.b02;
            const z = self.b01;

            const x2 = x + x;
            const y2 = y + y;
            const z2 = z + z;
            const xx = x * x2;
            const xy = x * y2;
            const xz = x * z2;
            const yy = y * y2;
            const yz = y * z2;
            const zz = z * z2;
            const wx = w * x2;
            const wy = w * y2;
            const wz = w * z2;

            return .{ .m = .{
                .{ 1 - (yy + zz), xy + wz, xz - wy, 0 },
                .{ xy - wz, 1 - (xx + zz), yz + wx, 0 },
                .{ xz + wy, yz - wx, 1 - (xx + yy), 0 },
                .{ 0, 0, 0, 1 },
            } };
        }

        /// Normalized linear interpolation.
        pub inline fn nlerp(from: Self, to: Self, t: Scalar) Self {
            var to_adj = to;
            if (from.a * to.a + from.b01 * to.b01 + from.b02 * to.b02 + from.b12 * to.b12 < 0) {
                to_adj = .{ .a = -to.a, .b01 = -to.b01, .b02 = -to.b02, .b12 = -to.b12 };
            }
            return (Self{
                .a = from.a + (to_adj.a - from.a) * t,
                .b01 = from.b01 + (to_adj.b01 - from.b01) * t,
                .b02 = from.b02 + (to_adj.b02 - from.b02) * t,
                .b12 = from.b12 + (to_adj.b12 - from.b12) * t,
            }).normalize();
        }

        /// Spherical linear interpolation.
        pub inline fn slerp(from: Self, to: Self, t: Scalar) Self {
            var d = from.a * to.a + from.b01 * to.b01 + from.b02 * to.b02 + from.b12 * to.b12;
            var to_adj = to;
            if (d < 0) {
                d = -d;
                to_adj = .{ .a = -to.a, .b01 = -to.b01, .b02 = -to.b02, .b12 = -to.b12 };
            }
            if (d > 0.9995) {
                return nlerp(from, to_adj, t);
            }
            const theta = math.acos(d);
            const sin_theta = @sin(theta);
            const s0 = @sin((1 - t) * theta) / sin_theta;
            const s1 = @sin(t * theta) / sin_theta;
            return Self{
                .a = from.a * s0 + to_adj.a * s1,
                .b01 = from.b01 * s0 + to_adj.b01 * s1,
                .b02 = from.b02 * s0 + to_adj.b02 * s1,
                .b12 = from.b12 * s0 + to_adj.b12 * s1,
            };
        }
    };
}

/// 2D versor (Pin(2) group element).
///
/// Represents either a rotation (even parity) or a reflection (odd parity)
/// in the Cl(2,0) Clifford algebra. A single type that unifies both proper
/// and improper isometries of the plane.
///
/// The 2 components are reinterpreted based on parity:
///   - Even: c[0] = scalar,  c[1] = e₁₂ bivector  (same layout as Rot2)
///   - Odd:  c[0] = e₁,      c[1] = e₂             (reflection normal)
///
/// Convention: the sandwich product is `α(Ṽ) · v · V` where α is the grade
/// automorphism and Ṽ is the reverse. `mul(a, b)` applies b first, then a,
/// matching `Rotor2Type.mul`. Internally this computes the Clifford product
/// with arguments swapped (`b · a`) because the vector sandwich has the
/// opposite composition order from the bivector sandwich used by Rot2.
///
/// See `.pi/journals/2026-03-03-versor-clifford-algebra-verification.md`
/// for the full derivation and computational verification of all formulas.
/// TODO: Extract contents of `.pi/journals/2026-03-03-versor-clifford-algebra-verification.md` to local reference
pub fn Versor2Type(comptime Scalar: type) type {
    const Vec2S = VecType(2, Scalar);
    const Mat2S = MatType(2, Scalar);
    const Mat3S = MatType(3, Scalar);
    const Rot2S = Rotor2Type(Scalar);

    return struct {
        const Self = @This();

        /// The 2 components of the active subspace.
        /// Even: c[0] = scalar (cos θ/2), c[1] = e₁₂ (sin θ/2).
        /// Odd:  c[0] = e₁, c[1] = e₂  (unit reflection normal).
        c: [2]Scalar,
        /// Whether this versor is even (rotation) or odd (reflection).
        parity: Parity,

        pub const Parity = enum(u1) { even, odd };

        /// The identity versor (no transformation).
        pub const identity = Self{ .c = .{ 1, 0 }, .parity = .even };

        // -----------------------------------------------------------------
        // Construction
        // -----------------------------------------------------------------

        /// Create an even versor from a Rot2 rotor. No sign change needed in 2D.
        pub inline fn fromRotor(rot: Rot2S) Self {
            return .{ .c = .{ rot.a, rot.b }, .parity = .even };
        }

        /// Create an even versor from a rotation angle (radians, CCW positive).
        pub inline fn fromAngle(theta: Scalar) Self {
            return fromRotor(Rot2S.fromAngle(theta));
        }

        /// Create an odd versor (reflection) from a unit normal vector.
        /// The reflection is across the line perpendicular to `normal`.
        pub inline fn fromReflection(normal: Vec2S) Self {
            return .{ .c = .{ normal.v[0], normal.v[1] }, .parity = .odd };
        }

        // -----------------------------------------------------------------
        // Composition — mul(a, b) applies b first, then a
        // Internally computes Clifford product b · a (swapped)
        // -----------------------------------------------------------------

        /// Compose two versors. `a.mul(b)` applies `b` first, then `a`.
        pub inline fn mul(self: Self, other: Self) Self {
            // Clifford product: left = other, right = self
            return switch (self.parity) {
                .even => switch (other.parity) {
                    // Even × Even → Even (complex number multiplication)
                    .even => .{
                        .c = .{
                            other.c[0] * self.c[0] - other.c[1] * self.c[1],
                            other.c[0] * self.c[1] + other.c[1] * self.c[0],
                        },
                        .parity = .even,
                    },
                    // Odd × Even → Odd
                    .odd => .{
                        .c = .{
                            self.c[0] * other.c[0] - self.c[1] * other.c[1],
                            self.c[1] * other.c[0] + self.c[0] * other.c[1],
                        },
                        .parity = .odd,
                    },
                },
                .odd => switch (other.parity) {
                    // Even × Odd → Odd
                    .even => .{
                        .c = .{
                            other.c[0] * self.c[0] + other.c[1] * self.c[1],
                            other.c[0] * self.c[1] - other.c[1] * self.c[0],
                        },
                        .parity = .odd,
                    },
                    // Odd × Odd → Even
                    .odd => .{
                        .c = .{
                            other.c[0] * self.c[0] + other.c[1] * self.c[1],
                            other.c[0] * self.c[1] - other.c[1] * self.c[0],
                        },
                        .parity = .even,
                    },
                },
            };
        }

        // -----------------------------------------------------------------
        // Application — v → α(Ṽ) · v · V
        // -----------------------------------------------------------------

        /// Transform a 2D vector by this versor (sandwich product).
        pub inline fn apply(self: Self, v: Vec2S) Vec2S {
            return switch (self.parity) {
                .even => {
                    // α(Ṽ) = Ṽ for even. Reduces to Ṽ·v·V = standard rotor rotation.
                    const a = self.c[0];
                    const b = self.c[1];
                    const a2 = a * a;
                    const b2 = b * b;
                    const ab2: Scalar = 2 * a * b;
                    return .{ .v = .{
                        (a2 - b2) * v.v[0] - ab2 * v.v[1],
                        ab2 * v.v[0] + (a2 - b2) * v.v[1],
                    } };
                },
                .odd => {
                    // α(Ṽ) = −Ṽ for odd (grade automorphism negates vector).
                    // Reduces to reflection: v − 2(v·n)n
                    const nx = self.c[0];
                    const ny = self.c[1];
                    const d: Scalar = 2 * (v.v[0] * nx + v.v[1] * ny);
                    return .{ .v = .{
                        v.v[0] - d * nx,
                        v.v[1] - d * ny,
                    } };
                },
            };
        }

        // -----------------------------------------------------------------
        // Reverse / inverse
        // -----------------------------------------------------------------

        /// Reverse (conjugate). For unit versors, this is the inverse transformation.
        /// Even: negate e₁₂ (grade 2). Odd: no change (grade 1 is self-reverse).
        pub inline fn reverse(self: Self) Self {
            return switch (self.parity) {
                .even => .{ .c = .{ self.c[0], -self.c[1] }, .parity = .even },
                .odd => self, // grade-1 only, no grade-3 in 2D
            };
        }

        // -----------------------------------------------------------------
        // Normalization
        // -----------------------------------------------------------------

        /// Squared norm of the versor.
        pub inline fn lenSqr(self: Self) Scalar {
            return self.c[0] * self.c[0] + self.c[1] * self.c[1];
        }

        /// Normalize to unit length.
        pub inline fn normalize(self: Self) Self {
            const inv_len = 1.0 / @sqrt(self.lenSqr());
            return .{ .c = .{ self.c[0] * inv_len, self.c[1] * inv_len }, .parity = self.parity };
        }

        // -----------------------------------------------------------------
        // Conversion
        // -----------------------------------------------------------------

        /// Convert to Rot2 if even. Returns null if odd (reflection).
        pub inline fn toRotor(self: Self) ?Rot2S {
            return switch (self.parity) {
                .even => .{ .a = self.c[0], .b = self.c[1] },
                .odd => null,
            };
        }

        /// Convert to a 2×2 matrix. Even → rotation matrix (det +1), odd → reflection (det −1).
        pub inline fn toMat2(self: Self) Mat2S {
            return switch (self.parity) {
                .even => {
                    const c = self.c[0] * self.c[0] - self.c[1] * self.c[1];
                    const s: Scalar = 2 * self.c[0] * self.c[1];
                    return .{ .m = .{
                        .{ c, s },
                        .{ -s, c },
                    } };
                },
                .odd => {
                    // Reflection matrix: I − 2nnᵀ
                    const nx = self.c[0];
                    const ny = self.c[1];
                    return .{ .m = .{
                        .{ 1 - 2 * nx * nx, -2 * nx * ny },
                        .{ -2 * nx * ny, 1 - 2 * ny * ny },
                    } };
                },
            };
        }

        /// Convert to a 3×3 homogeneous matrix.
        pub inline fn toMat3(self: Self) Mat3S {
            const m2 = self.toMat2();
            return .{ .m = .{
                .{ m2.m[0][0], m2.m[0][1], 0 },
                .{ m2.m[1][0], m2.m[1][1], 0 },
                .{ 0, 0, 1 },
            } };
        }

        /// Determinant of the transformation: +1 for even (rotation), −1 for odd (reflection).
        pub inline fn det(self: Self) Scalar {
            return switch (self.parity) {
                .even => 1,
                .odd => -1,
            };
        }

        // -----------------------------------------------------------------
        // Interpolation (within same parity)
        // -----------------------------------------------------------------

        /// Normalized linear interpolation. Both versors must have the same parity.
        pub inline fn nlerp(from: Self, to: Self, t: Scalar) Self {
            std.debug.assert(from.parity == to.parity);
            var to_adj = to;
            if (from.c[0] * to.c[0] + from.c[1] * to.c[1] < 0) {
                to_adj = .{ .c = .{ -to.c[0], -to.c[1] }, .parity = to.parity };
            }
            return (Self{
                .c = .{
                    from.c[0] + (to_adj.c[0] - from.c[0]) * t,
                    from.c[1] + (to_adj.c[1] - from.c[1]) * t,
                },
                .parity = from.parity,
            }).normalize();
        }

        /// Spherical linear interpolation. Both versors must have the same parity.
        pub inline fn slerp(from: Self, to: Self, t: Scalar) Self {
            std.debug.assert(from.parity == to.parity);
            var d = from.c[0] * to.c[0] + from.c[1] * to.c[1];
            var to_adj = to;
            if (d < 0) {
                d = -d;
                to_adj = .{ .c = .{ -to.c[0], -to.c[1] }, .parity = to.parity };
            }
            if (d > 0.9995) {
                return nlerp(from, to_adj, t);
            }
            const theta = math.acos(d);
            const sin_theta = @sin(theta);
            const s0 = @sin((1 - t) * theta) / sin_theta;
            const s1 = @sin(t * theta) / sin_theta;
            return Self{
                .c = .{
                    from.c[0] * s0 + to_adj.c[0] * s1,
                    from.c[1] * s0 + to_adj.c[1] * s1,
                },
                .parity = from.parity,
            };
        }
    };
}

/// 3D versor (Pin(3) group element).
///
/// Represents either a rotation (even parity) or a reflection/rotoreflection
/// (odd parity) in the Cl(3,0) Clifford algebra. Unifies all orientation-
/// preserving and orientation-reversing isometries of 3-space.
///
/// The 4 components are reinterpreted based on parity:
///   - Even: c[0] = scalar,  c[1] = e₁₂,  c[2] = e₁₃,  c[3] = e₂₃  (rotor)
///   - Odd:  c[0] = e₁,      c[1] = e₂,    c[2] = e₃,    c[3] = e₁₂₃ (reflection)
///
/// Convention: the sandwich product is `α(Ṽ) · v · V` where α is the grade
/// automorphism and Ṽ is the reverse. `mul(a, b)` applies b first, then a,
/// matching `Rotor3Type.mul`. Internally this computes the Clifford product
/// with arguments swapped (`b · a`) because the vector sandwich has the
/// opposite composition order from the bivector sandwich used by Rot3.
///
/// The e₁₃ component is NEGATED relative to the Rot3 convention due to the
/// Hodge dual: ⋆e₂ = −e₁₃. Conversions via `fromRotor`/`toRotor` handle this.
///
/// See `.pi/journals/2026-03-03-versor-clifford-algebra-verification.md`
/// for the full derivation and computational verification of all formulas.
pub fn Versor3Type(comptime Scalar: type) type {
    const Vec3S = VecType(3, Scalar);
    const Mat3S = MatType(3, Scalar);
    const Mat4S = MatType(4, Scalar);
    const Rot3S = Rotor3Type(Scalar);

    return struct {
        const Self = @This();

        /// The 4 components of the active subspace.
        /// Even: (scalar, e₁₂, e₁₃, e₂₃) — rotation.
        /// Odd:  (e₁, e₂, e₃, e₁₂₃)     — reflection/rotoreflection.
        c: [4]Scalar,
        /// Whether this versor is even (rotation) or odd (reflection/rotoreflection).
        parity: Parity,

        pub const Parity = enum(u1) { even, odd };

        /// The identity versor (no transformation).
        pub const identity = Self{ .c = .{ 1, 0, 0, 0 }, .parity = .even };

        // -----------------------------------------------------------------
        // Construction
        // -----------------------------------------------------------------

        /// Create an even versor from a Rot3 rotor.
        /// Negates the e₁₃ component to convert from quaternion to GA convention.
        pub inline fn fromRotor(rot: Rot3S) Self {
            return .{ .c = .{ rot.a, rot.b01, -rot.b02, rot.b12 }, .parity = .even };
        }

        /// Create an even versor from an axis-angle rotation (GA convention).
        /// The bivector is constructed via the Hodge dual of the axis:
        /// `B = az·e₁₂ − ay·e₁₃ + ax·e₂₃`
        pub inline fn fromAxisAngle(angle: Scalar, axis: Vec3S) Self {
            const half = angle * @as(Scalar, 0.5);
            const s = @sin(half);
            return .{
                .c = .{
                    @cos(half),
                    s * axis.v[2], // e₁₂ ← az
                    -s * axis.v[1], // e₁₃ ← −ay (Hodge dual sign)
                    s * axis.v[0], // e₂₃ ← ax
                },
                .parity = .even,
            };
        }

        /// Create an even versor that rotates unit vector `from` to unit vector `to`.
        pub inline fn fromVecs(from: Vec3S, to: Vec3S) Self {
            // Delegate to Rot3 and convert.
            return fromRotor(Rot3S.fromVecs(from, to));
        }

        /// Create an odd versor (reflection) from a unit normal vector.
        /// The reflection is across the plane perpendicular to `normal`.
        pub inline fn fromReflection(normal: Vec3S) Self {
            return .{ .c = .{ normal.v[0], normal.v[1], normal.v[2], 0 }, .parity = .odd };
        }

        // -----------------------------------------------------------------
        // Composition — mul(a, b) applies b first, then a
        // Internally computes Clifford product b · a (swapped)
        //
        // The four Clifford product cases are:
        //   Left × Right → Result
        // where Left = other, Right = self (because we compute other · self).
        // -----------------------------------------------------------------

        /// Compose two versors. `a.mul(b)` applies `b` first, then `a`.
        pub inline fn mul(self: Self, other: Self) Self {
            // Clifford product: left = other, right = self
            // Dispatch on (left_parity, right_parity) = (other.parity, self.parity)
            const L = other.c;
            const R = self.c;
            return switch (other.parity) {
                .even => switch (self.parity) {
                    // Even(L) × Even(R) → Even
                    .even => .{
                        .c = .{
                            L[0] * R[0] - L[1] * R[1] - L[2] * R[2] - L[3] * R[3],
                            L[0] * R[1] + L[1] * R[0] + L[3] * R[2] - L[2] * R[3],
                            L[0] * R[2] + L[2] * R[0] - L[3] * R[1] + L[1] * R[3],
                            L[0] * R[3] + L[3] * R[0] + L[2] * R[1] - L[1] * R[2],
                        },
                        .parity = .even,
                    },
                    // Even(L) × Odd(R) → Odd
                    .odd => .{
                        .c = .{
                            L[0] * R[0] + L[1] * R[1] + L[2] * R[2] - L[3] * R[3],
                            L[0] * R[1] - L[1] * R[0] + L[2] * R[3] + L[3] * R[2],
                            L[0] * R[2] - L[1] * R[3] - L[2] * R[0] - L[3] * R[1],
                            L[0] * R[3] + L[1] * R[2] - L[2] * R[1] + L[3] * R[0],
                        },
                        .parity = .odd,
                    },
                },
                .odd => switch (self.parity) {
                    // Odd(L) × Even(R) → Odd
                    .even => .{
                        .c = .{
                            R[0] * L[0] - R[1] * L[1] - R[2] * L[2] - R[3] * L[3],
                            R[1] * L[0] + R[0] * L[1] - R[3] * L[2] + R[2] * L[3],
                            R[2] * L[0] + R[3] * L[1] + R[0] * L[2] - R[1] * L[3],
                            R[3] * L[0] - R[2] * L[1] + R[1] * L[2] + R[0] * L[3],
                        },
                        .parity = .odd,
                    },
                    // Odd(L) × Odd(R) → Even
                    .odd => .{
                        .c = .{
                            L[0] * R[0] + L[1] * R[1] + L[2] * R[2] - L[3] * R[3],
                            L[0] * R[1] - L[1] * R[0] + L[2] * R[3] + L[3] * R[2],
                            L[0] * R[2] - L[1] * R[3] - L[2] * R[0] - L[3] * R[1],
                            L[0] * R[3] + L[1] * R[2] - L[2] * R[1] + L[3] * R[0],
                        },
                        .parity = .even,
                    },
                },
            };
        }

        // -----------------------------------------------------------------
        // Application — v → α(Ṽ) · v · V
        // -----------------------------------------------------------------

        /// Transform a 3D vector by this versor (sandwich product).
        pub inline fn apply(self: Self, v: Vec3S) Vec3S {
            return switch (self.parity) {
                .even => {
                    // For even versors, α(Ṽ) = Ṽ. This is the same rotation as
                    // Rot3.rotate but via the vector sandwich with GA sign convention.
                    //
                    // The optimized formula: v + a·t + cross(bv, t)
                    // where bv is the Hodge-dual axis vector and t = 2·cross(bv, v).
                    //
                    // GA component mapping to axis vector:
                    //   e₂₃ → x, −e₁₃ → y, e₁₂ → z
                    // (the negation on e₁₃ is the Hodge dual sign)
                    const a = self.c[0];
                    const bx = self.c[3]; // e₂₃
                    const by = -self.c[2]; // −e₁₃
                    const bz = self.c[1]; // e₁₂

                    const tx: Scalar = 2 * (by * v.v[2] - bz * v.v[1]);
                    const ty: Scalar = 2 * (bz * v.v[0] - bx * v.v[2]);
                    const tz: Scalar = 2 * (bx * v.v[1] - by * v.v[0]);

                    return .{ .v = .{
                        v.v[0] + a * tx + (by * tz - bz * ty),
                        v.v[1] + a * ty + (bz * tx - bx * tz),
                        v.v[2] + a * tz + (bx * ty - by * tx),
                    } };
                },
                .odd => {
                    // For odd versors, α(Ṽ) = −Ṽ.
                    // For a pure reflection (c[3]=0): v − 2(v·n)n
                    // For a general rotoreflection, use the full formula.
                    //
                    // Compute via two Clifford products:
                    //   P = V · v  (Odd × Odd → Even)
                    //   result = −(P · Ṽ) extracted vector part
                    // where Ṽ = (c[0], c[1], c[2], −c[3]).
                    //
                    // Optimized inline expansion:
                    const x = self.c[0];
                    const y = self.c[1];
                    const z = self.c[2];
                    const w = self.c[3];
                    const vx = v.v[0];
                    const vy = v.v[1];
                    const vz = v.v[2];

                    // Step 1: P = self · v_as_odd  (Odd × Odd → Even)
                    // v_as_odd = (vx, vy, vz, 0)
                    const pa = x * vx + y * vy + z * vz; // scalar
                    const p1 = x * vy - y * vx + w * vz; // e₁₂
                    const p2 = x * vz - w * vy - z * vx; // e₁₃
                    const p3 = w * vx + y * vz - z * vy; // e₂₃

                    // Step 2: −(P · Ṽ) where Ṽ = (x, y, z, −w)
                    // P(even) × Ṽ(odd) → Odd, then negate.
                    // Using Even(pa,p1,p2,p3) × Odd(x, y, z, −w) → Odd
                    const nw = -w;
                    return .{ .v = .{
                        -(pa * x + p1 * y + p2 * z - p3 * nw),
                        -(pa * y - p1 * x + p2 * nw + p3 * z),
                        -(pa * z - p1 * nw - p2 * x - p3 * y),
                    } };
                },
            };
        }

        // -----------------------------------------------------------------
        // Reverse / inverse
        // -----------------------------------------------------------------

        /// Reverse (conjugate). For unit versors, this is the inverse transformation.
        /// Even: negate grade-2 (bivectors). Odd: negate grade-3 (trivector), keep grade-1.
        pub inline fn reverse(self: Self) Self {
            return switch (self.parity) {
                .even => .{ .c = .{ self.c[0], -self.c[1], -self.c[2], -self.c[3] }, .parity = .even },
                .odd => .{ .c = .{ self.c[0], self.c[1], self.c[2], -self.c[3] }, .parity = .odd },
            };
        }

        // -----------------------------------------------------------------
        // Normalization
        // -----------------------------------------------------------------

        /// Squared norm of the versor.
        pub inline fn lenSqr(self: Self) Scalar {
            return self.c[0] * self.c[0] + self.c[1] * self.c[1] + self.c[2] * self.c[2] + self.c[3] * self.c[3];
        }

        /// Normalize to unit length.
        pub inline fn normalize(self: Self) Self {
            const inv_len = 1.0 / @sqrt(self.lenSqr());
            return .{
                .c = .{ self.c[0] * inv_len, self.c[1] * inv_len, self.c[2] * inv_len, self.c[3] * inv_len },
                .parity = self.parity,
            };
        }

        // -----------------------------------------------------------------
        // Conversion
        // -----------------------------------------------------------------

        /// Convert to Rot3 if even. Returns null if odd.
        /// Negates the e₁₃ component to convert from GA back to quaternion convention.
        pub inline fn toRotor(self: Self) ?Rot3S {
            return switch (self.parity) {
                .even => .{ .a = self.c[0], .b01 = self.c[1], .b02 = -self.c[2], .b12 = self.c[3] },
                .odd => null,
            };
        }

        /// Convert to a 3×3 matrix. Even → rotation (det +1), odd → reflection (det −1).
        pub inline fn toMat3(self: Self) Mat3S {
            return switch (self.parity) {
                .even => {
                    // Same as Rot3.toMat3 but via GA component mapping.
                    // Map GA components to quaternion: w=c[0], x=c[3](e₂₃), y=−c[2](e₁₃), z=c[1](e₁₂)
                    const w = self.c[0];
                    const ix = self.c[3];
                    const iy = -self.c[2];
                    const iz = self.c[1];

                    const x2 = ix + ix;
                    const y2 = iy + iy;
                    const z2 = iz + iz;
                    const xx = ix * x2;
                    const xy = ix * y2;
                    const xz = ix * z2;
                    const yy = iy * y2;
                    const yz = iy * z2;
                    const zz = iz * z2;
                    const wx = w * x2;
                    const wy = w * y2;
                    const wz = w * z2;

                    return .{ .m = .{
                        .{ 1 - (yy + zz), xy + wz, xz - wy },
                        .{ xy - wz, 1 - (xx + zz), yz + wx },
                        .{ xz + wy, yz - wx, 1 - (xx + yy) },
                    } };
                },
                .odd => {
                    // Build matrix by applying the versor to each basis vector.
                    const e1 = self.apply(Vec3S.init(.{ 1, 0, 0 }));
                    const e2 = self.apply(Vec3S.init(.{ 0, 1, 0 }));
                    const e3 = self.apply(Vec3S.init(.{ 0, 0, 1 }));
                    return .{ .m = .{
                        .{ e1.v[0], e1.v[1], e1.v[2] },
                        .{ e2.v[0], e2.v[1], e2.v[2] },
                        .{ e3.v[0], e3.v[1], e3.v[2] },
                    } };
                },
            };
        }

        /// Convert to a 4×4 matrix (upper-left 3×3, rest is identity).
        pub inline fn toMat4(self: Self) Mat4S {
            const m3 = self.toMat3();
            return .{ .m = .{
                .{ m3.m[0][0], m3.m[0][1], m3.m[0][2], 0 },
                .{ m3.m[1][0], m3.m[1][1], m3.m[1][2], 0 },
                .{ m3.m[2][0], m3.m[2][1], m3.m[2][2], 0 },
                .{ 0, 0, 0, 1 },
            } };
        }

        /// Determinant of the transformation: +1 for even (rotation), −1 for odd (reflection).
        pub inline fn det(self: Self) Scalar {
            return switch (self.parity) {
                .even => 1,
                .odd => -1,
            };
        }

        // -----------------------------------------------------------------
        // Interpolation (within same parity)
        // -----------------------------------------------------------------

        /// Normalized linear interpolation. Both versors must have the same parity.
        pub inline fn nlerp(from: Self, to: Self, t: Scalar) Self {
            std.debug.assert(from.parity == to.parity);
            var to_adj = to;
            const d = from.c[0] * to.c[0] + from.c[1] * to.c[1] + from.c[2] * to.c[2] + from.c[3] * to.c[3];
            if (d < 0) {
                to_adj = .{ .c = .{ -to.c[0], -to.c[1], -to.c[2], -to.c[3] }, .parity = to.parity };
            }
            return (Self{
                .c = .{
                    from.c[0] + (to_adj.c[0] - from.c[0]) * t,
                    from.c[1] + (to_adj.c[1] - from.c[1]) * t,
                    from.c[2] + (to_adj.c[2] - from.c[2]) * t,
                    from.c[3] + (to_adj.c[3] - from.c[3]) * t,
                },
                .parity = from.parity,
            }).normalize();
        }

        /// Spherical linear interpolation. Both versors must have the same parity.
        pub inline fn slerp(from: Self, to: Self, t: Scalar) Self {
            std.debug.assert(from.parity == to.parity);
            var d = from.c[0] * to.c[0] + from.c[1] * to.c[1] + from.c[2] * to.c[2] + from.c[3] * to.c[3];
            var to_adj = to;
            if (d < 0) {
                d = -d;
                to_adj = .{ .c = .{ -to.c[0], -to.c[1], -to.c[2], -to.c[3] }, .parity = to.parity };
            }
            if (d > 0.9995) {
                return nlerp(from, to_adj, t);
            }
            const theta = math.acos(d);
            const sin_theta = @sin(theta);
            const s0 = @sin((1 - t) * theta) / sin_theta;
            const s1 = @sin(t * theta) / sin_theta;
            return Self{
                .c = .{
                    from.c[0] * s0 + to_adj.c[0] * s1,
                    from.c[1] * s0 + to_adj.c[1] * s1,
                    from.c[2] * s0 + to_adj.c[2] * s1,
                    from.c[3] * s0 + to_adj.c[3] * s1,
                },
                .parity = from.parity,
            };
        }
    };
}

/// Affine reflection across a 2D line.
/// `p' = p − 2·dot(p − origin, normal) · normal`
pub fn Mirror2Type(comptime Scalar: type) type {
    const Vec2S = VecType(2, Scalar);
    const Versor2S = Versor2Type(Scalar);

    return struct {
        const Self = @This();

        /// Unit normal of the mirror line.
        normal: Vec2S,
        /// A point on the mirror line. Defaults to the origin.
        origin: Vec2S = Vec2S.zero,

        pub const acrossY = Self{ .normal = Vec2S.init(.{ 1, 0 }) };
        pub const acrossX = Self{ .normal = Vec2S.init(.{ 0, 1 }) };

        pub fn fromNormal(normal: Vec2S) Self {
            return .{ .normal = normal };
        }

        pub fn fromPlane(normal: Vec2S, origin: Vec2S) Self {
            return .{ .normal = normal, .origin = origin };
        }

        /// Reflect a point (affine — accounts for origin offset).
        pub fn applyPoint(self: Self, point: Vec2S) Vec2S {
            const diff = point.sub(self.origin);
            const d: Scalar = @as(Scalar, 2) * diff.dot(self.normal);
            return point.sub(self.normal.scale(d));
        }

        /// Reflect a direction (translation-invariant — ignores origin).
        pub fn applyDir(self: Self, dir: Vec2S) Vec2S {
            const d: Scalar = @as(Scalar, 2) * dir.dot(self.normal);
            return dir.sub(self.normal.scale(d));
        }

        /// Return the underlying odd versor (linear part only, no origin offset).
        pub fn toVersor(self: Self) Versor2S {
            return Versor2S.fromReflection(self.normal);
        }
    };
}

/// Affine reflection across a 3D plane.
/// `p' = p − 2·dot(p − origin, normal) · normal`
pub fn Mirror3Type(comptime Scalar: type) type {
    const Vec3S = VecType(3, Scalar);
    const Versor3S = Versor3Type(Scalar);

    return struct {
        const Self = @This();

        /// Unit normal of the mirror plane.
        normal: Vec3S,
        /// A point on the mirror plane. Defaults to the origin.
        origin: Vec3S = Vec3S.zero,

        pub const acrossYZ = Self{ .normal = Vec3S.init(.{ 1, 0, 0 }) };
        pub const acrossXZ = Self{ .normal = Vec3S.init(.{ 0, 1, 0 }) };
        pub const acrossXY = Self{ .normal = Vec3S.init(.{ 0, 0, 1 }) };

        pub fn fromNormal(normal: Vec3S) Self {
            return .{ .normal = normal };
        }

        pub fn fromPlane(normal: Vec3S, origin: Vec3S) Self {
            return .{ .normal = normal, .origin = origin };
        }

        /// Reflect a point (affine — accounts for origin offset).
        pub fn applyPoint(self: Self, point: Vec3S) Vec3S {
            const diff = point.sub(self.origin);
            const d: Scalar = @as(Scalar, 2) * diff.dot(self.normal);
            return point.sub(self.normal.scale(d));
        }

        /// Reflect a direction (translation-invariant — ignores origin).
        pub fn applyDir(self: Self, dir: Vec3S) Vec3S {
            const d: Scalar = @as(Scalar, 2) * dir.dot(self.normal);
            return dir.sub(self.normal.scale(d));
        }

        /// Return the underlying odd versor (linear part only, no origin offset).
        pub fn toVersor(self: Self) Versor3S {
            return Versor3S.fromReflection(self.normal);
        }
    };
}

/// Spatial transform combining translation, rotation, and non-uniform scale.
/// Composes to Mat4 in TRS order: scale first, then rotation, then translation.
pub fn TransformType(comptime Scalar: type) type {
    const Vec3S = VecType(3, Scalar);
    const Mat4S = MatType(4, Scalar);
    const Rot3S = Rotor3Type(Scalar);

    return struct {
        const Self = @This();

        translation: Vec3S = Vec3S.zero,
        rotation: Rot3S = Rot3S.identity,
        scaling: Vec3S = Vec3S.init(.{ 1, 1, 1 }),

        pub const identity: Self = .{};

        /// Convert to a column-major 4×4 matrix: T * R * S.
        pub fn toMat4(self: Self) Mat4S {
            const t = Mat4S.translation(self.translation);
            const r = self.rotation.toMat4();
            const s = Mat4S.scaling(self.scaling);
            return t.mul(r.mul(s));
        }

        pub fn fromTranslation(v: Vec3S) Self {
            return .{ .translation = v };
        }

        pub fn fromRotation(r: Rot3S) Self {
            return .{ .rotation = r };
        }
    };
}

const Rot2 = Rotor2Type(f32);
const Rot3 = Rotor3Type(f32);

const Rot2d = Rotor2Type(f64);
const Rot3d = Rotor3Type(f64);

const Versor2 = Versor2Type(f32);
const Versor3 = Versor3Type(f32);

const Versor2d = Versor2Type(f64);
const Versor3d = Versor3Type(f64);

const Mirror2 = Mirror2Type(f32);
const Mirror3 = Mirror3Type(f32);

const Transform = TransformType(f32);

const eps_normal: f32 = @sqrt(math.floatEps(f32));
const eps_loose: f32 = 1e-3;

test "Vec2: arithmetic" {
    const a = Vec2.init(.{ 1, 2 });
    const b = Vec2.init(.{ 3, 4 });

    try std.testing.expect(a.add(b).eql(Vec2.init(.{ 4, 6 }), 0));
    try std.testing.expect(a.sub(b).eql(Vec2.init(.{ -2, -2 }), 0));
    try std.testing.expect(a.mul(b).eql(Vec2.init(.{ 3, 8 }), 0));
    try std.testing.expect(a.scale(2).eql(Vec2.init(.{ 2, 4 }), 0));
    try std.testing.expect(a.neg().eql(Vec2.init(.{ -1, -2 }), 0));
}

test "Vec3: arithmetic" {
    const a = Vec3.init(.{ 1, 2, 3 });
    const b = Vec3.init(.{ 4, 5, 6 });

    try std.testing.expect(a.add(b).eql(Vec3.init(.{ 5, 7, 9 }), 0));
    try std.testing.expect(a.sub(b).eql(Vec3.init(.{ -3, -3, -3 }), 0));
    try std.testing.expect(a.scale(-1).eql(a.neg(), 0));
}

test "Vec: dot product" {
    const a = Vec3.init(.{ 1, 2, 3 });
    const b = Vec3.init(.{ 4, 5, 6 });
    // 1*4 + 2*5 + 3*6 = 32
    try std.testing.expectEqual(@as(f32, 32), a.dot(b));

    // Perpendicular vectors have dot = 0
    const x = Vec2.init(.{ 1, 0 });
    const y = Vec2.init(.{ 0, 1 });
    try std.testing.expectEqual(@as(f32, 0), x.dot(y));
}

test "Vec: length and normalize" {
    const v = Vec3.init(.{ 3, 4, 0 });
    try std.testing.expectEqual(@as(f32, 25), v.lenSqr());
    try std.testing.expectEqual(@as(f32, 5), v.len());

    const n = v.normalize();
    const eps = @sqrt(math.floatEps(f32));
    try std.testing.expect(@abs(n.len() - 1.0) < eps);
    try std.testing.expect(@abs(n.v[0] - 0.6) < eps);
    try std.testing.expect(@abs(n.v[1] - 0.8) < eps);
}

test "Vec: lerp" {
    const a = Vec3.init(.{ 0, 0, 0 });
    const b = Vec3.init(.{ 10, 20, 30 });

    try std.testing.expect(a.lerp(b, 0).eql(a, 0));
    try std.testing.expect(a.lerp(b, 1).eql(b, 0));
    try std.testing.expect(a.lerp(b, 0.5).eql(Vec3.init(.{ 5, 10, 15 }), 0));
}

test "Vec: project" {
    const v = Vec2.init(.{ 3, 4 });
    const onto = Vec2.init(.{ 1, 0 });
    const proj = v.project(onto);
    try std.testing.expect(proj.eql(Vec2.init(.{ 3, 0 }), 0));
}

test "Vec: per-component operations" {
    const v = Vec3.init(.{ -3, 0, 5 });
    try std.testing.expect(v.abs().eql(Vec3.init(.{ 3, 0, 5 }), 0));
    try std.testing.expect(v.sign().eql(Vec3.init(.{ -1, 0, 1 }), 0));

    const a = Vec3.init(.{ 1, 5, 3 });
    const b = Vec3.init(.{ 4, 2, 6 });
    try std.testing.expect(a.min(b).eql(Vec3.init(.{ 1, 2, 3 }), 0));
    try std.testing.expect(a.max(b).eql(Vec3.init(.{ 4, 5, 6 }), 0));
}

test "Vec: reductions" {
    const v = Vec3.init(.{ 1, 5, 3 });
    try std.testing.expectEqual(@as(f32, 1), v.minElement());
    try std.testing.expectEqual(@as(f32, 5), v.maxElement());
}

test "Vec: floor, ceil, round" {
    const v = Vec2.init(.{ 1.3, -2.7 });
    try std.testing.expect(v.floor().eql(Vec2.init(.{ 1, -3 }), 0));
    try std.testing.expect(v.ceil().eql(Vec2.init(.{ 2, -2 }), 0));
    try std.testing.expect(v.round().eql(Vec2.init(.{ 1, -3 }), 0));
}

test "Vec2: perp and angle" {
    const v = Vec2.init(.{ 1, 0 });
    try std.testing.expect(v.perp().eql(Vec2.init(.{ 0, 1 }), 0));
    try std.testing.expectEqual(@as(f32, 0), v.angle());

    const eps = @sqrt(math.floatEps(f32));
    const up = Vec2.init(.{ 0, 1 });
    try std.testing.expect(@abs(up.angle() - math.pi / 2.0) < eps);
}

test "Vec2: 2D cross product" {
    const a = Vec2.init(.{ 1, 0 });
    const b = Vec2.init(.{ 0, 1 });
    try std.testing.expectEqual(@as(f32, 1), a.cross(b));
    try std.testing.expectEqual(@as(f32, -1), b.cross(a));
}

test "Vec3: 3D cross product" {
    const x = Vec3.init(.{ 1, 0, 0 });
    const y = Vec3.init(.{ 0, 1, 0 });
    const z = Vec3.init(.{ 0, 0, 1 });

    try std.testing.expect(x.cross(y).eql(z, 0));
    try std.testing.expect(y.cross(z).eql(x, 0));
    try std.testing.expect(z.cross(x).eql(y, 0));

    // Anti-commutativity
    try std.testing.expect(y.cross(x).eql(z.neg(), 0));
}

test "Vec: integer operations" {
    const a = Vec3i.init(.{ 1, 2, 3 });
    const b = Vec3i.init(.{ 4, 5, 6 });

    try std.testing.expect(a.add(b).eql(Vec3i.init(.{ 5, 7, 9 })));
    try std.testing.expect(a.mul(b).eql(Vec3i.init(.{ 4, 10, 18 })));
    try std.testing.expectEqual(@as(i32, 32), a.dot(b));
}

test "Vec: constants" {
    try std.testing.expect(Vec3.zero.eql(Vec3.init(.{ 0, 0, 0 }), 0));
    try std.testing.expect(Vec3.one.eql(Vec3.init(.{ 1, 1, 1 }), 0));
    try std.testing.expect(Vec3.splat(7).eql(Vec3.init(.{ 7, 7, 7 }), 0));
}

test "Mat: identity" {
    const I = Mat3.identity;
    const v = Vec3.init(.{ 2, 3, 4 });
    try std.testing.expect(I.mulVec(v).eql(v, 0));
    try std.testing.expect(I.mul(I).eql(I, 0));
}

test "Mat: fromCols, col, row" {
    const c0 = Vec2.init(.{ 1, 2 });
    const c1 = Vec2.init(.{ 3, 4 });
    const m = Mat2.fromCols(.{ c0, c1 });

    try std.testing.expect(m.col(0).eql(c0, 0));
    try std.testing.expect(m.col(1).eql(c1, 0));
    try std.testing.expect(m.row(0).eql(Vec2.init(.{ 1, 3 }), 0));
    try std.testing.expect(m.row(1).eql(Vec2.init(.{ 2, 4 }), 0));
}

test "Mat: arithmetic" {
    const a = Mat2.fromCols(.{
        Vec2.init(.{ 1, 2 }),
        Vec2.init(.{ 3, 4 }),
    });
    const b = Mat2.fromCols(.{
        Vec2.init(.{ 5, 6 }),
        Vec2.init(.{ 7, 8 }),
    });

    const sum = a.add(b);
    try std.testing.expect(sum.col(0).eql(Vec2.init(.{ 6, 8 }), 0));
    try std.testing.expect(sum.col(1).eql(Vec2.init(.{ 10, 12 }), 0));

    const diff = a.sub(b);
    try std.testing.expect(diff.col(0).eql(Vec2.init(.{ -4, -4 }), 0));
    try std.testing.expect(diff.col(1).eql(Vec2.init(.{ -4, -4 }), 0));

    const scaled = a.scale(2);
    try std.testing.expect(scaled.col(0).eql(Vec2.init(.{ 2, 4 }), 0));

    try std.testing.expect(a.neg().add(a).eql(Mat2.zero, 0));
}

test "Mat2: multiply" {
    // Column-major: m[col][row]
    // A = [[1,3],[2,4]] (row-major reading: row0=[1,3], row1=[2,4])
    const a = Mat2.fromCols(.{
        Vec2.init(.{ 1, 2 }),
        Vec2.init(.{ 3, 4 }),
    });
    // B = [[5,7],[6,8]]
    const b = Mat2.fromCols(.{
        Vec2.init(.{ 5, 6 }),
        Vec2.init(.{ 7, 8 }),
    });
    // AB: col0 = A * b_col0 = [1*5+3*6, 2*5+4*6] = [23, 34]
    //     col1 = A * b_col1 = [1*7+3*8, 2*7+4*8] = [31, 46]
    const ab = a.mul(b);
    try std.testing.expect(ab.col(0).eql(Vec2.init(.{ 23, 34 }), 0));
    try std.testing.expect(ab.col(1).eql(Vec2.init(.{ 31, 46 }), 0));
}

test "Mat: mulVec" {
    const m = Mat3.fromCols(.{
        Vec3.init(.{ 1, 0, 0 }),
        Vec3.init(.{ 0, 1, 0 }),
        Vec3.init(.{ 0, 0, 1 }),
    });
    const v = Vec3.init(.{ 2, 3, 4 });
    try std.testing.expect(m.mulVec(v).eql(v, 0));

    // Scale x by 2
    const s = Mat3.fromCols(.{
        Vec3.init(.{ 2, 0, 0 }),
        Vec3.init(.{ 0, 1, 0 }),
        Vec3.init(.{ 0, 0, 1 }),
    });
    try std.testing.expect(s.mulVec(v).eql(Vec3.init(.{ 4, 3, 4 }), 0));
}

test "Mat: transpose" {
    const m = Mat2.fromCols(.{
        Vec2.init(.{ 1, 2 }),
        Vec2.init(.{ 3, 4 }),
    });
    const mt = m.transpose();
    try std.testing.expect(mt.col(0).eql(Vec2.init(.{ 1, 3 }), 0));
    try std.testing.expect(mt.col(1).eql(Vec2.init(.{ 2, 4 }), 0));

    // Double transpose = original
    try std.testing.expect(mt.transpose().eql(m, 0));
}

test "Mat: transpose of identity is identity" {
    try std.testing.expect(Mat3.identity.transpose().eql(Mat3.identity, 0));
    try std.testing.expect(Mat4.identity.transpose().eql(Mat4.identity, 0));
}

test "Mat: trace" {
    const m = Mat3.fromCols(.{
        Vec3.init(.{ 2, 0, 0 }),
        Vec3.init(.{ 0, 3, 0 }),
        Vec3.init(.{ 0, 0, 5 }),
    });
    try std.testing.expectEqual(@as(f32, 10), m.trace());
    try std.testing.expectEqual(@as(f32, 3), Mat3.identity.trace());
}

test "Mat2: det" {
    const m = Mat2.fromCols(.{
        Vec2.init(.{ 1, 3 }),
        Vec2.init(.{ 2, 4 }),
    });
    // det = 1*4 - 2*3 = -2
    try std.testing.expectEqual(@as(f32, -2), m.det());
    try std.testing.expectEqual(@as(f32, 1), Mat2.identity.det());
}

test "Mat3: det" {
    const m = Mat3.fromCols(.{
        Vec3.init(.{ 1, 4, 7 }),
        Vec3.init(.{ 2, 5, 8 }),
        Vec3.init(.{ 3, 6, 10 }),
    });

    // This is a well-known matrix with det = -3
    const eps: f32 = 1e-5;
    try std.testing.expect(@abs(m.det() - (-3.0)) < eps);
    try std.testing.expectEqual(@as(f32, 1), Mat3.identity.det());
}

test "Mat4: det" {
    try std.testing.expectEqual(@as(f32, 1), Mat4.identity.det());

    // Scaling matrix: det = product of diagonal
    const s = Mat4.scaling(Vec3.init(.{ 2, 3, 4 }));
    try std.testing.expectEqual(@as(f32, 24), s.det());

    // Rotation matrices have det = 1
    const pi = math.pi;
    const r = Mat4.rotation(Vec3.init(.{ 0, 1, 0 }), pi / 3.0);
    const eps: f32 = 1e-5;
    try std.testing.expect(@abs(r.det() - 1.0) < eps);
}

test "Mat2: inverse" {
    const m = Mat2.fromCols(.{
        Vec2.init(.{ 4, 3 }),
        Vec2.init(.{ 2, 1 }),
    });
    const inv = m.inverse();
    // M * M^-1 = I
    if (!m.mul(inv).eql(Mat2.identity, eps_normal)) return error.TestUnexpectedResult;
    if (!inv.mul(m).eql(Mat2.identity, eps_normal)) return error.TestUnexpectedResult;
}

test "Mat3: inverse" {
    const m = Mat3.fromCols(.{
        Vec3.init(.{ 1, 4, 7 }),
        Vec3.init(.{ 2, 5, 8 }),
        Vec3.init(.{ 3, 6, 10 }),
    });
    const inv = m.inverse();
    if (!m.mul(inv).eql(Mat3.identity, eps_normal)) return error.TestUnexpectedResult;
    if (!inv.mul(m).eql(Mat3.identity, eps_normal)) return error.TestUnexpectedResult;
}

test "Mat4: inverse" {
    // Translation inverse
    const t = Mat4.translation(Vec3.init(.{ 3, 4, 5 }));
    const t_inv = t.inverse();
    if (!t.mul(t_inv).eql(Mat4.identity, eps_normal)) return error.TestUnexpectedResult;

    // Rotation inverse
    const pi = math.pi;
    const r = Mat4.rotation(Vec3.init(.{ 1, 1, 1 }).normalize(), pi / 3.0);
    const r_inv = r.inverse();
    if (!r.mul(r_inv).eql(Mat4.identity, eps_normal)) return error.TestUnexpectedResult;

    // Combined transform
    const m = t.mul(r).mul(Mat4.scaling(Vec3.init(.{ 2, 3, 4 })));
    const m_inv = m.inverse();
    if (!m.mul(m_inv).eql(Mat4.identity, eps_normal)) return error.TestUnexpectedResult;
    if (!m_inv.mul(m).eql(Mat4.identity, eps_normal)) return error.TestUnexpectedResult;
}

test "Mat4: identity mulVec" {
    const v = Vec4.init(.{ 1, 2, 3, 4 });
    try std.testing.expect(Mat4.identity.mulVec(v).eql(v, 0));
}

test "Mat: mul associativity" {
    const a = Mat2.fromCols(.{
        Vec2.init(.{ 1, 2 }),
        Vec2.init(.{ 3, 4 }),
    });
    const b = Mat2.fromCols(.{
        Vec2.init(.{ 5, 6 }),
        Vec2.init(.{ 7, 8 }),
    });
    const c = Mat2.fromCols(.{
        Vec2.init(.{ 9, 10 }),
        Vec2.init(.{ 11, 12 }),
    });

    // (A*B)*C == A*(B*C)
    try std.testing.expect(a.mul(b).mul(c).eql(a.mul(b.mul(c)), 0));
}

test "Rotor2: fromAngle + rotate" {
    const pi = math.pi;
    const e1 = Vec2.init(.{ 1, 0 });
    const e2 = Vec2.init(.{ 0, 1 });

    // 90° CCW: e1 → e2
    const r90 = Rot2.fromAngle(pi / 2.0);
    if (!r90.rotate(e1).eql(e2, eps_normal)) return error.TestUnexpectedResult;

    // 180°: e1 → -e1
    const r180 = Rot2.fromAngle(pi);
    if (!r180.rotate(e1).eql(Vec2.init(.{ -1, 0 }), eps_normal)) return error.TestUnexpectedResult;

    // 45°
    const r45 = Rot2.fromAngle(pi / 4.0);
    const s = @sqrt(2.0) / 2.0;
    if (!r45.rotate(e1).eql(Vec2.init(.{ s, s }), eps_normal)) return error.TestUnexpectedResult;
}

test "Rotor2: fromVecs" {
    const e1 = Vec2.init(.{ 1, 0 });
    const e2 = Vec2.init(.{ 0, 1 });

    // from e1 to e2 (90° CCW)
    const r = Rot2.fromVecs(e1, e2);
    if (!r.rotate(e1).eql(e2, eps_normal)) return error.TestUnexpectedResult;

    // from e2 to e1 (90° CW)
    const r2 = Rot2.fromVecs(e2, e1);
    if (!r2.rotate(e2).eql(e1, eps_normal)) return error.TestUnexpectedResult;

    // 180° edge case
    const r180 = Rot2.fromVecs(e1, Vec2.init(.{ -1, 0 }));
    if (!r180.rotate(e1).eql(Vec2.init(.{ -1, 0 }), eps_normal)) return error.TestUnexpectedResult;
}

test "Rotor2: mul composes rotations" {
    const pi = math.pi;
    const e1 = Vec2.init(.{ 1, 0 });

    // Two 45° rotations = one 90°
    const r45 = Rot2.fromAngle(pi / 4.0);
    const r90 = r45.mul(r45);
    if (!r90.rotate(e1).eql(Vec2.init(.{ 0, 1 }), eps_normal)) return error.TestUnexpectedResult;
}

test "Rotor2: reverse undoes rotation" {
    const pi = math.pi;
    const v = Vec2.init(.{ 3, 4 });
    const r = Rot2.fromAngle(pi / 3.0);
    const rotated = r.rotate(v);
    const back = r.reverse().rotate(rotated);
    if (!back.eql(v, eps_normal)) return error.TestUnexpectedResult;
}

test "Rotor2: angle roundtrip" {
    const pi = math.pi;
    const r = Rot2.fromAngle(1.23);
    if (@abs(r.angle() - 1.23) > 1e-5) return error.TestUnexpectedResult;

    const r2 = Rot2.fromAngle(-pi / 6.0);
    if (@abs(r2.angle() - (-pi / 6.0)) > 1e-5) return error.TestUnexpectedResult;
}

test "Rotor2: toMat2 matches rotate" {
    const pi = math.pi;
    const r = Rot2.fromAngle(pi / 3.0);
    const m = r.toMat2();
    const v = Vec2.init(.{ 2, -1 });
    if (!m.mulVec(v).eql(r.rotate(v), eps_normal)) return error.TestUnexpectedResult;
}

test "Rotor2: slerp endpoints" {
    const pi = math.pi;
    const r0 = Rot2.fromAngle(0);
    const r1 = Rot2.fromAngle(pi / 2.0);

    const s0 = Rot2.slerp(r0, r1, 0);
    if (!(@abs(s0.a - r0.a) <= eps_normal and @abs(s0.b - r0.b) <= eps_normal)) return error.TestUnexpectedResult;

    const s1 = Rot2.slerp(r0, r1, 1);
    if (!(@abs(s1.a - r1.a) <= eps_normal and @abs(s1.b - r1.b) <= eps_normal)) return error.TestUnexpectedResult;

    // Midpoint should be 45°
    const smid = Rot2.slerp(r0, r1, 0.5);
    if (@abs(smid.angle() - pi / 4.0) > 1e-5) return error.TestUnexpectedResult;
}

test "Rotor3: fromAxisAngle + rotate" {
    const pi = math.pi;
    const e1 = Vec3.init(.{ 1, 0, 0 });
    const e2 = Vec3.init(.{ 0, 1, 0 });
    const e3 = Vec3.init(.{ 0, 0, 1 });

    // 90° around z: e1 → e2
    const rz = Rot3.fromAxisAngle(pi / 2.0, e3);
    if (!rz.rotate(e1).eql(e2, eps_normal)) return error.TestUnexpectedResult;

    // 90° around y: e1 → -e3
    const ry = Rot3.fromAxisAngle(pi / 2.0, e2);
    if (!ry.rotate(e1).eql(Vec3.init(.{ 0, 0, -1 }), eps_normal)) return error.TestUnexpectedResult;

    // 90° around x: e2 → e3
    const rx = Rot3.fromAxisAngle(pi / 2.0, e1);
    if (!rx.rotate(e2).eql(e3, eps_normal)) return error.TestUnexpectedResult;
}

test "Rotor3: fromVecs" {
    const e1 = Vec3.init(.{ 1, 0, 0 });
    const e2 = Vec3.init(.{ 0, 1, 0 });
    const e3 = Vec3.init(.{ 0, 0, 1 });

    // e1 → e2
    const r = Rot3.fromVecs(e1, e2);
    if (!r.rotate(e1).eql(e2, eps_normal)) return error.TestUnexpectedResult;

    // e1 → e3
    const r2 = Rot3.fromVecs(e1, e3);
    if (!r2.rotate(e1).eql(e3, eps_normal)) return error.TestUnexpectedResult;

    // 180° edge case: e1 → -e1
    const r180 = Rot3.fromVecs(e1, Vec3.init(.{ -1, 0, 0 }));
    if (!r180.rotate(e1).eql(Vec3.init(.{ -1, 0, 0 }), eps_normal)) return error.TestUnexpectedResult;
}

test "Rotor3: mul composes rotations" {
    const pi = math.pi;
    const e1 = Vec3.init(.{ 1, 0, 0 });
    const e3 = Vec3.init(.{ 0, 0, 1 });

    // Two 45° around z = one 90° around z
    const r45 = Rot3.fromAxisAngle(pi / 4.0, e3);
    const r90 = r45.mul(r45);
    if (!r90.rotate(e1).eql(Vec3.init(.{ 0, 1, 0 }), eps_normal)) return error.TestUnexpectedResult;
}

test "Rotor3: reverse undoes rotation" {
    const pi = math.pi;
    const v = Vec3.init(.{ 1, 2, 3 });
    const r = Rot3.fromAxisAngle(pi / 5.0, Vec3.init(.{ 1, 1, 1 }).normalize());
    const back = r.reverse().rotate(r.rotate(v));
    if (!back.eql(v, eps_normal)) return error.TestUnexpectedResult;
}

test "Rotor3: toMat3 matches rotate" {
    const pi = math.pi;
    const r = Rot3.fromAxisAngle(pi / 3.0, Vec3.init(.{ 0, 1, 1 }).normalize());
    const m = r.toMat3();
    const v = Vec3.init(.{ 2, -1, 3 });
    if (!m.mulVec(v).eql(r.rotate(v), eps_normal)) return error.TestUnexpectedResult;
}

test "Rotor3: toMat4 matches rotate" {
    const pi = math.pi;
    const r = Rot3.fromAxisAngle(pi / 4.0, Vec3.init(.{ 1, 0, 0 }));
    const m = r.toMat4();
    const v = Vec4.init(.{ 2, -1, 3, 1 });
    const rotated = r.rotate(Vec3.init(.{ 2, -1, 3 }));
    const mat_result = m.mulVec(v);
    if (@abs(mat_result.v[0] - rotated.v[0]) > 1e-5) return error.TestUnexpectedResult;
    if (@abs(mat_result.v[1] - rotated.v[1]) > 1e-5) return error.TestUnexpectedResult;
    if (@abs(mat_result.v[2] - rotated.v[2]) > 1e-5) return error.TestUnexpectedResult;
    if (@abs(mat_result.v[3] - 1.0) > 1e-5) return error.TestUnexpectedResult;
}

test "Rotor3: slerp endpoints and midpoint" {
    const pi = math.pi;
    const e3 = Vec3.init(.{ 0, 0, 1 });
    const r0 = Rot3.identity;
    const r1 = Rot3.fromAxisAngle(pi / 2.0, e3);

    // t=0 → r0
    const s0 = Rot3.slerp(r0, r1, 0);
    if (!(@abs(s0.a - r0.a) <= eps_normal and @abs(s0.b01 - r0.b01) <= eps_normal and @abs(s0.b02 - r0.b02) <= eps_normal and @abs(s0.b12 - r0.b12) <= eps_normal)) return error.TestUnexpectedResult;

    // t=1 → r1
    const s1 = Rot3.slerp(r0, r1, 1);
    if (!(@abs(s1.a - r1.a) <= eps_normal and @abs(s1.b01 - r1.b01) <= eps_normal and @abs(s1.b02 - r1.b02) <= eps_normal and @abs(s1.b12 - r1.b12) <= eps_normal)) return error.TestUnexpectedResult;

    // t=0.5 → 45° around z
    const smid = Rot3.slerp(r0, r1, 0.5);
    const expected = Rot3.fromAxisAngle(pi / 4.0, e3);
    if (!(@abs(smid.a - expected.a) <= eps_normal and @abs(smid.b01 - expected.b01) <= eps_normal and @abs(smid.b02 - expected.b02) <= eps_normal and @abs(smid.b12 - expected.b12) <= eps_normal)) return error.TestUnexpectedResult;
}

test "Mat3: translation (2D homogeneous)" {
    const t = Mat3.translation(Vec2.init(.{ 5, 7 }));

    // Identity with last column set to (5, 7, 1)
    try std.testing.expect(t.col(0).eql(Vec3.init(.{ 1, 0, 0 }), 0));
    try std.testing.expect(t.col(1).eql(Vec3.init(.{ 0, 1, 0 }), 0));
    try std.testing.expect(t.col(2).eql(Vec3.init(.{ 5, 7, 1 }), 0));

    // Translating a 2D point (w=1)
    const p = Vec3.init(.{ 1, 2, 1 });
    try std.testing.expect(t.mulVec(p).eql(Vec3.init(.{ 6, 9, 1 }), 0));

    // Direction (w=0) unaffected
    const d = Vec3.init(.{ 1, 0, 0 });
    try std.testing.expect(t.mulVec(d).eql(d, 0));
}

test "Mat4: translation" {
    const t = Mat4.translation(Vec3.init(.{ 3, 4, 5 }));

    // Should be identity with column 3 set to (3, 4, 5, 1)
    try std.testing.expect(t.col(0).eql(Vec4.init(.{ 1, 0, 0, 0 }), 0));
    try std.testing.expect(t.col(1).eql(Vec4.init(.{ 0, 1, 0, 0 }), 0));
    try std.testing.expect(t.col(2).eql(Vec4.init(.{ 0, 0, 1, 0 }), 0));
    try std.testing.expect(t.col(3).eql(Vec4.init(.{ 3, 4, 5, 1 }), 0));

    // Translating a point (w=1)
    const p = Vec4.init(.{ 1, 2, 3, 1 });
    const tp = t.mulVec(p);
    try std.testing.expect(tp.eql(Vec4.init(.{ 4, 6, 8, 1 }), 0));

    // Translating a direction (w=0) should not change it
    const d = Vec4.init(.{ 1, 0, 0, 0 });
    try std.testing.expect(t.mulVec(d).eql(d, 0));
}

test "Mat2: scaling" {
    const s = Mat2.scaling(Vec2.init(.{ 3, 5 }));
    try std.testing.expect(s.col(0).eql(Vec2.init(.{ 3, 0 }), 0));
    try std.testing.expect(s.col(1).eql(Vec2.init(.{ 0, 5 }), 0));

    const v = Vec2.init(.{ 2, 4 });
    try std.testing.expect(s.mulVec(v).eql(Vec2.init(.{ 6, 20 }), 0));
}

test "Mat3: scaling (2D homogeneous)" {
    const s = Mat3.scaling(Vec2.init(.{ 2, 3 }));
    try std.testing.expect(s.col(0).eql(Vec3.init(.{ 2, 0, 0 }), 0));
    try std.testing.expect(s.col(1).eql(Vec3.init(.{ 0, 3, 0 }), 0));
    try std.testing.expect(s.col(2).eql(Vec3.init(.{ 0, 0, 1 }), 0));

    // Scale a 2D point (w=1)
    const p = Vec3.init(.{ 4, 5, 1 });
    try std.testing.expect(s.mulVec(p).eql(Vec3.init(.{ 8, 15, 1 }), 0));
}

test "Mat4: scaling" {
    const s = Mat4.scaling(Vec3.init(.{ 2, 3, 4 }));

    // Diagonal should be (2, 3, 4, 1)
    try std.testing.expect(s.col(0).eql(Vec4.init(.{ 2, 0, 0, 0 }), 0));
    try std.testing.expect(s.col(1).eql(Vec4.init(.{ 0, 3, 0, 0 }), 0));
    try std.testing.expect(s.col(2).eql(Vec4.init(.{ 0, 0, 4, 0 }), 0));
    try std.testing.expect(s.col(3).eql(Vec4.init(.{ 0, 0, 0, 1 }), 0));

    const v = Vec4.init(.{ 1, 1, 1, 1 });
    try std.testing.expect(s.mulVec(v).eql(Vec4.init(.{ 2, 3, 4, 1 }), 0));
}

test "Mat4: lookAt" {
    const eps = @sqrt(math.floatEps(f32));

    // Camera at (0,0,5), looking at origin, Y up
    const eye = Vec3.init(.{ 0, 0, 5 });
    const target = Vec3.init(.{ 0, 0, 0 });
    const up = Vec3.init(.{ 0, 1, 0 });
    const view = Mat4.lookAt(eye, target, up) orelse return error.TestUnexpectedResult;

    // Eye should map to origin
    const eye_h = Vec4.init(.{ 0, 0, 5, 1 });
    const result = view.mulVec(eye_h);
    try std.testing.expect(@abs(result.v[0]) < eps);
    try std.testing.expect(@abs(result.v[1]) < eps);
    try std.testing.expect(@abs(result.v[2]) < eps);

    // Target should be along -Z in view space (z < 0)
    const target_h = Vec4.init(.{ 0, 0, 0, 1 });
    const target_view = view.mulVec(target_h);
    try std.testing.expect(target_view.v[2] < 0);

    // Up direction should map to +Y
    const up_dir = Vec4.init(.{ 0, 1, 0, 0 });
    const up_view = view.mulVec(up_dir);
    try std.testing.expect(@abs(up_view.v[1] - 1.0) < eps);
}

test "Mat4: lookAt degenerate" {
    const eye = Vec3.init(.{ 0, 0, 5 });
    const target = Vec3.init(.{ 0, 0, 0 });

    // Up parallel to forward → null
    const up_parallel = Vec3.init(.{ 0, 0, -1 }); // same as fwd direction
    try std.testing.expect(Mat4.lookAt(eye, target, up_parallel) == null);

    // Eye == target → null
    try std.testing.expect(Mat4.lookAt(eye, eye, Vec3.init(.{ 0, 1, 0 })) == null);
}

test "Mat4: perspective" {
    const pi = math.pi;
    const p = Mat4.perspective(pi / 2.0, 1.0, 0.1, 100.0);
    const eps: f32 = 1e-5;

    // Near plane point (0,0,-near,1) should map to z=0 after perspective divide
    const near_pt = p.mulVec(Vec4.init(.{ 0, 0, -0.1, 1 }));
    const near_ndc_z = near_pt.v[2] / near_pt.v[3];
    try std.testing.expect(@abs(near_ndc_z) < eps);

    // Far plane point (0,0,-far,1) should map to z=1 after perspective divide
    const far_pt = p.mulVec(Vec4.init(.{ 0, 0, -100.0, 1 }));
    const far_ndc_z = far_pt.v[2] / far_pt.v[3];
    try std.testing.expect(@abs(far_ndc_z - 1.0) < eps);

    // Center point should map to (0,0) in xy
    const center = p.mulVec(Vec4.init(.{ 0, 0, -1.0, 1 }));
    try std.testing.expect(@abs(center.v[0]) < eps);
    try std.testing.expect(@abs(center.v[1]) < eps);
}

test "Mat4: ortho" {
    const o = Mat4.ortho(-1, 1, -1, 1, 0, 1);
    const eps: f32 = 1e-5;

    // Depth midpoint z=-0.5 (halfway between -near=0 and -far=-1) maps to 0.5
    const center = o.mulVec(Vec4.init(.{ 0, 0, -0.5, 1 }));
    try std.testing.expect(@abs(center.v[0]) < eps);
    try std.testing.expect(@abs(center.v[1]) < eps);
    try std.testing.expect(@abs(center.v[2] - 0.5) < eps);

    // Near plane z=-near=0 maps to 0
    const near_pt = o.mulVec(Vec4.init(.{ 0, 0, 0, 1 }));
    try std.testing.expect(@abs(near_pt.v[2]) < eps);

    // Far plane z=-far=-1 maps to 1
    const far_pt = o.mulVec(Vec4.init(.{ 0, 0, -1, 1 }));
    try std.testing.expect(@abs(far_pt.v[2] - 1.0) < eps);

    // Corners should map to ±1 in xy
    const corner = o.mulVec(Vec4.init(.{ 1, 1, 0, 1 }));
    try std.testing.expect(@abs(corner.v[0] - 1.0) < eps);
    try std.testing.expect(@abs(corner.v[1] - 1.0) < eps);

    const corner_neg = o.mulVec(Vec4.init(.{ -1, -1, 0, 1 }));
    try std.testing.expect(@abs(corner_neg.v[0] - (-1.0)) < eps);
    try std.testing.expect(@abs(corner_neg.v[1] - (-1.0)) < eps);
}

test "Mat2: rotation" {
    const pi = math.pi;
    const eps = @sqrt(math.floatEps(f32));

    // 90° CCW: (1,0) → (0,1)
    const r = Mat2.rotation(pi / 2.0);
    const v = Vec2.init(.{ 1, 0 });
    const rv = r.mulVec(v);
    try std.testing.expect(@abs(rv.v[0]) < eps);
    try std.testing.expect(@abs(rv.v[1] - 1.0) < eps);

    // 0° should be identity
    try std.testing.expect(Mat2.rotation(0).eql(Mat2.identity, 0));
}

test "Mat3: rotation (2D homogeneous)" {
    const pi = math.pi;
    const eps = @sqrt(math.floatEps(f32));

    // 90° CCW: (1,0,1) → (0,1,1)
    const r = Mat3.rotation(pi / 2.0);
    const v = Vec3.init(.{ 1, 0, 1 });
    const rv = r.mulVec(v);
    try std.testing.expect(@abs(rv.v[0]) < eps);
    try std.testing.expect(@abs(rv.v[1] - 1.0) < eps);
    try std.testing.expect(@abs(rv.v[2] - 1.0) < eps);
}

test "Mat4: rotation (axis-angle)" {
    const pi = math.pi;
    const eps = @sqrt(math.floatEps(f32));
    const e1 = Vec3.init(.{ 1, 0, 0 });
    const e2 = Vec3.init(.{ 0, 1, 0 });
    const e3 = Vec3.init(.{ 0, 0, 1 });

    // 90° around Z: (1,0,0) → (0,1,0)
    const rz = Mat4.rotation(e3, pi / 2.0);
    const v = Vec4.init(.{ 1, 0, 0, 1 });
    const rzv = rz.mulVec(v);
    try std.testing.expect(@abs(rzv.v[0]) < eps);
    try std.testing.expect(@abs(rzv.v[1] - 1.0) < eps);
    try std.testing.expect(@abs(rzv.v[2]) < eps);
    try std.testing.expect(@abs(rzv.v[3] - 1.0) < eps);

    // 90° around Y: (1,0,0) → (0,0,-1)
    const ry = Mat4.rotation(e2, pi / 2.0);
    const ryv = ry.mulVec(v);
    try std.testing.expect(@abs(ryv.v[0]) < eps);
    try std.testing.expect(@abs(ryv.v[2] - (-1.0)) < eps);

    // 90° around X: (0,1,0) → (0,0,1)
    const rx = Mat4.rotation(e1, pi / 2.0);
    const vy = Vec4.init(.{ 0, 1, 0, 1 });
    const rxv = rx.mulVec(vy);
    try std.testing.expect(@abs(rxv.v[1]) < eps);
    try std.testing.expect(@abs(rxv.v[2] - 1.0) < eps);

    // Identity rotation (0 angle)
    try std.testing.expect(Mat4.rotation(e3, 0).eql(Mat4.identity, 0));
}

test "Mat3: fromMat4" {
    // Extract upper-left 3x3 from a Mat4
    const m4 = Mat4.fromCols(.{
        Vec4.init(.{ 1, 2, 3, 0 }),
        Vec4.init(.{ 5, 6, 7, 0 }),
        Vec4.init(.{ 9, 10, 11, 0 }),
        Vec4.init(.{ 13, 14, 15, 1 }),
    });
    const m3 = Mat3.fromMat4(m4);
    try std.testing.expect(m3.col(0).eql(Vec3.init(.{ 1, 2, 3 }), 0));
    try std.testing.expect(m3.col(1).eql(Vec3.init(.{ 5, 6, 7 }), 0));
    try std.testing.expect(m3.col(2).eql(Vec3.init(.{ 9, 10, 11 }), 0));

    // Roundtrip: rotation Mat4 → Mat3 should match Rotor3.toMat3
    const pi = math.pi;
    const axis = Vec3.init(.{ 0, 1, 0 });
    const rot4 = Mat4.rotation(axis, pi / 3.0);
    const rot3 = Mat3.fromMat4(rot4);
    const rotor = Rot3.fromAxisAngle(pi / 3.0, axis);
    if (!rot3.eql(rotor.toMat3(), eps_normal)) return error.TestUnexpectedResult;
}

test "Rotor3: fromEuler basic" {
    const pi = math.pi;
    const e1 = Vec3.init(.{ 1, 0, 0 });
    const e2 = Vec3.init(.{ 0, 1, 0 });
    const e3 = Vec3.init(.{ 0, 0, 1 });

    // Pure yaw 90° around y: e1 → -e3
    const r_yaw = Rot3.fromEuler(pi / 2.0, 0, 0);
    if (!r_yaw.rotate(e1).eql(Vec3.init(.{ 0, 0, -1 }), eps_normal)) return error.TestUnexpectedResult;

    // Pure pitch 90° around x: e2 → e3
    const r_pitch = Rot3.fromEuler(0, pi / 2.0, 0);
    if (!r_pitch.rotate(e2).eql(e3, eps_normal)) return error.TestUnexpectedResult;

    // Pure roll 90° around z: e1 → e2
    const r_roll = Rot3.fromEuler(0, 0, pi / 2.0);
    if (!r_roll.rotate(e1).eql(e2, eps_normal)) return error.TestUnexpectedResult;
}

// ── Versor2 tests ──

test "Versor2: identity" {
    const v = Vec2.init(.{ 3, 4 });
    const result = Versor2.identity.apply(v);
    try std.testing.expect(result.eql(v, eps_normal));
}

test "Versor2: rotation matches Rot2" {
    const pi = math.pi;
    const angle: f32 = pi / 3.0;
    const v = Vec2.init(.{ 1, 0 });

    const rot = Rot2.fromAngle(angle);
    const versor = Versor2.fromAngle(angle);

    try std.testing.expect(rot.rotate(v).eql(versor.apply(v), eps_normal));
}

test "Versor2: reflection across Y axis" {
    // Normal = (1, 0) → reflects X coordinate
    const refl = Versor2.fromReflection(Vec2.init(.{ 1, 0 }));
    const v = Vec2.init(.{ 3, 4 });
    const result = refl.apply(v);
    try std.testing.expect(result.eql(Vec2.init(.{ -3, 4 }), eps_normal));
    try std.testing.expectEqual(Versor2.Parity.odd, refl.parity);
}

test "Versor2: reflection across X axis" {
    const refl = Versor2.fromReflection(Vec2.init(.{ 0, 1 }));
    const result = refl.apply(Vec2.init(.{ 3, 4 }));
    try std.testing.expect(result.eql(Vec2.init(.{ 3, -4 }), eps_normal));
}

test "Versor2: two reflections compose to rotation" {
    // Reflecting across X then Y should give 180° rotation
    const refl_x = Versor2.fromReflection(Vec2.init(.{ 1, 0 }));
    const refl_y = Versor2.fromReflection(Vec2.init(.{ 0, 1 }));
    const composed = refl_y.mul(refl_x); // apply refl_x first
    try std.testing.expectEqual(Versor2.Parity.even, composed.parity);

    const v = Vec2.init(.{ 3, 4 });
    const result = composed.apply(v);
    try std.testing.expect(result.eql(Vec2.init(.{ -3, -4 }), eps_normal));
}

test "Versor2: mul applies b first then a" {
    const refl = Versor2.fromReflection(Vec2.init(.{ 1, 0 }));
    const rot = Versor2.fromAngle(math.pi / 2.0);
    const composed = rot.mul(refl); // reflect first, then rotate

    const v = Vec2.init(.{ 1, 0 });
    const step1 = refl.apply(v); // (-1, 0)
    const step2 = rot.apply(step1); // (0, -1)
    const result = composed.apply(v);
    try std.testing.expect(result.eql(step2, eps_normal));
}

test "Versor2: roundtrip toRotor/fromRotor" {
    const angle: f32 = 1.23;
    const rot = Rot2.fromAngle(angle);
    const versor = Versor2.fromRotor(rot);
    const back = versor.toRotor().?;
    try std.testing.expect(@abs(rot.a - back.a) <= eps_normal);
    try std.testing.expect(@abs(rot.b - back.b) <= eps_normal);
}

test "Versor2: toMat2 rotation" {
    const angle: f32 = math.pi / 4.0;
    const versor = Versor2.fromAngle(angle);
    const rot = Rot2.fromAngle(angle);
    try std.testing.expect(versor.toMat2().eql(rot.toMat2(), eps_normal));
}

test "Versor2: toMat2 reflection" {
    const refl = Versor2.fromReflection(Vec2.init(.{ 1, 0 }));
    const m = refl.toMat2();
    // Should be diag(-1, 1)
    try std.testing.expect(@abs(m.m[0][0] - @as(f32, -1)) <= eps_normal);
    try std.testing.expect(@abs(m.m[1][1] - @as(f32, 1)) <= eps_normal);
}

test "Versor2: slerp endpoints" {
    const r0 = Versor2.fromAngle(0);
    const r1 = Versor2.fromAngle(math.pi / 2.0);
    const s0 = Versor2.slerp(r0, r1, 0);
    const s1 = Versor2.slerp(r0, r1, 1);
    const v = Vec2.init(.{ 1, 0 });
    try std.testing.expect(s0.apply(v).eql(r0.apply(v), eps_normal));
    try std.testing.expect(s1.apply(v).eql(r1.apply(v), eps_normal));
}

test "Versor2: slerp between reflections" {
    const r0 = Versor2.fromReflection(Vec2.init(.{ 1, 0 }));
    const r1 = Versor2.fromReflection(Vec2.init(.{ 0, 1 }));
    const mid = Versor2.slerp(r0, r1, 0.5);
    try std.testing.expectEqual(Versor2.Parity.odd, mid.parity);
    // Midpoint should be unit length
    try std.testing.expect(@abs(mid.lenSqr() - @as(f32, 1.0)) <= eps_normal);
}

// ── Versor3 tests ──

test "Versor3: identity" {
    const v = Vec3.init(.{ 3, 4, 5 });
    const result = Versor3.identity.apply(v);
    try std.testing.expect(result.eql(v, eps_normal));
}

test "Versor3: rotation matches Rot3" {
    const pi = math.pi;
    const axis = Vec3.init(.{ 0, 1, 0 });
    const angle: f32 = pi / 3.0;
    const v = Vec3.init(.{ 1, 0, 0 });

    const rot = Rot3.fromAxisAngle(angle, axis);
    const versor = Versor3.fromRotor(rot);

    try std.testing.expect(rot.rotate(v).eql(versor.apply(v), eps_normal));
}

test "Versor3: fromAxisAngle matches Rot3 rotation" {
    const pi = math.pi;
    const axis = Vec3.init(.{ 0, 1, 0 });
    const v = Vec3.init(.{ 1, 0, 0 });

    // 90° around Y: (1,0,0) → (0,0,-1)
    const versor = Versor3.fromAxisAngle(pi / 2.0, axis);
    const result = versor.apply(v);
    try std.testing.expect(result.eql(Vec3.init(.{ 0, 0, -1 }), eps_normal));

    // Also verify via Rot3
    const rot = Rot3.fromAxisAngle(pi / 2.0, axis);
    try std.testing.expect(result.eql(rot.rotate(v), eps_normal));
}

test "Versor3: reflection across YZ plane" {
    const refl = Versor3.fromReflection(Vec3.init(.{ 1, 0, 0 }));
    const v = Vec3.init(.{ 3, 4, 5 });
    const result = refl.apply(v);
    try std.testing.expect(result.eql(Vec3.init(.{ -3, 4, 5 }), eps_normal));
    try std.testing.expectEqual(Versor3.Parity.odd, refl.parity);
}

test "Versor3: reflection across XZ plane" {
    const refl = Versor3.fromReflection(Vec3.init(.{ 0, 1, 0 }));
    const result = refl.apply(Vec3.init(.{ 3, 4, 5 }));
    try std.testing.expect(result.eql(Vec3.init(.{ 3, -4, 5 }), eps_normal));
}

test "Versor3: reflection across XY plane" {
    const refl = Versor3.fromReflection(Vec3.init(.{ 0, 0, 1 }));
    const result = refl.apply(Vec3.init(.{ 3, 4, 5 }));
    try std.testing.expect(result.eql(Vec3.init(.{ 3, 4, -5 }), eps_normal));
}

test "Versor3: two reflections compose to rotation" {
    const refl_x = Versor3.fromReflection(Vec3.init(.{ 1, 0, 0 }));
    const refl_y = Versor3.fromReflection(Vec3.init(.{ 0, 1, 0 }));
    const composed = refl_y.mul(refl_x); // reflect X first, then Y
    try std.testing.expectEqual(Versor3.Parity.even, composed.parity);

    // Two perpendicular reflections = 180° rotation around Z
    const v = Vec3.init(.{ 1, 0, 0 });
    const result = composed.apply(v);
    try std.testing.expect(result.eql(Vec3.init(.{ -1, 0, 0 }), eps_normal));
}

test "Versor3: mul applies b first then a" {
    const refl = Versor3.fromReflection(Vec3.init(.{ 1, 0, 0 }));
    const rot = Versor3.fromAxisAngle(math.pi / 2.0, Vec3.init(.{ 0, 1, 0 }));
    const composed = rot.mul(refl); // reflect first, then rotate

    const v = Vec3.init(.{ 1, 0, 0 });
    const step1 = refl.apply(v);
    const step2 = rot.apply(step1);
    const result = composed.apply(v);
    try std.testing.expect(result.eql(step2, eps_normal));
}

test "Versor3: three-way composition" {
    const r1 = Versor3.fromReflection(Vec3.init(.{ 1, 0, 0 }));
    const r2 = Versor3.fromAxisAngle(math.pi / 4.0, Vec3.init(.{ 0, 0, 1 }));
    const r3 = Versor3.fromReflection(Vec3.init(.{ 0, 1, 0 }));

    // r3.mul(r2).mul(r1) = apply r1, then r2, then r3
    const composed = r3.mul(r2).mul(r1);

    const v = Vec3.init(.{ 1, 2, 3 });
    const step1 = r1.apply(v);
    const step2 = r2.apply(step1);
    const step3 = r3.apply(step2);
    const result = composed.apply(v);
    try std.testing.expect(result.eql(step3, eps_loose));
}

test "Versor3: roundtrip toRotor/fromRotor" {
    const axis = Vec3.init(.{ 0.5774, 0.5774, 0.5774 });
    const angle: f32 = 1.23;
    const rot = Rot3.fromAxisAngle(angle, axis.normalize());
    const versor = Versor3.fromRotor(rot);
    const back = versor.toRotor().?;
    try std.testing.expect(@abs(rot.a - back.a) <= eps_normal);
    try std.testing.expect(@abs(rot.b01 - back.b01) <= eps_normal);
    try std.testing.expect(@abs(rot.b02 - back.b02) <= eps_normal);
    try std.testing.expect(@abs(rot.b12 - back.b12) <= eps_normal);
}

test "Versor3: toRotor returns null for odd" {
    const refl = Versor3.fromReflection(Vec3.init(.{ 1, 0, 0 }));
    try std.testing.expectEqual(@as(?Rot3, null), refl.toRotor());
}

test "Versor3: toMat3 rotation matches Rot3" {
    const axis = Vec3.init(.{ 0, 1, 0 });
    const angle: f32 = math.pi / 3.0;
    const versor = Versor3.fromAxisAngle(angle, axis);
    const rot = Rot3.fromAxisAngle(angle, axis);
    try std.testing.expect(versor.toMat3().eql(rot.toMat3(), eps_normal));
}

test "Versor3: toMat3 reflection has det = -1" {
    const refl = Versor3.fromReflection(Vec3.init(.{ 1, 0, 0 }));
    const m = refl.toMat3();
    // det of reflection across YZ = -1
    // m should be diag(-1, 1, 1)
    try std.testing.expect(@abs(m.m[0][0] - @as(f32, -1)) <= eps_normal);
    try std.testing.expect(@abs(m.m[1][1] - @as(f32, 1)) <= eps_normal);
    try std.testing.expect(@abs(m.m[2][2] - @as(f32, 1)) <= eps_normal);
}

test "Versor3: reflection preserves length" {
    const refl = Versor3.fromReflection(Vec3.init(.{ 0.5774, 0.5774, 0.5774 }).normalize());
    const v = Vec3.init(.{ 3, 4, 5 });
    const result = refl.apply(v);
    try std.testing.expect(@abs(v.len() - result.len()) <= eps_normal);
}

test "Versor3: reverse inverts transformation" {
    const versor = Versor3.fromAxisAngle(1.23, Vec3.init(.{ 0, 1, 0 }));
    const v = Vec3.init(.{ 1, 2, 3 });
    const forward = versor.apply(v);
    const back = versor.reverse().apply(forward);
    try std.testing.expect(back.eql(v, eps_normal));
}

test "Versor3: reverse inverts reflection" {
    const refl = Versor3.fromReflection(Vec3.init(.{ 0.5774, 0.5774, 0.5774 }).normalize());
    const v = Vec3.init(.{ 1, 2, 3 });
    const forward = refl.apply(v);
    const back = refl.reverse().apply(forward);
    try std.testing.expect(back.eql(v, eps_normal));
}

test "Versor3: slerp endpoints" {
    const r0 = Versor3.fromAxisAngle(0, Vec3.init(.{ 0, 1, 0 }));
    const r1 = Versor3.fromAxisAngle(math.pi / 2.0, Vec3.init(.{ 0, 1, 0 }));
    const s1 = Versor3.slerp(r0, r1, 1);
    // At t=1, should match r1
    const v = Vec3.init(.{ 1, 0, 0 });
    try std.testing.expect(s1.apply(v).eql(r1.apply(v), eps_normal));
}

test "Versor3: slerp between reflections" {
    const r0 = Versor3.fromReflection(Vec3.init(.{ 1, 0, 0 }));
    const r1 = Versor3.fromReflection(Vec3.init(.{ 0, 1, 0 }));
    const mid = Versor3.slerp(r0, r1, 0.5);
    try std.testing.expectEqual(Versor3.Parity.odd, mid.parity);
    try std.testing.expect(@abs(mid.lenSqr() - @as(f32, 1.0)) <= eps_normal);
}

test "Versor3: det" {
    const rot = Versor3.fromAxisAngle(1.0, Vec3.init(.{ 0, 1, 0 }));
    try std.testing.expectEqual(@as(f32, 1), rot.det());
    const refl = Versor3.fromReflection(Vec3.init(.{ 1, 0, 0 }));
    try std.testing.expectEqual(@as(f32, -1), refl.det());
}

// ── Mirror2 tests ──

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

// ── Mirror3 tests ──

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

// ── Transform tests ──

test "Transform: identity produces identity matrix" {
    const t = Transform.identity;
    const m = t.toMat4();
    inline for (0..4) |col| {
        inline for (0..4) |row| {
            const expected: f32 = if (col == row) 1.0 else 0.0;
            try std.testing.expectApproxEqAbs(expected, m.m[col][row], 1e-6);
        }
    }
}

test "Transform: translation-only" {
    const t = Transform.fromTranslation(Vec3.init(.{ 3, 4, 5 }));
    const m = t.toMat4();
    const expected = Mat4.translation(Vec3.init(.{ 3, 4, 5 }));
    try std.testing.expect(m.eql(expected, eps_normal));
}

test "Transform: rotation-only" {
    const axis = Vec3.init(.{ 0, 1, 0 });
    const angle: f32 = math.pi / 4.0;
    const rot = Rot3.fromAxisAngle(angle, axis);
    const t = Transform.fromRotation(rot);
    const m = t.toMat4();
    const expected = rot.toMat4();
    try std.testing.expect(m.eql(expected, eps_normal));
}

test "Transform: TRS applies scale then rotation then translation" {
    const t = Transform{
        .translation = Vec3.init(.{ 10, 0, 0 }),
        .rotation = Rot3.identity,
        .scaling = Vec3.init(.{ 2, 2, 2 }),
    };
    const m = t.toMat4();
    const point = Vec3.init(.{ 1, 0, 0 });
    const result = m.mulVec(Vec4.init(.{ point.v[0], point.v[1], point.v[2], 1.0 }));
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), result.v[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.v[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.v[2], 1e-6);
}
