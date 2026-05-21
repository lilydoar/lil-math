//! Discrete direction enums for grid-based spatial reasoning.
//!
//! Four types form a family, named by direction count:
//!
//!   - `Dir4`  — 4 cardinal directions on a 2D grid (N/E/S/W)
//!   - `Dir8`  — 8 compass directions on a 2D grid (cardinals + diagonals)
//!   - `Dir6`  — 6 face directions on a 3D grid (N/E/S/W/Up/Down)
//!   - `Dir26` — 26 neighbor directions on a 3D grid (faces + edges + vertices)
//!
//! ## Axis conventions
//!
//!   **2D (Dir4, Dir8):** +X = east, +Y = north.
//!   `forward()` returns `Vec2i`.
//!
//!   **3D (Dir6, Dir26):** +X = east, +Y = up, +Z = north.
//!   Right-handed, Y-up — consistent with `linalg.zig`.
//!   `forward()` returns `Vec3i`.
//!
//! ## Integer layout
//!
//! Backing values are chosen so that conversions between types are
//! cheap casts or bit ops:
//!
//!   - Dir4 values 0–3 map directly to Dir6 values 0–3 (horizontal subset).
//!   - Dir8 even values are cardinals: `Dir4 → Dir8` is `val << 1`.
//!   - Dir26 values 0–5 match Dir6 (face subset).

const std = @import("std");
const linalg = @import("linalg.zig");
const Vec2i = linalg.VecType(2, i32);
const Vec3i = linalg.VecType(3, i32);

// ---------------------------------------------------------------------------
// Dir4
// ---------------------------------------------------------------------------

/// 2 directions on a 1D axis: positive (+X) and negative (−X).
///
/// Backed by `u1`.
pub const Dir2 = enum(u1) {
    pos = 0,
    neg = 1,

    /// Both directions in enum order.
    pub const all: [2]Dir2 = .{ .pos, .neg };

    /// Unit step in this direction as a signed integer (+1 or −1).
    pub fn forward(self: Dir2) i32 {
        return switch (self) {
            .pos => 1,
            .neg => -1,
        };
    }

    /// The opposite direction.
    pub fn opposite(self: Dir2) Dir2 {
        return @enumFromInt(@as(u1, @intFromEnum(self)) +% 1);
    }
};

/// 4 cardinal directions on a 2D grid.
///
/// Axis convention: +X = east, +Y = north.
/// Backed by `u2`; values 0–3 match the horizontal subset of `Dir6`.
pub const Dir4 = enum(u2) {
    north = 0,
    east = 1,
    south = 2,
    west = 3,

    /// All four directions in enum order.
    pub const all: [4]Dir4 = .{ .north, .east, .south, .west };

    /// Unit step in this direction as a 2D integer vector.
    pub fn forward(self: Dir4) Vec2i {
        return switch (self) {
            .north => Vec2i.init(.{ 0, 1 }),
            .east => Vec2i.init(.{ 1, 0 }),
            .south => Vec2i.init(.{ 0, -1 }),
            .west => Vec2i.init(.{ -1, 0 }),
        };
    }

    /// The direction facing the opposite way (180° rotation).
    pub fn opposite(self: Dir4) Dir4 {
        return @enumFromInt(@as(u2, @intFromEnum(self)) +% 2);
    }

    /// Rotate 90° clockwise (north → east → south → west).
    pub fn cw(self: Dir4) Dir4 {
        return @enumFromInt(@as(u2, @intFromEnum(self)) +% 1);
    }

    /// Rotate 90° counter-clockwise (north → west → south → east).
    pub fn ccw(self: Dir4) Dir4 {
        return @enumFromInt(@as(u2, @intFromEnum(self)) -% 1);
    }

    /// Relative facing between two Dir4 directions.
    ///
    /// The rotation that takes `self` to `other`:
    ///   - `same`     — 0°, facing the same way
    ///   - `cw`       — 90° clockwise (other is to your right)
    ///   - `opposite` — 180°, facing away from each other
    ///   - `ccw`      — 90° counter-clockwise (other is to your left)
    pub const Rel = enum(u2) {
        same = 0,
        cw = 1,
        opposite = 2,
        ccw = 3,

        /// Reverse the relative direction (cw ↔ ccw, same/opposite unchanged).
        pub fn invert(self: Rel) Rel {
            return @enumFromInt(@as(u2, 0) -% @intFromEnum(self));
        }
    };

    /// Compute the relative facing from `self` to `other`.
    pub fn rel(self: Dir4, other: Dir4) Rel {
        return @enumFromInt(@as(u2, @intFromEnum(other)) -% @intFromEnum(self));
    }

    /// Embed into `Dir6` as a horizontal direction (Y=0 plane in 3D).
    pub fn toDir6(self: Dir4) Dir6 {
        return @enumFromInt(@as(u3, @intFromEnum(self)));
    }

    /// Embed into `Dir8` as a cardinal direction.
    pub fn toDir8(self: Dir4) Dir8 {
        return @enumFromInt(@as(u3, @intFromEnum(self)) << 1);
    }
};

// ---------------------------------------------------------------------------
// Dir8
// ---------------------------------------------------------------------------

/// 8 compass directions on a 2D grid (cardinals + diagonals).
///
/// Axis convention: +X = east, +Y = north.
/// Backed by `u3`; even values (0,2,4,6) are the four cardinals, matching
/// `Dir4` via `val << 1` / `val >> 1`.
pub const Dir8 = enum(u3) {
    north = 0,
    north_east = 1,
    east = 2,
    south_east = 3,
    south = 4,
    south_west = 5,
    west = 6,
    north_west = 7,

    /// All eight directions in enum order.
    pub const all: [8]Dir8 = .{
        .north, .north_east, .east, .south_east,
        .south, .south_west, .west, .north_west,
    };

    /// Unit step in this direction as a 2D integer vector.
    /// Diagonal steps have magnitude (1,1), not normalized.
    pub fn forward(self: Dir8) Vec2i {
        return switch (self) {
            .north => Vec2i.init(.{ 0, 1 }),
            .north_east => Vec2i.init(.{ 1, 1 }),
            .east => Vec2i.init(.{ 1, 0 }),
            .south_east => Vec2i.init(.{ 1, -1 }),
            .south => Vec2i.init(.{ 0, -1 }),
            .south_west => Vec2i.init(.{ -1, -1 }),
            .west => Vec2i.init(.{ -1, 0 }),
            .north_west => Vec2i.init(.{ -1, 1 }),
        };
    }

    /// The direction facing the opposite way (180° rotation).
    pub fn opposite(self: Dir8) Dir8 {
        return @enumFromInt(@as(u3, @intFromEnum(self)) +% 4);
    }

    /// Rotate 45° clockwise.
    pub fn cw(self: Dir8) Dir8 {
        return @enumFromInt(@as(u3, @intFromEnum(self)) +% 1);
    }

    /// Rotate 45° counter-clockwise.
    pub fn ccw(self: Dir8) Dir8 {
        return @enumFromInt(@as(u3, @intFromEnum(self)) -% 1);
    }

    /// Relative facing between two Dir8 directions.
    ///
    /// The rotation that takes `self` to `other`, in 45° increments:
    ///   - `same`      — 0°
    ///   - `cw_45`     — 45° clockwise
    ///   - `cw_90`     — 90° clockwise
    ///   - `cw_135`    — 135° clockwise
    ///   - `opposite`  — 180°
    ///   - `ccw_135`   — 135° counter-clockwise
    ///   - `ccw_90`    — 90° counter-clockwise
    ///   - `ccw_45`    — 45° counter-clockwise
    pub const Rel = enum(u3) {
        same = 0,
        cw_45 = 1,
        cw_90 = 2,
        cw_135 = 3,
        opposite = 4,
        ccw_135 = 5,
        ccw_90 = 6,
        ccw_45 = 7,

        /// Reverse the relative direction.
        pub fn invert(self: Rel) Rel {
            return @enumFromInt(@as(u3, 0) -% @intFromEnum(self));
        }

        /// True if the relative turn is strictly clockwise (excludes same/opposite).
        pub fn isCw(self: Rel) bool {
            const v = @intFromEnum(self);
            return v >= 1 and v <= 3;
        }

        /// True if the relative turn is strictly counter-clockwise (excludes same/opposite).
        pub fn isCcw(self: Rel) bool {
            const v = @intFromEnum(self);
            return v >= 5 and v <= 7;
        }
    };

    /// Compute the relative facing from `self` to `other`.
    pub fn rel(self: Dir8, other: Dir8) Rel {
        return @enumFromInt(@as(u3, @intFromEnum(other)) -% @intFromEnum(self));
    }

    /// True for axis-aligned directions (N, E, S, W).
    pub fn isCardinal(self: Dir8) bool {
        return @intFromEnum(self) & 1 == 0;
    }

    /// True for diagonal directions (NE, SE, SW, NW).
    pub fn isDiagonal(self: Dir8) bool {
        return @intFromEnum(self) & 1 == 1;
    }

    /// Convert to `Dir4` if this is a cardinal direction, otherwise `null`.
    pub fn toDir4(self: Dir8) ?Dir4 {
        const v = @intFromEnum(self);
        if (v & 1 != 0) return null;
        return @enumFromInt(@as(u2, @intCast(v >> 1)));
    }
};

// ---------------------------------------------------------------------------
// Dir6
// ---------------------------------------------------------------------------

/// 6 face directions on a 3D grid (axis-aligned neighbors).
///
/// Axis convention: +X = east, +Y = up, +Z = north (right-handed, Y-up).
/// Backed by `u3`; values 0–3 match `Dir4` (horizontal subset).
pub const Dir6 = enum(u3) {
    north = 0,
    east = 1,
    south = 2,
    west = 3,
    up = 4,
    down = 5,

    /// Spatial axis (X, Y, or Z).
    pub const Axis = enum { x, y, z };

    /// Direction along an axis (positive or negative).
    pub const Sign = enum { positive, negative };

    /// All six directions in enum order.
    pub const all: [6]Dir6 = .{ .north, .east, .south, .west, .up, .down };

    /// Unit step in this direction as a 3D integer vector.
    pub fn forward(self: Dir6) Vec3i {
        return switch (self) {
            .north => Vec3i.init(.{ 0, 0, 1 }),
            .east => Vec3i.init(.{ 1, 0, 0 }),
            .south => Vec3i.init(.{ 0, 0, -1 }),
            .west => Vec3i.init(.{ -1, 0, 0 }),
            .up => Vec3i.init(.{ 0, 1, 0 }),
            .down => Vec3i.init(.{ 0, -1, 0 }),
        };
    }

    /// The direction facing the opposite way.
    pub fn opposite(self: Dir6) Dir6 {
        return switch (self) {
            .north => .south,
            .east => .west,
            .south => .north,
            .west => .east,
            .up => .down,
            .down => .up,
        };
    }

    /// Which axis this direction lies along.
    pub fn axis(self: Dir6) Axis {
        return switch (self) {
            .east, .west => .x,
            .up, .down => .y,
            .north, .south => .z,
        };
    }

    /// Whether this direction points in the positive or negative axis direction.
    pub fn sign(self: Dir6) Sign {
        return switch (self) {
            .north, .east, .up => .positive,
            .south, .west, .down => .negative,
        };
    }

    /// True for horizontal directions (N/E/S/W).
    pub fn isHorizontal(self: Dir6) bool {
        return @intFromEnum(self) < 4;
    }

    /// True for vertical directions (Up/Down).
    pub fn isVertical(self: Dir6) bool {
        return @intFromEnum(self) >= 4;
    }

    /// Convert to `Dir4` if this is a horizontal direction, otherwise `null`.
    pub fn toDir4(self: Dir6) ?Dir4 {
        const v = @intFromEnum(self);
        if (v >= 4) return null;
        return @enumFromInt(@as(u2, @intCast(v)));
    }

    /// Embed into `Dir26` as a face direction.
    pub fn toDir26(self: Dir6) Dir26 {
        return @enumFromInt(@as(u5, @intFromEnum(self)));
    }
};

// ---------------------------------------------------------------------------
// Dir26
// ---------------------------------------------------------------------------

/// 26 neighbor directions on a 3D grid (faces + edges + vertices).
///
/// Axis convention: +X = east, +Y = up, +Z = north (right-handed, Y-up).
/// Backed by `u5`. Values 0–5 match `Dir6` (face directions).
/// Values 6–17 are edge directions (two non-zero components).
/// Values 18–25 are vertex directions (three non-zero components).
pub const Dir26 = enum(u5) {
    // Faces (6) — match Dir6 layout
    north = 0,
    east = 1,
    south = 2,
    west = 3,
    up = 4,
    down = 5,

    // Edges (12) — two non-zero components
    north_east = 6,
    south_east = 7,
    south_west = 8,
    north_west = 9,
    up_north = 10,
    up_east = 11,
    up_south = 12,
    up_west = 13,
    down_north = 14,
    down_east = 15,
    down_south = 16,
    down_west = 17,

    // Vertices (8) — three non-zero components
    up_north_east = 18,
    up_south_east = 19,
    up_south_west = 20,
    up_north_west = 21,
    down_north_east = 22,
    down_south_east = 23,
    down_south_west = 24,
    down_north_west = 25,

    /// Neighbor class: face, edge, or vertex connectivity.
    pub const Kind = enum { face, edge, vertex };

    /// All 26 directions in enum order.
    pub const all: [26]Dir26 = blk: {
        var dirs: [26]Dir26 = undefined;
        for (0..26) |i| dirs[i] = @enumFromInt(i);
        break :blk dirs;
    };

    /// Unit step in this direction as a 3D integer vector.
    /// Edge and vertex steps have magnitude > 1 (not normalized).
    pub fn forward(self: Dir26) Vec3i {
        // (x, y, z) where +X=east, +Y=up, +Z=north
        return switch (self) {
            // Faces
            .north => Vec3i.init(.{ 0, 0, 1 }),
            .east => Vec3i.init(.{ 1, 0, 0 }),
            .south => Vec3i.init(.{ 0, 0, -1 }),
            .west => Vec3i.init(.{ -1, 0, 0 }),
            .up => Vec3i.init(.{ 0, 1, 0 }),
            .down => Vec3i.init(.{ 0, -1, 0 }),

            // Edges
            .north_east => Vec3i.init(.{ 1, 0, 1 }),
            .south_east => Vec3i.init(.{ 1, 0, -1 }),
            .south_west => Vec3i.init(.{ -1, 0, -1 }),
            .north_west => Vec3i.init(.{ -1, 0, 1 }),
            .up_north => Vec3i.init(.{ 0, 1, 1 }),
            .up_east => Vec3i.init(.{ 1, 1, 0 }),
            .up_south => Vec3i.init(.{ 0, 1, -1 }),
            .up_west => Vec3i.init(.{ -1, 1, 0 }),
            .down_north => Vec3i.init(.{ 0, -1, 1 }),
            .down_east => Vec3i.init(.{ 1, -1, 0 }),
            .down_south => Vec3i.init(.{ 0, -1, -1 }),
            .down_west => Vec3i.init(.{ -1, -1, 0 }),

            // Vertices
            .up_north_east => Vec3i.init(.{ 1, 1, 1 }),
            .up_south_east => Vec3i.init(.{ 1, 1, -1 }),
            .up_south_west => Vec3i.init(.{ -1, 1, -1 }),
            .up_north_west => Vec3i.init(.{ -1, 1, 1 }),
            .down_north_east => Vec3i.init(.{ 1, -1, 1 }),
            .down_south_east => Vec3i.init(.{ 1, -1, -1 }),
            .down_south_west => Vec3i.init(.{ -1, -1, -1 }),
            .down_north_west => Vec3i.init(.{ -1, -1, 1 }),
        };
    }

    /// The direction facing the opposite way.
    pub fn opposite(self: Dir26) Dir26 {
        const f = self.forward();
        return fromOffset(Vec3i.init(.{ 0, 0, 0 }).sub(f)).?;
    }

    /// Whether this is a face (6), edge (12), or vertex (8) direction.
    pub fn kind(self: Dir26) Kind {
        const v = @intFromEnum(self);
        if (v < 6) return .face;
        if (v < 18) return .edge;
        return .vertex;
    }

    /// Convert to `Dir6` if this is a face direction, otherwise `null`.
    pub fn toDir6(self: Dir26) ?Dir6 {
        const v = @intFromEnum(self);
        if (v >= 6) return null;
        return @enumFromInt(@as(u3, @intCast(v)));
    }

    /// Look up a Dir26 from a 3D offset.
    ///
    /// Returns `null` if `offset` is the zero vector or any component
    /// falls outside the range -1..1.
    pub fn fromOffset(offset: Vec3i) ?Dir26 {
        const x = offset.v[0];
        const y = offset.v[1];
        const z = offset.v[2];
        if (x == 0 and y == 0 and z == 0) return null;
        if (x < -1 or x > 1 or y < -1 or y > 1 or z < -1 or z > 1) return null;

        // Encode as index into 3x3x3 cube minus center.
        // Map -1,0,1 → 0,1,2 for each axis, then look up.
        const ux: usize = @intCast(x + 1);
        const uy: usize = @intCast(y + 1);
        const uz: usize = @intCast(z + 1);
        return offset_lut[ux + uy * 3 + uz * 9];
    }

    const offset_lut: [27]?Dir26 = blk: {
        var lut: [27]?Dir26 = .{null} ** 27;
        for (all) |d| {
            const f = d.forward();
            const ux: usize = @intCast(f.v[0] + 1);
            const uy: usize = @intCast(f.v[1] + 1);
            const uz: usize = @intCast(f.v[2] + 1);
            lut[ux + uy * 3 + uz * 9] = d;
        }
        break :blk lut;
    };
};

// ---------------------------------------------------------------------------
// DirSetType
// ---------------------------------------------------------------------------

/// Generic set type for direction enums, backed by a bitmask with one bit
/// per variant. Set operations (union, intersect, complement) are single
/// bitwise ops. Iteration yields directions in enum order via `@ctz`.
///
/// Requires a contiguous enum with values `0..N-1`.
pub fn DirSetType(comptime Dir: type) type {
    const fields = @typeInfo(Dir).@"enum".fields;
    const n = fields.len;
    const Bits = std.meta.Int(.unsigned, n);
    const ShiftAmt = std.math.Log2Int(Bits);

    comptime {
        for (fields, 0..) |field, i| {
            if (field.value != i)
                @compileError("DirSetType requires contiguous enum values starting from 0");
        }
    }

    return struct {
        const Self = @This();

        bits: Bits = 0,

        /// The empty set.
        pub const empty = Self{};

        /// The complete set containing all directions.
        pub const full = Self{ .bits = std.math.maxInt(Bits) };

        /// A set containing a single direction.
        pub fn of(dir: Dir) Self {
            return .{ .bits = @as(Bits, 1) << @as(ShiftAmt, @intCast(@intFromEnum(dir))) };
        }

        /// True if the set contains `dir`.
        pub fn contains(self: Self, dir: Dir) bool {
            return self.bits & of(dir).bits != 0;
        }

        /// Add `dir` to the set.
        pub fn insert(self: Self, dir: Dir) Self {
            return .{ .bits = self.bits | of(dir).bits };
        }

        /// Remove `dir` from the set.
        pub fn remove(self: Self, dir: Dir) Self {
            return .{ .bits = self.bits & ~of(dir).bits };
        }

        /// Toggle the presence of `dir` (add if absent, remove if present).
        pub fn toggle(self: Self, dir: Dir) Self {
            return .{ .bits = self.bits ^ of(dir).bits };
        }

        /// Union of two sets (all elements in either set).
        pub fn unionWith(self: Self, other: Self) Self {
            return .{ .bits = self.bits | other.bits };
        }

        /// Intersection of two sets (only elements in both).
        pub fn intersect(self: Self, other: Self) Self {
            return .{ .bits = self.bits & other.bits };
        }

        /// Difference of two sets (elements in `self` but not in `other`).
        pub fn diff(self: Self, other: Self) Self {
            return .{ .bits = self.bits & ~other.bits };
        }

        /// Symmetric difference (elements in exactly one set).
        pub fn symmetricDiff(self: Self, other: Self) Self {
            return .{ .bits = self.bits ^ other.bits };
        }

        /// Complement (all directions not in this set).
        pub fn complement(self: Self) Self {
            return .{ .bits = ~self.bits };
        }

        /// True if `self` is a subset of `other`.
        pub fn subsetOf(self: Self, other: Self) bool {
            return self.bits & ~other.bits == 0;
        }

        /// True if `self` is a superset of `other`.
        pub fn supersetOf(self: Self, other: Self) bool {
            return other.subsetOf(self);
        }

        /// Number of directions in the set.
        pub fn count(self: Self) usize {
            return @popCount(self.bits);
        }

        /// True if the set contains no directions.
        pub fn isEmpty(self: Self) bool {
            return self.bits == 0;
        }

        /// True if two sets contain the same directions.
        pub fn eql(self: Self, other: Self) bool {
            return self.bits == other.bits;
        }

        /// Iterate over set directions in enum order.
        pub fn iterator(self: Self) Iterator {
            return .{ .bits = self.bits };
        }

        pub const Iterator = struct {
            bits: Bits,

            pub fn next(it: *Iterator) ?Dir {
                if (it.bits == 0) return null;
                const idx: ShiftAmt = @intCast(@ctz(it.bits));
                it.bits &= it.bits - 1;
                return @enumFromInt(idx);
            }
        };
    };
}

// ---------------------------------------------------------------------------
// Ori24
// ---------------------------------------------------------------------------

/// The 24 proper rotations of a cube (chiral octahedral symmetry group).
///
/// Encodes the full 3D orientation of a grid-aligned object. Each element
/// maps the local coordinate frame (right +X, up +Y, forward +Z) to a
/// world-space frame.
///
/// ## Axis convention
///
/// Same as `Dir6`: +X = east, +Y = up, +Z = north (right-handed, Y-up).
///
/// ## Integer layout
///
/// Backed by `u5` (0–23), laid out as `facing_index * 4 + spin`:
///   - `facing_index` (0–5): `Dir6` ordinal of the forward (+Z) direction.
///   - `spin` (0–3): CW rotation step around the facing axis.
///
/// Spin 0 ("canonical up") per facing:
///   - Horizontal (N/E/S/W): local +Y → world +Y (upright).
///   - Facing up (+Y): local +Y → world -Z (tilted back from identity).
///   - Facing down (-Y): local +Y → world +Z (tilted forward from identity).
///
/// All operations use comptime-generated lookup tables — every method is O(1).
pub const Ori24 = struct {
    _val: u5,

    /// The identity orientation (no rotation).
    pub const identity: Ori24 = .{ ._val = 0 };

    /// All 24 orientations in enum order.
    pub const all: [24]Ori24 = blk: {
        var a: [24]Ori24 = undefined;
        for (0..24) |i| a[i] = .{ ._val = @intCast(i) };
        break :blk a;
    };

    /// Which direction the local forward axis (+Z) points in world space.
    pub fn facing(self: Ori24) Dir6 {
        return @enumFromInt(facing_lut[self._val]);
    }

    /// Which direction the local up axis (+Y) points in world space.
    pub fn up(self: Ori24) Dir6 {
        return @enumFromInt(up_lut[self._val]);
    }

    /// Which direction the local right axis (+X) points in world space.
    pub fn right(self: Ori24) Dir6 {
        return @enumFromInt(right_lut[self._val]);
    }

    /// The spin component: CW rotation (0–3) around the facing axis.
    pub fn spin(self: Ori24) u2 {
        return @intCast(self._val % 4);
    }

    /// Compose two orientations (matrix multiplication order).
    ///
    /// `a.mul(b).apply(v) == a.apply(b.apply(v))`.
    pub fn mul(self: Ori24, other: Ori24) Ori24 {
        return .{ ._val = mul_lut[self._val][other._val] };
    }

    /// The inverse orientation.
    ///
    /// `self.mul(self.inverse()) == identity`.
    pub fn inverse(self: Ori24) Ori24 {
        return .{ ._val = inv_lut[self._val] };
    }

    /// Transform a direction by this orientation.
    ///
    /// Maps a local-frame direction to the corresponding world-frame direction.
    pub fn apply(self: Ori24, dir: Dir6) Dir6 {
        return @enumFromInt(apply_lut[self._val][@intFromEnum(dir)]);
    }

    /// Construct from facing direction with canonical up (spin 0).
    pub fn fromFacing(forward: Dir6) Ori24 {
        return .{ ._val = @as(u5, @intFromEnum(forward)) * 4 };
    }

    /// Construct from facing and up directions.
    ///
    /// Returns `null` if `forward` and `up_dir` are not perpendicular.
    pub fn fromAxes(forward: Dir6, up_dir: Dir6) ?Ori24 {
        const val = from_axes_lut[@intFromEnum(forward)][@intFromEnum(up_dir)];
        if (val == 0xFF) return null;
        return .{ ._val = @intCast(val) };
    }

    /// Construct from a raw backing integer.
    ///
    /// Returns `null` if `v` is out of range (valid values are 0–23).
    /// Prefer the typed constructors (`fromFacing`, `fromAxes`) over this.
    pub fn fromRaw(v: u5) ?Ori24 {
        if (v >= 24) return null;
        return .{ ._val = v };
    }

    /// Returns the raw backing integer (0–23).
    pub fn rawVal(self: Ori24) u5 {
        return self._val;
    }

    // -- Comptime internals --------------------------------------------------

    const Mat = [3][3]i8; // Column-major: mat[col][row]

    const dir6_vecs: [6][3]i8 = .{
        .{ 0, 0, 1 }, // north (+Z)
        .{ 1, 0, 0 }, // east  (+X)
        .{ 0, 0, -1 }, // south (-Z)
        .{ -1, 0, 0 }, // west  (-X)
        .{ 0, 1, 0 }, // up    (+Y)
        .{ 0, -1, 0 }, // down  (-Y)
    };

    // Canonical up for spin 0, indexed by Dir6 facing.
    const default_ups: [6][3]i8 = .{
        .{ 0, 1, 0 }, // north → +Y
        .{ 0, 1, 0 }, // east  → +Y
        .{ 0, 1, 0 }, // south → +Y
        .{ 0, 1, 0 }, // west  → +Y
        .{ 0, 0, -1 }, // up    → -Z
        .{ 0, 0, 1 }, // down  → +Z
    };

    fn crossVec(a: [3]i8, b: [3]i8) [3]i8 {
        return .{
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        };
    }

    fn vecToDir6(v: [3]i8) u3 {
        for (dir6_vecs, 0..) |dv, i| {
            if (dv[0] == v[0] and dv[1] == v[1] and dv[2] == v[2])
                return @intCast(i);
        }
        unreachable;
    }

    fn matMulMat(a: Mat, b: Mat) Mat {
        var result: Mat = .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } };
        for (0..3) |j| {
            for (0..3) |i| {
                for (0..3) |k| {
                    result[j][i] += a[k][i] * b[j][k];
                }
            }
        }
        return result;
    }

    fn matEql(a: Mat, b: Mat) bool {
        inline for (0..3) |i| {
            inline for (0..3) |j| {
                if (a[i][j] != b[i][j]) return false;
            }
        }
        return true;
    }

    fn matIndex(m: Mat) u5 {
        for (matrices, 0..) |mat, i| {
            if (matEql(mat, m)) return @intCast(i);
        }
        unreachable;
    }

    // All 24 rotation matrices, ordered as facing_index * 4 + spin.
    const matrices: [24]Mat = blk: {
        var mats: [24]Mat = undefined;
        for (0..6) |fi| {
            const fwd = dir6_vecs[fi];
            var up_vec = default_ups[fi];
            for (0..4) |si| {
                const right_vec = crossVec(up_vec, fwd);
                mats[fi * 4 + si] = .{ right_vec, up_vec, fwd };
                // Rotate up 90° CW around facing axis for next spin.
                up_vec = crossVec(up_vec, fwd);
            }
        }
        break :blk mats;
    };

    const facing_lut: [24]u3 = blk: {
        var lut: [24]u3 = undefined;
        for (0..24) |i| lut[i] = vecToDir6(matrices[i][2]);
        break :blk lut;
    };

    const up_lut: [24]u3 = blk: {
        var lut: [24]u3 = undefined;
        for (0..24) |i| lut[i] = vecToDir6(matrices[i][1]);
        break :blk lut;
    };

    const right_lut: [24]u3 = blk: {
        var lut: [24]u3 = undefined;
        for (0..24) |i| lut[i] = vecToDir6(matrices[i][0]);
        break :blk lut;
    };

    const mul_lut: [24][24]u5 = blk: {
        @setEvalBranchQuota(100_000);
        var lut: [24][24]u5 = undefined;
        for (0..24) |a| {
            for (0..24) |b| {
                lut[a][b] = matIndex(matMulMat(matrices[a], matrices[b]));
            }
        }
        break :blk lut;
    };

    const inv_lut: [24]u5 = blk: {
        @setEvalBranchQuota(100_000);
        var lut: [24]u5 = undefined;
        for (0..24) |i| {
            // Inverse of orthogonal matrix = transpose.
            var t: Mat = undefined;
            for (0..3) |r| {
                for (0..3) |c| {
                    t[c][r] = matrices[i][r][c];
                }
            }
            lut[i] = matIndex(t);
        }
        break :blk lut;
    };

    const apply_lut: [24][6]u3 = blk: {
        @setEvalBranchQuota(100_000);
        var lut: [24][6]u3 = undefined;
        for (0..24) |oi| {
            for (0..6) |di| {
                const v = dir6_vecs[di];
                var result: [3]i8 = .{ 0, 0, 0 };
                for (0..3) |i| {
                    for (0..3) |k| {
                        result[i] += matrices[oi][k][i] * v[k];
                    }
                }
                lut[oi][di] = vecToDir6(result);
            }
        }
        break :blk lut;
    };

    const from_axes_lut: [6][6]u8 = blk: {
        var lut: [6][6]u8 = .{.{0xFF} ** 6} ** 6;
        for (0..24) |i| {
            lut[facing_lut[i]][up_lut[i]] = @intCast(i);
        }
        break :blk lut;
    };
};

// ===========================================================================
// Tests
// ===========================================================================

const testing = std.testing;

test "Dir4 forward offsets" {
    try testing.expectEqual([2]i32{ 0, 1 }, Dir4.north.forward().v);
    try testing.expectEqual([2]i32{ 1, 0 }, Dir4.east.forward().v);
    try testing.expectEqual([2]i32{ 0, -1 }, Dir4.south.forward().v);
    try testing.expectEqual([2]i32{ -1, 0 }, Dir4.west.forward().v);
}

test "Dir4 opposite round-trip" {
    for (Dir4.all) |d| {
        try testing.expectEqual(d, d.opposite().opposite());
    }
    try testing.expectEqual(Dir4.south, Dir4.north.opposite());
    try testing.expectEqual(Dir4.west, Dir4.east.opposite());
}

test "Dir4 cw/ccw cycle" {
    var d = Dir4.north;
    for (0..4) |_| d = d.cw();
    try testing.expectEqual(Dir4.north, d);

    d = Dir4.north;
    for (0..4) |_| d = d.ccw();
    try testing.expectEqual(Dir4.north, d);

    try testing.expectEqual(Dir4.east, Dir4.north.cw());
    try testing.expectEqual(Dir4.west, Dir4.north.ccw());
}

test "Dir4 cw and ccw are inverses" {
    for (Dir4.all) |d| {
        try testing.expectEqual(d, d.cw().ccw());
        try testing.expectEqual(d, d.ccw().cw());
    }
}

test "Dir4 → Dir8 → Dir4 round-trip" {
    for (Dir4.all) |d| {
        const d8 = d.toDir8();
        try testing.expect(d8.isCardinal());
        try testing.expectEqual(d, d8.toDir4().?);
    }
}

test "Dir4 → Dir6 → Dir4 round-trip" {
    for (Dir4.all) |d| {
        const d6 = d.toDir6();
        try testing.expect(d6.isHorizontal());
        try testing.expectEqual(d, d6.toDir4().?);
    }
}

test "Dir8 forward offsets" {
    try testing.expectEqual([2]i32{ 0, 1 }, Dir8.north.forward().v);
    try testing.expectEqual([2]i32{ 1, 1 }, Dir8.north_east.forward().v);
    try testing.expectEqual([2]i32{ 1, 0 }, Dir8.east.forward().v);
    try testing.expectEqual([2]i32{ 1, -1 }, Dir8.south_east.forward().v);
}

test "Dir8 opposite round-trip" {
    for (Dir8.all) |d| {
        try testing.expectEqual(d, d.opposite().opposite());
    }
    try testing.expectEqual(Dir8.south_west, Dir8.north_east.opposite());
}

test "Dir8 cw/ccw cycle" {
    var d = Dir8.north;
    for (0..8) |_| d = d.cw();
    try testing.expectEqual(Dir8.north, d);

    try testing.expectEqual(Dir8.north_east, Dir8.north.cw());
    try testing.expectEqual(Dir8.north_west, Dir8.north.ccw());
}

test "Dir8 cardinal/diagonal classification" {
    var cardinal_count: usize = 0;
    var diagonal_count: usize = 0;
    for (Dir8.all) |d| {
        if (d.isCardinal()) cardinal_count += 1;
        if (d.isDiagonal()) diagonal_count += 1;
        // Mutually exclusive
        try testing.expect(d.isCardinal() != d.isDiagonal());
    }
    try testing.expectEqual(@as(usize, 4), cardinal_count);
    try testing.expectEqual(@as(usize, 4), diagonal_count);
}

test "Dir8 toDir4 returns null for diagonals" {
    try testing.expect(Dir8.north_east.toDir4() == null);
    try testing.expect(Dir8.south_west.toDir4() == null);
    try testing.expect(Dir8.north.toDir4() != null);
}

test "Dir6 forward offsets" {
    try testing.expectEqual([3]i32{ 0, 0, 1 }, Dir6.north.forward().v);
    try testing.expectEqual([3]i32{ 1, 0, 0 }, Dir6.east.forward().v);
    try testing.expectEqual([3]i32{ 0, 1, 0 }, Dir6.up.forward().v);
    try testing.expectEqual([3]i32{ 0, -1, 0 }, Dir6.down.forward().v);
}

test "Dir6 opposite round-trip" {
    for (Dir6.all) |d| {
        try testing.expectEqual(d, d.opposite().opposite());
    }
    try testing.expectEqual(Dir6.south, Dir6.north.opposite());
    try testing.expectEqual(Dir6.down, Dir6.up.opposite());
}

test "Dir6 axis and sign" {
    try testing.expectEqual(Dir6.Axis.z, Dir6.north.axis());
    try testing.expectEqual(Dir6.Sign.positive, Dir6.north.sign());
    try testing.expectEqual(Dir6.Axis.z, Dir6.south.axis());
    try testing.expectEqual(Dir6.Sign.negative, Dir6.south.sign());
    try testing.expectEqual(Dir6.Axis.y, Dir6.up.axis());
    try testing.expectEqual(Dir6.Axis.x, Dir6.east.axis());
}

test "Dir6 horizontal/vertical classification" {
    var h: usize = 0;
    var v: usize = 0;
    for (Dir6.all) |d| {
        if (d.isHorizontal()) h += 1;
        if (d.isVertical()) v += 1;
        try testing.expect(d.isHorizontal() != d.isVertical());
    }
    try testing.expectEqual(@as(usize, 4), h);
    try testing.expectEqual(@as(usize, 2), v);
}

test "Dir6 toDir4 returns null for vertical" {
    try testing.expect(Dir6.up.toDir4() == null);
    try testing.expect(Dir6.down.toDir4() == null);
    try testing.expect(Dir6.north.toDir4() != null);
}

test "Dir6 → Dir26 → Dir6 round-trip" {
    for (Dir6.all) |d| {
        const d26 = d.toDir26();
        try testing.expectEqual(Dir26.Kind.face, d26.kind());
        try testing.expectEqual(d, d26.toDir6().?);
    }
}

test "Dir26 kind counts" {
    var face: usize = 0;
    var edge: usize = 0;
    var vertex: usize = 0;
    for (Dir26.all) |d| {
        switch (d.kind()) {
            .face => face += 1,
            .edge => edge += 1,
            .vertex => vertex += 1,
        }
    }
    try testing.expectEqual(@as(usize, 6), face);
    try testing.expectEqual(@as(usize, 12), edge);
    try testing.expectEqual(@as(usize, 8), vertex);
}

test "Dir26 opposite round-trip" {
    for (Dir26.all) |d| {
        try testing.expectEqual(d, d.opposite().opposite());
    }
    try testing.expectEqual(Dir26.south_west, Dir26.north_east.opposite());
    try testing.expectEqual(Dir26.down_south_west, Dir26.up_north_east.opposite());
}

test "Dir26 all offsets are unique and non-zero" {
    var seen = [_]bool{false} ** 27;
    for (Dir26.all) |d| {
        const f = d.forward();
        const ux: usize = @intCast(f.v[0] + 1);
        const uy: usize = @intCast(f.v[1] + 1);
        const uz: usize = @intCast(f.v[2] + 1);
        const idx = ux + uy * 3 + uz * 9;
        // Not the zero vector (center of 3x3x3)
        try testing.expect(idx != 13);
        // Not seen before
        try testing.expect(!seen[idx]);
        seen[idx] = true;
    }
}

test "Dir26 fromOffset round-trip" {
    for (Dir26.all) |d| {
        try testing.expectEqual(d, Dir26.fromOffset(d.forward()).?);
    }
    // Zero offset returns null
    try testing.expect(Dir26.fromOffset(Vec3i.init(.{ 0, 0, 0 })) == null);
    // Out-of-range offset returns null
    try testing.expect(Dir26.fromOffset(Vec3i.init(.{ 2, 0, 0 })) == null);
}

test "Dir26 toDir6 returns null for edges and vertices" {
    try testing.expect(Dir26.north_east.toDir6() == null);
    try testing.expect(Dir26.up_north_east.toDir6() == null);
    try testing.expect(Dir26.north.toDir6() != null);
}

test "Dir4 rel: specific cases" {
    try testing.expectEqual(Dir4.Rel.same, Dir4.north.rel(.north));
    try testing.expectEqual(Dir4.Rel.cw, Dir4.north.rel(.east));
    try testing.expectEqual(Dir4.Rel.opposite, Dir4.north.rel(.south));
    try testing.expectEqual(Dir4.Rel.ccw, Dir4.north.rel(.west));
    try testing.expectEqual(Dir4.Rel.ccw, Dir4.east.rel(.north));
}

test "Dir4 rel: self is always same" {
    for (Dir4.all) |d| {
        try testing.expectEqual(Dir4.Rel.same, d.rel(d));
    }
}

test "Dir4 rel: opposite is always opposite" {
    for (Dir4.all) |d| {
        try testing.expectEqual(Dir4.Rel.opposite, d.rel(d.opposite()));
    }
}

test "Dir4 rel: cw/ccw consistent with rotation methods" {
    for (Dir4.all) |d| {
        try testing.expectEqual(Dir4.Rel.cw, d.rel(d.cw()));
        try testing.expectEqual(Dir4.Rel.ccw, d.rel(d.ccw()));
    }
}

test "Dir4 Rel.invert" {
    try testing.expectEqual(Dir4.Rel.same, Dir4.Rel.same.invert());
    try testing.expectEqual(Dir4.Rel.opposite, Dir4.Rel.opposite.invert());
    try testing.expectEqual(Dir4.Rel.ccw, Dir4.Rel.cw.invert());
    try testing.expectEqual(Dir4.Rel.cw, Dir4.Rel.ccw.invert());
}

test "Dir4 rel: a.rel(b).invert() == b.rel(a)" {
    for (Dir4.all) |a| {
        for (Dir4.all) |b| {
            try testing.expectEqual(a.rel(b).invert(), b.rel(a));
        }
    }
}

test "Dir8 rel: specific cases" {
    try testing.expectEqual(Dir8.Rel.same, Dir8.north.rel(.north));
    try testing.expectEqual(Dir8.Rel.cw_45, Dir8.north.rel(.north_east));
    try testing.expectEqual(Dir8.Rel.cw_90, Dir8.north.rel(.east));
    try testing.expectEqual(Dir8.Rel.opposite, Dir8.north.rel(.south));
    try testing.expectEqual(Dir8.Rel.ccw_45, Dir8.north.rel(.north_west));
}

test "Dir8 rel: self is always same" {
    for (Dir8.all) |d| {
        try testing.expectEqual(Dir8.Rel.same, d.rel(d));
    }
}

test "Dir8 rel: cw/ccw consistent with rotation methods" {
    for (Dir8.all) |d| {
        try testing.expectEqual(Dir8.Rel.cw_45, d.rel(d.cw()));
        try testing.expectEqual(Dir8.Rel.ccw_45, d.rel(d.ccw()));
    }
}

test "Dir8 Rel.isCw/isCcw" {
    try testing.expect(Dir8.Rel.cw_45.isCw());
    try testing.expect(Dir8.Rel.cw_90.isCw());
    try testing.expect(Dir8.Rel.cw_135.isCw());
    try testing.expect(!Dir8.Rel.same.isCw());
    try testing.expect(!Dir8.Rel.opposite.isCw());
    try testing.expect(!Dir8.Rel.ccw_90.isCw());

    try testing.expect(Dir8.Rel.ccw_45.isCcw());
    try testing.expect(Dir8.Rel.ccw_90.isCcw());
    try testing.expect(Dir8.Rel.ccw_135.isCcw());
    try testing.expect(!Dir8.Rel.same.isCcw());
    try testing.expect(!Dir8.Rel.opposite.isCcw());
    try testing.expect(!Dir8.Rel.cw_90.isCcw());
}

test "Dir8 rel: a.rel(b).invert() == b.rel(a)" {
    for (Dir8.all) |a| {
        for (Dir8.all) |b| {
            try testing.expectEqual(a.rel(b).invert(), b.rel(a));
        }
    }
}

// -- DirSet tests ------------------------------------------------------------

test "DirSet6 insert/contains/remove" {
    const DirSet6 = DirSetType(Dir6);
    var s = DirSet6.empty;
    try testing.expect(!s.contains(.north));
    s = s.insert(.north).insert(.up);
    try testing.expect(s.contains(.north));
    try testing.expect(s.contains(.up));
    try testing.expect(!s.contains(.east));
    try testing.expectEqual(@as(usize, 2), s.count());

    s = s.remove(.north);
    try testing.expect(!s.contains(.north));
    try testing.expectEqual(@as(usize, 1), s.count());
}

test "DirSet6 set algebra" {
    const DirSet6 = DirSetType(Dir6);
    const a = DirSet6.of(.north).insert(.east);
    const b = DirSet6.of(.east).insert(.south);

    // Union
    const u = a.unionWith(b);
    try testing.expect(u.contains(.north));
    try testing.expect(u.contains(.east));
    try testing.expect(u.contains(.south));
    try testing.expectEqual(@as(usize, 3), u.count());

    // Intersection
    const i = a.intersect(b);
    try testing.expectEqual(@as(usize, 1), i.count());
    try testing.expect(i.contains(.east));

    // Difference
    const d = a.diff(b);
    try testing.expectEqual(@as(usize, 1), d.count());
    try testing.expect(d.contains(.north));

    // Symmetric difference
    const sd = a.symmetricDiff(b);
    try testing.expectEqual(@as(usize, 2), sd.count());
    try testing.expect(sd.contains(.north));
    try testing.expect(sd.contains(.south));
}

test "DirSet6 complement and full" {
    const DirSet6 = DirSetType(Dir6);
    try testing.expectEqual(@as(usize, 6), DirSet6.full.count());
    try testing.expectEqual(@as(usize, 0), DirSet6.empty.count());
    try testing.expect(DirSet6.full.complement().isEmpty());
    try testing.expect(DirSet6.empty.complement().eql(DirSet6.full));
}

test "DirSet6 subset/superset" {
    const DirSet6 = DirSetType(Dir6);
    const small = DirSet6.of(.north);
    const big = DirSet6.of(.north).insert(.east);
    try testing.expect(small.subsetOf(big));
    try testing.expect(!big.subsetOf(small));
    try testing.expect(big.supersetOf(small));
    try testing.expect(DirSet6.empty.subsetOf(small));
}

test "DirSet4 iterator" {
    const DirSet4 = DirSetType(Dir4);
    const s = DirSet4.of(.south).insert(.north).insert(.west);
    var it = s.iterator();

    // Should yield in enum order (north=0, south=2, west=3).
    try testing.expectEqual(Dir4.north, it.next().?);
    try testing.expectEqual(Dir4.south, it.next().?);
    try testing.expectEqual(Dir4.west, it.next().?);
    try testing.expect(it.next() == null);
}

test "DirSet toggle" {
    const DirSet6 = DirSetType(Dir6);
    var s = DirSet6.of(.north);
    s = s.toggle(.north).toggle(.east);
    try testing.expect(!s.contains(.north));
    try testing.expect(s.contains(.east));
}

// -- Ori24 tests -------------------------------------------------------------

test "Ori24 identity" {
    const id = Ori24.identity;
    try testing.expectEqual(Dir6.north, id.facing());
    try testing.expectEqual(Dir6.up, id.up());
    try testing.expectEqual(Dir6.east, id.right());
    try testing.expectEqual(@as(u2, 0), id.spin());
}

test "Ori24 identity mul" {
    const id = Ori24.identity;
    for (Ori24.all) |o| {
        try testing.expectEqual(o._val, id.mul(o)._val);
        try testing.expectEqual(o._val, o.mul(id)._val);
    }
}

test "Ori24 inverse" {
    const id = Ori24.identity;
    for (Ori24.all) |o| {
        try testing.expectEqual(id._val, o.mul(o.inverse())._val);
        try testing.expectEqual(id._val, o.inverse().mul(o)._val);
    }
}

test "Ori24 all 24 are distinct" {
    var seen = [_]bool{false} ** 24;
    for (Ori24.all) |o| {
        try testing.expect(!seen[o._val]);
        seen[o._val] = true;
    }
}

test "Ori24 facing/up/right are perpendicular and right-handed" {
    for (Ori24.all) |o| {
        const f = o.facing();
        const u = o.up();
        const r = o.right();
        // All on different axes (perpendicular).
        try testing.expect(f.axis() != u.axis());
        try testing.expect(f.axis() != r.axis());
        try testing.expect(u.axis() != r.axis());
    }
}

test "Ori24 apply identity" {
    const id = Ori24.identity;
    for (Dir6.all) |d| {
        try testing.expectEqual(d, id.apply(d));
    }
}

test "Ori24 apply matches facing/up/right" {
    for (Ori24.all) |o| {
        try testing.expectEqual(o.right(), o.apply(.east));
        try testing.expectEqual(o.up(), o.apply(.up));
        try testing.expectEqual(o.facing(), o.apply(.north));
    }
}

test "Ori24 mul associativity" {
    // Test a sampling of triples.
    const samples = [_]u5{ 0, 1, 5, 10, 15, 23 };
    for (samples) |ai| {
        const a = Ori24{ ._val = ai };
        for (samples) |bi| {
            const b = Ori24{ ._val = bi };
            for (samples) |ci| {
                const c = Ori24{ ._val = ci };
                try testing.expectEqual(a.mul(b).mul(c)._val, a.mul(b.mul(c))._val);
            }
        }
    }
}

test "Ori24 fromFacing" {
    for (Dir6.all) |d| {
        const o = Ori24.fromFacing(d);
        try testing.expectEqual(d, o.facing());
        try testing.expectEqual(@as(u2, 0), o.spin());
    }
}

test "Ori24 fromAxes round-trip" {
    for (Ori24.all) |o| {
        const rebuilt = Ori24.fromAxes(o.facing(), o.up()).?;
        try testing.expectEqual(o._val, rebuilt._val);
    }
}

test "Ori24 fromAxes rejects parallel" {
    try testing.expect(Ori24.fromAxes(.north, .north) == null);
    try testing.expect(Ori24.fromAxes(.north, .south) == null);
    try testing.expect(Ori24.fromAxes(.up, .down) == null);
}

test "Ori24 mul-apply consistency" {
    // a.mul(b).apply(v) == a.apply(b.apply(v))
    const samples = [_]u5{ 0, 3, 7, 12, 19, 23 };
    for (samples) |ai| {
        const a = Ori24{ ._val = ai };
        for (samples) |bi| {
            const b = Ori24{ ._val = bi };
            for (Dir6.all) |d| {
                try testing.expectEqual(a.apply(b.apply(d)), a.mul(b).apply(d));
            }
        }
    }
}
