//! Half-edge mesh: a connectivity data structure for manifold surfaces.
//!
//! Uses the XOR twin trick: half-edge pairs are allocated at consecutive
//! even/odd indices, so `twin(e) = e ^ 1`. This encodes the twin
//! relationship in the index bits with zero storage overhead.
//!
//! Each half-edge stores:
//!   - `vertex`: the vertex this half-edge points TO
//!   - `face`: the face to the LEFT (null for boundary edges)
//!   - `next`: the next half-edge around the same face (CCW winding)
//!
//! Each vertex stores one outgoing half-edge index (for traversal).
//! Each face stores one half-edge index on its boundary (for traversal).
//!
//! Winding convention: faces wind counter-clockwise. For a triangle
//! ABC, the half-edges run A→B→C→A around the face. The twin of
//! each half-edge borders the adjacent face (or the boundary).
//!
//! Boundary edges: half-edges with `face == null` are boundary edges.
//! After `finalize()`, boundary half-edges form closed loops around
//! holes in the mesh, with their `next` pointers chained together.
//!
//! Construction uses a separate `Builder`:
//!   1. Create a builder with `Builder.init()`
//!   2. Add vertices with `addVertex()`
//!   3. Add faces with `addFace()` (vertices in CCW order)
//!   4. Call `finalize()` to produce a `HalfEdgeMesh`
//!
//! The builder owns temporary construction state (edge lookup map)
//! that is freed on finalize. The resulting mesh carries only the
//! final topology arrays.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Typed index into the half-edge array.
pub const HalfEdgeId = struct { idx: u32 };
/// Typed index into the vertex array.
pub const VertexId = struct { idx: u32 };
/// Typed index into the face array.
pub const FaceId = struct { idx: u32 };

/// A directed half-edge in the mesh.
/// Half-edges are paired: each edge has two half-edges (twins) pointing
/// in opposite directions. The twin relationship is encoded via the XOR
/// trick: `twin(e) = e ^ 1`.
pub const HalfEdge = struct {
    /// Vertex this half-edge points TO.
    /// Null only transiently during construction; always set in a finalised mesh.
    vertex: ?VertexId,
    /// Face to the left of this half-edge. Null for boundary edges.
    face: ?FaceId,
    /// Next half-edge around the same face in CCW order.
    /// For boundary edges, links to the next boundary half-edge.
    /// Null only transiently during construction; always set in a finalised mesh.
    next: ?HalfEdgeId,
};

/// A vertex in the mesh.
/// Stores one outgoing half-edge for topology traversal.
pub const Vertex = struct {
    /// One outgoing half-edge from this vertex.
    /// For boundary vertices, points to a boundary half-edge.
    /// Null only transiently during construction; always set in a finalised mesh.
    edge: ?HalfEdgeId,
};

/// A face in the mesh.
/// Stores one half-edge on the face's boundary.
pub const Face = struct {
    /// One half-edge on this face's boundary (CCW winding).
    edge: HalfEdgeId,
};

/// Immutable half-edge mesh topology.
/// Produced by `Builder.finalize()`. The mesh owns its topology arrays
/// and must be freed with `deinit()`.
pub const HalfEdgeMesh = struct {
    half_edges: []const HalfEdge,
    vertices: []const Vertex,
    faces: []const Face,

    /// Frees all topology arrays.
    /// The mesh becomes invalid after this call.
    pub fn deinit(self: *HalfEdgeMesh, allocator: Allocator) void {
        allocator.free(self.half_edges);
        allocator.free(self.vertices);
        allocator.free(self.faces);
        self.* = undefined;
    }

    // ── Queries ──────────────────────────────────────────────────────

    /// Returns the twin of a half-edge (XOR trick: `e ^ 1`).
    /// The twin points in the opposite direction and borders the adjacent face.
    pub inline fn twin(e: HalfEdgeId) HalfEdgeId {
        return .{ .idx = e.idx ^ 1 };
    }

    /// Returns the vertex this half-edge originates from.
    pub fn edgeOrigin(self: *const HalfEdgeMesh, e: HalfEdgeId) VertexId {
        return self.half_edges[twin(e).idx].vertex.?;
    }

    /// Returns the vertex this half-edge points to.
    pub fn edgeTarget(self: *const HalfEdgeMesh, e: HalfEdgeId) VertexId {
        return self.half_edges[e.idx].vertex.?;
    }

    /// Returns true if this half-edge is on the boundary (has no face).
    pub fn isBoundary(self: *const HalfEdgeMesh, e: HalfEdgeId) bool {
        return self.half_edges[e.idx].face == null;
    }

    /// Returns the total number of half-edges.
    pub fn halfEdgeCount(self: *const HalfEdgeMesh) u32 {
        return @intCast(self.half_edges.len);
    }

    /// Returns the number of edges (half-edge pairs).
    /// Always half the number of half-edges.
    pub fn edgeCount(self: *const HalfEdgeMesh) u32 {
        return @intCast(self.half_edges.len / 2);
    }

    /// Returns the number of vertices.
    pub fn vertexCount(self: *const HalfEdgeMesh) u32 {
        return @intCast(self.vertices.len);
    }

    /// Returns the number of faces.
    pub fn faceCount(self: *const HalfEdgeMesh) u32 {
        return @intCast(self.faces.len);
    }

    /// Returns the Euler characteristic: V - E + F.
    /// For a closed genus-0 mesh (sphere topology), this is 2.
    /// For a mesh with one boundary loop, this is 1.
    pub fn eulerCharacteristic(self: *const HalfEdgeMesh) i32 {
        const v: i32 = @intCast(self.vertices.len);
        const e: i32 = @intCast(self.half_edges.len / 2);
        const f: i32 = @intCast(self.faces.len);
        return v - e + f;
    }

    // ── Iterators ────────────────────────────────────────────────────

    /// Iterator for half-edges around a face.
    /// Follows `next` pointers in CCW order.
    pub const FaceEdgeIter = struct {
        mesh: *const HalfEdgeMesh,
        start: HalfEdgeId,
        current: HalfEdgeId,
        started: bool,

        /// Returns the next half-edge around the face, or null when complete.
        pub fn next(self: *FaceEdgeIter) ?HalfEdgeId {
            if (self.started and self.current.idx == self.start.idx) return null;
            self.started = true;
            const result = self.current;
            self.current = self.mesh.half_edges[result.idx].next.?;
            return result;
        }
    };

    /// Returns an iterator over the half-edges around `face`.
    pub fn faceEdges(self: *const HalfEdgeMesh, face: FaceId) FaceEdgeIter {
        const start = self.faces[face.idx].edge;
        return .{
            .mesh = self,
            .start = start,
            .current = start,
            .started = false,
        };
    }

    /// Iterator for vertices around a face in winding order.
    pub const FaceVertexIter = struct {
        inner: FaceEdgeIter,

        /// Returns the next vertex around the face, or null when complete.
        pub fn next(self: *FaceVertexIter) ?VertexId {
            const he = self.inner.next() orelse return null;
            return self.inner.mesh.edgeOrigin(he);
        }
    };

    /// Returns an iterator over the vertices of `face` in CCW winding order.
    pub fn faceVertices(self: *const HalfEdgeMesh, face: FaceId) FaceVertexIter {
        return .{ .inner = self.faceEdges(face) };
    }

    /// Iterator for outgoing half-edges around a vertex.
    /// Visits spokes in CW order (opposite of face winding).
    /// For boundary vertices, visits from one boundary edge CW to the other.
    pub const VertexEdgeIter = struct {
        mesh: *const HalfEdgeMesh,
        start: HalfEdgeId,
        current: HalfEdgeId,
        started: bool,

        /// Returns the next outgoing half-edge, or null when complete.
        pub fn next(self: *VertexEdgeIter) ?HalfEdgeId {
            if (self.started and self.current.idx == self.start.idx) return null;
            self.started = true;
            const result = self.current;
            const t = twin(result);
            const n = self.mesh.half_edges[t.idx].next.?;
            self.current = n;
            return result;
        }
    };

    /// Returns an iterator over outgoing half-edges from `vert`.
    pub fn vertexEdges(self: *const HalfEdgeMesh, vert: VertexId) VertexEdgeIter {
        const start = self.vertices[vert.idx].edge.?;
        return .{
            .mesh = self,
            .start = start,
            .current = start,
            .started = false,
        };
    }

    /// Returns the number of edges incident to `vert` (vertex valence).
    pub fn vertexValence(self: *const HalfEdgeMesh, vert: VertexId) u32 {
        var iter = self.vertexEdges(vert);
        var count: u32 = 0;
        while (iter.next()) |_| count += 1;
        return count;
    }

    /// Returns true if `vert` is on the mesh boundary.
    /// A vertex is on the boundary if its stored half-edge has no face.
    pub fn isVertexBoundary(self: *const HalfEdgeMesh, vert: VertexId) bool {
        const edge = self.vertices[vert.idx].edge.?;
        return self.half_edges[edge.idx].face == null;
    }

    // ── Builder ──────────────────────────────────────────────────────

    /// Mutable builder for constructing a `HalfEdgeMesh`.
    /// Owns temporary construction state (edge lookup map) that is freed
    /// on `finalize()`. After finalization, the builder is poisoned and
    /// cannot be reused.
    pub const Builder = struct {
        half_edges: std.ArrayListUnmanaged(HalfEdge),
        vertices: std.ArrayListUnmanaged(Vertex),
        faces: std.ArrayListUnmanaged(Face),
        edge_lookup: std.AutoHashMapUnmanaged(VertexPair, HalfEdgeId),

        const VertexPair = struct { from: VertexId, to: VertexId };

        pub fn init() Builder {
            return .{
                .half_edges = .empty,
                .vertices = .empty,
                .faces = .empty,
                .edge_lookup = .empty,
            };
        }

        /// Discards the builder without producing a mesh.
        /// Frees all internal state.
        pub fn deinit(self: *Builder, allocator: Allocator) void {
            self.half_edges.deinit(allocator);
            self.vertices.deinit(allocator);
            self.faces.deinit(allocator);
            self.edge_lookup.deinit(allocator);
            self.* = undefined;
        }

        /// Adds a new vertex to the mesh.
        /// Returns the vertex's ID.
        pub fn addVertex(self: *Builder, allocator: Allocator) Allocator.Error!VertexId {
            const id: VertexId = .{ .idx = @intCast(self.vertices.items.len) };
            try self.vertices.append(allocator, .{ .edge = null });
            return id;
        }

        /// Adds a face from vertices given in CCW winding order.
        /// Each vertex pair gets a half-edge pair allocated (or reuses an
        /// existing pair if the reverse edge was already created by an
        /// adjacent face).
        /// Asserts that `verts` has at least 3 vertices.
        pub fn addFace(self: *Builder, allocator: Allocator, verts: []const VertexId) Allocator.Error!FaceId {
            std.debug.assert(verts.len >= 3);

            const face_id: FaceId = .{ .idx = @intCast(self.faces.items.len) };

            // Allocate or find the half-edge for each edge of this face.
            for (verts, 0..) |_, i| {
                const v_from = verts[i];
                const v_to = verts[(i + 1) % verts.len];
                _ = try self.getOrCreateEdgePair(allocator, v_from, v_to);
            }

            // Wire up all the half-edges for this face.
            for (verts, 0..) |_, i| {
                const v_from = verts[i];
                const v_to = verts[(i + 1) % verts.len];
                const v_next_to = verts[(i + 2) % verts.len];

                const he = self.edge_lookup.get(.{ .from = v_from, .to = v_to }).?;
                const he_next = self.edge_lookup.get(.{ .from = v_to, .to = v_next_to }).?;

                self.half_edges.items[he.idx].vertex = v_to;
                self.half_edges.items[he.idx].face = face_id;
                self.half_edges.items[he.idx].next = he_next;

                if (self.vertices.items[v_from.idx].edge == null) {
                    self.vertices.items[v_from.idx].edge = he;
                }
            }

            const face_edge = self.edge_lookup.get(.{ .from = verts[0], .to = verts[1] }).?;
            try self.faces.append(allocator, .{ .edge = face_edge });

            return face_id;
        }

        /// Finalizes the mesh: links boundary loops, fixes vertex pointers,
        /// frees construction state, and returns an immutable `HalfEdgeMesh`.
        /// The builder is poisoned after this call and cannot be reused.
        pub fn finalize(self: *Builder, allocator: Allocator) Allocator.Error!HalfEdgeMesh {
            // Link boundary half-edges into closed loops.
            for (self.half_edges.items, 0..) |*he, i| {
                if (he.face != null) continue;
                if (he.next != null) continue;

                var cursor: HalfEdgeId = twin(.{ .idx = @intCast(i) });

                while (true) {
                    const prev_he = self.prevOnFace(cursor);
                    const candidate = twin(prev_he);

                    if (self.half_edges.items[candidate.idx].face == null) {
                        he.next = candidate;
                        break;
                    }
                    cursor = candidate;
                }
            }

            // Fix vertex edge pointers for boundary vertices.
            for (self.half_edges.items, 0..) |he, i| {
                if (he.face != null) continue;
                const boundary_he: HalfEdgeId = .{ .idx = @intCast(i) };
                const origin = self.half_edges.items[twin(boundary_he).idx].vertex.?;
                self.vertices.items[origin.idx].edge = boundary_he;
            }

            // Free lookup — no longer needed.
            self.edge_lookup.deinit(allocator);
            self.edge_lookup = .empty;

            // Transfer ownership of the arrays to the mesh.
            // toOwnedSlice() shrinks the backing allocation to exactly
            // items.len so that deinit's allocator.free() sees a matching size.
            const mesh = HalfEdgeMesh{
                .half_edges = try self.half_edges.toOwnedSlice(allocator),
                .vertices = try self.vertices.toOwnedSlice(allocator),
                .faces = try self.faces.toOwnedSlice(allocator),
            };

            return mesh;
        }

        fn getOrCreateEdgePair(
            self: *Builder,
            allocator: Allocator,
            from: VertexId,
            to: VertexId,
        ) Allocator.Error!HalfEdgeId {
            const forward_key = VertexPair{ .from = from, .to = to };

            if (self.edge_lookup.get(forward_key)) |existing| {
                return existing;
            }

            const he: HalfEdgeId = .{ .idx = @intCast(self.half_edges.items.len) };
            std.debug.assert(he.idx % 2 == 0);

            const blank = HalfEdge{
                .vertex = null,
                .face = null,
                .next = null,
            };
            try self.half_edges.append(allocator, blank);
            try self.half_edges.append(allocator, blank);

            self.half_edges.items[he.idx ^ 1].vertex = from;

            try self.edge_lookup.put(allocator, forward_key, he);
            try self.edge_lookup.put(allocator, .{ .from = to, .to = from }, .{ .idx = he.idx ^ 1 });

            return he;
        }

        fn prevOnFace(self: *const Builder, e: HalfEdgeId) HalfEdgeId {
            var cursor = e;
            while (true) {
                const n = self.half_edges.items[cursor.idx].next.?;
                if (n.idx == e.idx) return cursor;
                cursor = n;
            }
        }
    };
};

// ── Tests ────────────────────────────────────────────────────────────

const testing = std.testing;

test "single triangle" {
    const allocator = testing.allocator;
    var builder = HalfEdgeMesh.Builder.init();

    const v0 = try builder.addVertex(allocator);
    const v1 = try builder.addVertex(allocator);
    const v2 = try builder.addVertex(allocator);

    const f0 = try builder.addFace(allocator, &.{ v0, v1, v2 });
    var mesh = try builder.finalize(allocator);
    defer mesh.deinit(allocator);

    try testing.expectEqual(@as(u32, 3), mesh.vertexCount());
    try testing.expectEqual(@as(u32, 3), mesh.edgeCount());
    try testing.expectEqual(@as(u32, 6), mesh.halfEdgeCount());
    try testing.expectEqual(@as(u32, 1), mesh.faceCount());

    // Euler: V - E + F = 3 - 3 + 1 = 1 (one boundary loop)
    try testing.expectEqual(@as(i32, 1), mesh.eulerCharacteristic());

    // Face vertex iteration should give us back v0, v1, v2.
    var fv = mesh.faceVertices(f0);
    try testing.expectEqual(v0, fv.next().?);
    try testing.expectEqual(v1, fv.next().?);
    try testing.expectEqual(v2, fv.next().?);
    try testing.expectEqual(@as(?VertexId, null), fv.next());

    // All vertices are boundary vertices.
    try testing.expect(mesh.isVertexBoundary(v0));
    try testing.expect(mesh.isVertexBoundary(v1));
    try testing.expect(mesh.isVertexBoundary(v2));

    // Each vertex has valence 2 (one interior spoke + one boundary spoke).
    try testing.expectEqual(@as(u32, 2), mesh.vertexValence(v0));
    try testing.expectEqual(@as(u32, 2), mesh.vertexValence(v1));
    try testing.expectEqual(@as(u32, 2), mesh.vertexValence(v2));
}

test "twin XOR" {
    for (0..20) |i| {
        const e: HalfEdgeId = .{ .idx = @intCast(i) };
        try testing.expectEqual(e, HalfEdgeMesh.twin(HalfEdgeMesh.twin(e)));
    }
    try testing.expectEqual(HalfEdgeId{ .idx = 1 }, HalfEdgeMesh.twin(.{ .idx = 0 }));
    try testing.expectEqual(HalfEdgeId{ .idx = 0 }, HalfEdgeMesh.twin(.{ .idx = 1 }));
    try testing.expectEqual(HalfEdgeId{ .idx = 3 }, HalfEdgeMesh.twin(.{ .idx = 2 }));
    try testing.expectEqual(HalfEdgeId{ .idx = 2 }, HalfEdgeMesh.twin(.{ .idx = 3 }));
}

test "two triangles sharing an edge" {
    const allocator = testing.allocator;
    var builder = HalfEdgeMesh.Builder.init();

    const v0 = try builder.addVertex(allocator);
    const v1 = try builder.addVertex(allocator);
    const v2 = try builder.addVertex(allocator);
    const v3 = try builder.addVertex(allocator);

    _ = try builder.addFace(allocator, &.{ v0, v1, v2 });
    _ = try builder.addFace(allocator, &.{ v0, v2, v3 });
    var mesh = try builder.finalize(allocator);
    defer mesh.deinit(allocator);

    try testing.expectEqual(@as(u32, 4), mesh.vertexCount());
    try testing.expectEqual(@as(u32, 5), mesh.edgeCount());
    try testing.expectEqual(@as(u32, 2), mesh.faceCount());

    // Euler: V - E + F = 4 - 5 + 2 = 1
    try testing.expectEqual(@as(i32, 1), mesh.eulerCharacteristic());

    try testing.expect(mesh.isVertexBoundary(v0));
    try testing.expect(mesh.isVertexBoundary(v1));
    try testing.expect(mesh.isVertexBoundary(v2));
    try testing.expect(mesh.isVertexBoundary(v3));

    try testing.expectEqual(@as(u32, 3), mesh.vertexValence(v0));
    try testing.expectEqual(@as(u32, 2), mesh.vertexValence(v1));
    try testing.expectEqual(@as(u32, 3), mesh.vertexValence(v2));
    try testing.expectEqual(@as(u32, 2), mesh.vertexValence(v3));
}

test "quad from two triangles — boundary loop" {
    const allocator = testing.allocator;
    var builder = HalfEdgeMesh.Builder.init();

    const v0 = try builder.addVertex(allocator);
    const v1 = try builder.addVertex(allocator);
    const v2 = try builder.addVertex(allocator);
    const v3 = try builder.addVertex(allocator);

    _ = try builder.addFace(allocator, &.{ v0, v1, v2 });
    _ = try builder.addFace(allocator, &.{ v0, v2, v3 });
    var mesh = try builder.finalize(allocator);
    defer mesh.deinit(allocator);

    // Walk the boundary loop: should visit exactly 4 boundary edges.
    var boundary_count: u32 = 0;
    var boundary_start: ?HalfEdgeId = null;

    for (mesh.half_edges, 0..) |he, i| {
        if (he.face == null) {
            boundary_start = .{ .idx = @intCast(i) };
            break;
        }
    }

    if (boundary_start) |start| {
        var cursor = start;
        while (true) {
            boundary_count += 1;
            cursor = mesh.half_edges[cursor.idx].next.?;
            if (cursor.idx == start.idx) break;
            if (boundary_count > 100) break;
        }
    }

    try testing.expectEqual(@as(u32, 4), boundary_count);
}

test "tetrahedron — closed mesh" {
    const allocator = testing.allocator;
    var builder = HalfEdgeMesh.Builder.init();

    const v0 = try builder.addVertex(allocator);
    const v1 = try builder.addVertex(allocator);
    const v2 = try builder.addVertex(allocator);
    const v3 = try builder.addVertex(allocator);

    _ = try builder.addFace(allocator, &.{ v0, v1, v2 });
    _ = try builder.addFace(allocator, &.{ v0, v3, v1 });
    _ = try builder.addFace(allocator, &.{ v1, v3, v2 });
    _ = try builder.addFace(allocator, &.{ v0, v2, v3 });
    var mesh = try builder.finalize(allocator);
    defer mesh.deinit(allocator);

    try testing.expectEqual(@as(u32, 4), mesh.vertexCount());
    try testing.expectEqual(@as(u32, 6), mesh.edgeCount());
    try testing.expectEqual(@as(u32, 4), mesh.faceCount());

    // Euler characteristic = 2 for a closed mesh.
    try testing.expectEqual(@as(i32, 2), mesh.eulerCharacteristic());

    // No boundary edges.
    for (mesh.half_edges) |he| {
        try testing.expect(he.face != null);
    }

    // No boundary vertices.
    for (0..4) |i| {
        try testing.expect(!mesh.isVertexBoundary(.{ .idx = @intCast(i) }));
    }

    // Each vertex of a tetrahedron has valence 3.
    for (0..4) |i| {
        try testing.expectEqual(@as(u32, 3), mesh.vertexValence(.{ .idx = @intCast(i) }));
    }
}
