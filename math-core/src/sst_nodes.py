"""
SymmetryKit — SST Node System (Python Reference Implementation)
================================================================
The functional alphabet: every node is a matrix-producing operation.

Phase 3: T, R, S, Scope, Platform, Mirror, Radial, Instance + JSON parser
Phase 6: Spread, Field, Connect, Collapse

Reference: core/node_algebra.md, core/extended_state_algebra.md
"""

from __future__ import annotations
import json
import math
import copy
from typing import List, Dict, Optional, Any, Tuple
from math_core import (Vec3, Mat4, MatrixStack, align_to_surface,
                       platform_reflect, SK_EPSILON, SK_DEG2RAD)


# ===========================================================================
# Walker State — Extended with Spread mode
# ===========================================================================

class WalkerState:
    """Full traversal state for SST evaluation."""

    def __init__(self):
        self.stack = MatrixStack()
        self.platforms: Dict[str, Mat4] = {}
        self.active_platform: Optional[str] = None
        self.sym_depth: int = 0
        self.buffer: List[Dict] = []

        # Phase 6: Spread mode
        self.context_mode: str = 'SINGLE'  # 'SINGLE' or 'SPREAD'
        self.spread_buffer: List[Mat4] = []
        self.spread_index: int = -1
        self.spread_topology: str = 'NONE'  # NONE, SEQUENTIAL, CLOSED, GRID
        self.grid_nx: int = 0
        self.grid_ny: int = 0

    @property
    def M(self) -> Mat4:
        return self.stack.current

    @M.setter
    def M(self, value: Mat4):
        self.stack.current = value

    def push(self):
        self.stack.push()

    def pop(self):
        self.stack.pop()

    def transform(self, T: Mat4):
        self.stack.transform(T)

    def is_spread(self) -> bool:
        return self.context_mode == 'SPREAD'

    def enter_spread(self, matrices: List[Mat4]):
        self.spread_buffer = matrices
        self.context_mode = 'SPREAD'
        self.spread_index = -1
        self.spread_topology = 'NONE'

    def exit_spread(self):
        self.spread_buffer = []
        self.context_mode = 'SINGLE'
        self.spread_index = -1
        self.spread_topology = 'NONE'
        self.grid_nx = 0
        self.grid_ny = 0

    def get_active_platform_matrix(self) -> Mat4:
        if self.active_platform and self.active_platform in self.platforms:
            return self.platforms[self.active_platform]
        return Mat4.identity()

    def emit(self, mesh_ref: str, tags: Optional[Dict[str, str]] = None):
        entry = {
            'mesh': mesh_ref,
            'transform': self.M.copy(),
            'sym_depth': self.sym_depth,
            'spread_index': self.spread_index,
            'tags': dict(tags) if tags else {},
        }
        if self.sym_depth > 0:
            entry['tags']['is_symmetry'] = 'true'
        self.buffer.append(entry)

    def reset(self):
        self.stack.reset()
        self.platforms.clear()
        self.active_platform = None
        self.sym_depth = 0
        self.buffer.clear()
        self.exit_spread()


# ===========================================================================
# INode — Base class
# ===========================================================================

class Node:
    """Base class for all SST nodes."""

    def __init__(self, name: str = 'Node'):
        self.name = name
        self.children: List[Node] = []

    def add_child(self, child: Node) -> Node:
        self.children.append(child)
        return child

    def execute(self, state: WalkerState):
        self.exec_self(state)
        self.execute_children(state)

    def exec_self(self, state: WalkerState):
        pass

    def execute_children(self, state: WalkerState):
        for child in self.children:
            child.execute(state)


# ===========================================================================
# Transform Nodes — T, R, S
# ===========================================================================

class TranslateNode(Node):
    def __init__(self, v: Vec3):
        super().__init__('T')
        self.v = v

    def exec_self(self, state: WalkerState):
        state.transform(Mat4.translate(self.v))


class RotateNode(Node):
    def __init__(self, axis: str, degrees: float):
        super().__init__('R')
        self.axis = axis
        self.degrees = degrees

    def exec_self(self, state: WalkerState):
        state.transform(Mat4.rotate(self.axis, self.degrees))


class ScaleNode(Node):
    def __init__(self, v: Vec3):
        super().__init__('S')
        self.v = v

    def exec_self(self, state: WalkerState):
        state.transform(Mat4.scale(self.v))


# ===========================================================================
# Scope / Structure Nodes
# ===========================================================================

class ScopeNode(Node):
    def __init__(self):
        super().__init__('Scope')

    def execute(self, state: WalkerState):
        state.push()
        self.execute_children(state)
        state.pop()


class PlatformNode(Node):
    def __init__(self, platform_id: str, from_current: bool = True):
        super().__init__('Platform')
        self.platform_id = platform_id
        self.from_current = from_current
        self.explicit_matrix = Mat4.identity()

    def exec_self(self, state: WalkerState):
        mat = state.M.copy() if self.from_current else self.explicit_matrix
        state.platforms[self.platform_id] = mat
        state.active_platform = self.platform_id


# ===========================================================================
# Flow Nodes — Mirror, Radial
# ===========================================================================

class MirrorNode(Node):
    """Bifurcation: real branch + reflected branch.
    M_reflect = P x S x P^-1"""

    def __init__(self, axis: str = 'X'):
        super().__init__('Mirror')
        self.axis = axis

    def execute(self, state: WalkerState):
        # Both branches must start from the same parent state.
        # In row-vector post-multiply convention (p' = p @ M), M_reflect must
        # be applied AFTER children's transforms: T_child @ M_reflect.
        # We let children emit normally then post-multiply M_reflect onto
        # all newly emitted geometry from the reflected branch.

        # PATH A: Real branch
        state.push()
        self.execute_children(state)
        state.pop()

        # PATH B: Reflected branch
        if state.active_platform is not None:
            P = state.get_active_platform_matrix()
            M_reflect = platform_reflect(P, self.axis)

            emit_start = len(state.buffer)

            state.push()
            state.sym_depth += 1

            self.execute_children(state)

            state.sym_depth -= 1
            state.pop()

            # Post-apply reflection to all emissions from the reflected branch
            for i in range(emit_start, len(state.buffer)):
                state.buffer[i]['transform'] = state.buffer[i]['transform'] @ M_reflect
                state.buffer[i]['sym_depth'] = max(state.buffer[i]['sym_depth'], 1)


class RadialNode(Node):
    """Replicate n times rotated around axis."""

    def __init__(self, n: int, axis: str = 'Z'):
        super().__init__('Radial')
        self.n = n
        self.axis = axis

    def execute(self, state: WalkerState):
        if self.n <= 0:
            return

        angle_step = 360.0 / self.n

        for i in range(self.n):
            state.push()
            state.transform(Mat4.rotate(self.axis, i * angle_step))

            if i > 0:
                state.sym_depth += 1

            self.execute_children(state)

            if i > 0:
                state.sym_depth -= 1

            state.pop()


# ===========================================================================
# Emission Node — Instance
# ===========================================================================

class InstanceNode(Node):
    def __init__(self, mesh_ref: str, tags: Optional[Dict[str, str]] = None):
        super().__init__('Instance')
        self.mesh_ref = mesh_ref
        self.tags = tags or {}

    def exec_self(self, state: WalkerState):
        emit_tags = dict(self.tags)
        emit_tags['is_symmetry'] = 'true' if state.sym_depth > 0 else 'false'
        state.emit(self.mesh_ref, emit_tags)


# ===========================================================================
# Phase 6 — Spread Node
# ===========================================================================

class SpreadNode(Node):
    """Generate array of matrices, execute children once per element."""

    def __init__(self, source: str = 'Linear', **params):
        super().__init__('Spread')
        self.source = source
        self.params = params

    def execute(self, state: WalkerState):
        matrices = self._generate(state)
        if not matrices:
            return

        # Save previous spread state (for nesting)
        prev_mode   = state.context_mode
        prev_buffer = state.spread_buffer
        prev_index  = state.spread_index
        prev_topo   = state.spread_topology

        state.enter_spread(matrices)

        for i, M in enumerate(matrices):
            state.spread_index = i
            state.push()
            state.transform(M)
            self.execute_children(state)
            state.pop()

        # Restore
        state.context_mode    = prev_mode
        state.spread_buffer   = prev_buffer
        state.spread_index    = prev_index
        state.spread_topology = prev_topo

    def _generate(self, state: WalkerState) -> List[Mat4]:
        s = self.source.lower()
        if s == 'linear':
            return self._gen_linear()
        elif s == 'radial':
            return self._gen_radial()
        elif s == 'grid':
            return self._gen_grid()
        elif s == 'range':
            return self._gen_range()
        return []

    def _gen_linear(self) -> List[Mat4]:
        start = self.params.get('start', Vec3(0, 0, 0))
        end   = self.params.get('end', Vec3(1000, 0, 0))
        n     = self.params.get('n', 10)
        if isinstance(start, (list, tuple)):
            start = Vec3(*start)
        if isinstance(end, (list, tuple)):
            end = Vec3(*end)

        result = []
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0.0
            pos = Vec3(
                start.x + (end.x - start.x) * t,
                start.y + (end.y - start.y) * t,
                start.z + (end.z - start.z) * t,
            )
            result.append(Mat4.translate(pos))
        return result

    def _gen_radial(self) -> List[Mat4]:
        n      = self.params.get('n', 8)
        radius = self.params.get('radius', 100.0)
        axis   = self.params.get('axis', 'Z')
        result = []

        for i in range(n):
            angle = 2 * math.pi * i / n
            if axis == 'Z':
                offset = Vec3(math.cos(angle) * radius, math.sin(angle) * radius, 0)
            elif axis == 'Y':
                offset = Vec3(math.cos(angle) * radius, 0, math.sin(angle) * radius)
            else:
                offset = Vec3(0, math.cos(angle) * radius, math.sin(angle) * radius)

            M = Mat4.translate(offset)
            R = Mat4.rotate(axis, math.degrees(angle))
            result.append(M @ R)

        return result

    def _gen_grid(self) -> List[Mat4]:
        nx      = self.params.get('nx', 5)
        ny      = self.params.get('ny', 5)
        spacing = self.params.get('spacing', Vec3(100, 100, 0))
        if isinstance(spacing, (list, tuple)):
            spacing = Vec3(*spacing)

        half_x = (nx - 1) * spacing.x * 0.5
        half_y = (ny - 1) * spacing.y * 0.5

        result = []
        for iy in range(ny):
            for ix in range(nx):
                pos = Vec3(ix * spacing.x - half_x, iy * spacing.y - half_y, spacing.z)
                result.append(Mat4.translate(pos))
        return result

    def _gen_range(self) -> List[Mat4]:
        n = self.params.get('n', 10)
        return [Mat4.identity() for _ in range(n)]


# ===========================================================================
# Phase 6 — Field Node
# ===========================================================================

class FieldNode(Node):
    """Spatially-varying transformation: noise, attractor, repel, vortex."""

    def __init__(self, field_type: str = 'Noise', operation: str = 'ADVECT',
                 intensity: float = 1.0, scale: float = 0.01,
                 target: Optional[Vec3] = None, seed: int = 0):
        super().__init__('Field')
        self.field_type = field_type
        self.operation  = operation
        self.intensity  = intensity
        self.scale      = scale
        self.target     = target or Vec3(0, 0, 0)
        self.seed       = seed

    def exec_self(self, state: WalkerState):
        pos = state.M.get_translation()
        field_val = self._evaluate(pos)
        displacement = Vec3(
            field_val.x * self.intensity,
            field_val.y * self.intensity,
            field_val.z * self.intensity,
        )

        if self.operation == 'ADVECT':
            state.transform(Mat4.translate(displacement))
        elif self.operation == 'SCALE':
            mag = displacement.length()
            if mag > SK_EPSILON:
                state.transform(Mat4.scale(Vec3(mag, mag, mag)))

    def _evaluate(self, p: Vec3) -> Vec3:
        ft = self.field_type.lower()

        if ft == 'noise':
            # Simple pseudo-noise (hash-based, deterministic)
            def _hash_noise(x: float, y: float, z: float) -> float:
                n = math.sin(x * 12.9898 + y * 78.233 + z * 45.164) * 43758.5453
                return n - math.floor(n) - 0.5

            sx, sy, sz = p.x * self.scale, p.y * self.scale, p.z * self.scale
            return Vec3(
                _hash_noise(sx, sy, sz),
                _hash_noise(sx + 31.416, sy, sz),
                _hash_noise(sx, sy + 47.853, sz),
            )

        elif ft == 'attractor':
            d = self.target - p
            dist = d.length()
            if dist < 0.001:
                return Vec3(0, 0, 0)
            return d.normalized() * (1.0 / max(dist * dist, 0.01))

        elif ft == 'repel':
            d = p - self.target
            dist = d.length()
            if dist < 0.001:
                return Vec3(0, 0, 0)
            return d.normalized() * (1.0 / max(dist * dist, 0.01))

        elif ft == 'vortex':
            rel = p - self.target
            up = Vec3(0, 1, 0)
            return up.cross(rel)

        return Vec3(0, 0, 0)


# ===========================================================================
# Phase 6B — Frame Node (Topology → Matrices)
# ===========================================================================

class FrameNode(Node):
    """
    Extract coordinate frames from mesh topology features.

    Query topology components (vertices, edges, faces, curves) and convert
    each to a local coordinate frame (4x4 matrix). Enters SPREAD mode.

    Query syntax:
        "vertex[*]"     — All vertices
        "vertex[0:10]"  — Vertices 0-9
        "edge[*]"       — All edges (frame at midpoint)
        "face[*]"       — All faces (frame at centroid)
        "curve[n=100]"  — n samples along curve

    Frame modes:
        DARBOUX   — Surface frame (normal, principal tangent, binormal)
        FRENET    — Curve frame (tangent, normal, binormal)
        PARALLEL  — Parallel transport along curve
        VERTEX    — Vertex normal + longest edge direction

    Since we don't have access to actual mesh geometry in the pure-math layer,
    this implementation generates synthetic frames for testing. In the UE5
    binding, real mesh queries are performed.
    """

    def __init__(self, geo_ref: str = '', query: str = 'vertex[*]',
                 mode: str = 'DARBOUX', count: int = 0):
        super().__init__('Frame')
        self.geo_ref = geo_ref
        self.query = query
        self.mode = mode
        self.count = count  # For synthetic generation when geo_ref is empty

    def execute(self, state: WalkerState):
        # Parse the query to determine component type and range
        frames = self._generate_frames()

        if len(frames) == 0:
            self.execute_children(state)
            return

        # Enter spread mode
        state.enter_spread(frames)
        state.spread_topology = 'SEQUENTIAL'

        for i, M in enumerate(frames):
            state.spread_index = i
            state.push()
            state.transform(M)
            self.execute_children(state)
            state.pop()

        state.exit_spread()

    def _generate_frames(self) -> List[Mat4]:
        """
        Generate frames based on query.

        In pure Python (no mesh access), we generate synthetic frames for testing:
        - vertex[*] or vertex[n] → n points on a sphere
        - edge[*] or edge[n]     → n points along an edge loop
        - face[*] or face[n]     → n points on a grid
        - curve[n=...]           → n points along a helix

        Real implementations query actual mesh topology.
        """
        import re

        # Parse query: "component[range]" or "component[key=value]"
        match = re.match(r'(\w+)\[([^\]]*)\]', self.query)
        if not match:
            return []

        component = match.group(1).lower()
        range_str = match.group(2)

        # Determine count
        n = self.count if self.count > 0 else 16

        if range_str == '*':
            pass  # Use default n
        elif ':' in range_str:
            parts = range_str.split(':')
            if len(parts) == 2:
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                    n = end - start
                except ValueError:
                    pass
        elif '=' in range_str:
            # key=value format, e.g., "n=100"
            parts = range_str.split('=')
            if len(parts) == 2:
                try:
                    n = int(parts[1])
                except ValueError:
                    pass
        else:
            try:
                n = int(range_str)
            except ValueError:
                pass

        n = max(1, min(n, 1000))  # Clamp

        if component == 'vertex':
            return self._synth_vertex_frames(n)
        elif component == 'edge':
            return self._synth_edge_frames(n)
        elif component == 'face':
            return self._synth_face_frames(n)
        elif component == 'curve':
            return self._synth_curve_frames(n)
        else:
            return []

    def _synth_vertex_frames(self, n: int) -> List[Mat4]:
        """Generate n frames distributed on a sphere (Fibonacci lattice)."""
        frames = []
        golden = (1 + math.sqrt(5)) / 2
        for i in range(n):
            theta = 2 * math.pi * i / golden
            phi = math.acos(1 - 2 * (i + 0.5) / n)
            x = math.sin(phi) * math.cos(theta)
            y = math.sin(phi) * math.sin(theta)
            z = math.cos(phi)
            pos = Vec3(x * 100, y * 100, z * 100)
            normal = Vec3(x, y, z)
            frames.append(self._build_frame(pos, normal))
        return frames

    def _synth_edge_frames(self, n: int) -> List[Mat4]:
        """Generate n frames along a circular edge loop."""
        frames = []
        for i in range(n):
            t = 2 * math.pi * i / n
            pos = Vec3(math.cos(t) * 100, math.sin(t) * 100, 0)
            tangent = Vec3(-math.sin(t), math.cos(t), 0)
            normal = Vec3(0, 0, 1)
            frames.append(self._build_frame_tn(pos, tangent, normal))
        return frames

    def _synth_face_frames(self, n: int) -> List[Mat4]:
        """Generate n frames on a grid of faces."""
        frames = []
        side = int(math.ceil(math.sqrt(n)))
        for i in range(n):
            ix = i % side
            iy = i // side
            pos = Vec3(ix * 50 - side * 25, iy * 50 - side * 25, 0)
            normal = Vec3(0, 0, 1)
            frames.append(self._build_frame(pos, normal))
        return frames

    def _synth_curve_frames(self, n: int) -> List[Mat4]:
        """Generate n frames along a helix curve (Frenet frame)."""
        frames = []
        height = 200
        radius = 100
        turns = 2
        for i in range(n):
            t = i / max(1, n - 1)
            angle = t * turns * 2 * math.pi
            x = math.cos(angle) * radius
            y = math.sin(angle) * radius
            z = t * height
            pos = Vec3(x, y, z)

            # Tangent (derivative of helix)
            tx = -math.sin(angle) * radius
            ty = math.cos(angle) * radius
            tz = height / (turns * 2 * math.pi)
            tangent = Vec3(tx, ty, tz).normalized()

            # Normal (toward center)
            normal = Vec3(-math.cos(angle), -math.sin(angle), 0)

            frames.append(self._build_frame_tn(pos, tangent, normal))
        return frames

    def _build_frame(self, pos: Vec3, normal: Vec3) -> Mat4:
        """Build an orthonormal frame from position and normal (Darboux-like)."""
        z = normal.normalized()
        up = Vec3(0, 1, 0)
        if abs(z.dot(up)) > 0.99:
            up = Vec3(1, 0, 0)
        x = up.cross(z).normalized()
        y = z.cross(x)
        return self._compose(x, y, z, pos)

    def _build_frame_tn(self, pos: Vec3, tangent: Vec3, normal: Vec3) -> Mat4:
        """Build frame from tangent and normal (Frenet-like)."""
        y = tangent.normalized()
        z = normal.normalized()
        x = y.cross(z).normalized()
        z = x.cross(y)  # Ensure orthogonality
        return self._compose(x, y, z, pos)

    def _compose(self, x: Vec3, y: Vec3, z: Vec3, pos: Vec3) -> Mat4:
        """Compose orthonormal axes + position into Mat4."""
        m = Mat4.identity()
        m.set_row(0, x)
        m.set_row(1, y)
        m.set_row(2, z)
        m.m[3][0] = pos.x
        m.m[3][1] = pos.y
        m.m[3][2] = pos.z
        return m


# ===========================================================================
# Phase 6 — Connect Node
# ===========================================================================

class ConnectNode(Node):
    """Declare spread connectivity topology."""

    def __init__(self, strategy: str = 'SEQUENTIAL', nx: int = 0, ny: int = 0):
        super().__init__('Connect')
        self.strategy = strategy
        self.nx = nx
        self.ny = ny

    def exec_self(self, state: WalkerState):
        state.spread_topology = self.strategy
        if self.strategy == 'GRID':
            state.grid_nx = self.nx
            state.grid_ny = self.ny


# ===========================================================================
# Phase 6 — Collapse Node
# ===========================================================================

class CollapseNode(Node):
    """Consume spread, emit geometry, return to single mode."""

    def __init__(self, collapse_type: str = 'INSTANCES', mesh_ref: str = '',
                 radius: float = 10.0, segments: int = 8):
        super().__init__('Collapse')
        self.collapse_type = collapse_type
        self.mesh_ref = mesh_ref
        self.radius   = radius
        self.segments = segments

    def execute(self, state: WalkerState):
        ct = self.collapse_type.upper()

        if ct == 'INSTANCES':
            self._collapse_instances(state)
        elif ct == 'TUBE':
            self._collapse_tube(state)
        else:
            # Points, Polyline, Ribbon — emit metadata
            if state.is_spread():
                for i, M in enumerate(state.spread_buffer):
                    state.spread_index = i
                    state.push()
                    state.transform(M)
                    state.emit('__topology__', {
                        'collapse_type': ct.lower(),
                        'spread_index': str(i),
                        'spread_count': str(len(state.spread_buffer)),
                    })
                    state.pop()
                state.exit_spread()

        self.execute_children(state)

    def _collapse_instances(self, state: WalkerState):
        if not state.is_spread():
            return
        for i, M in enumerate(state.spread_buffer):
            state.spread_index = i
            state.push()
            state.transform(M)
            state.emit(self.mesh_ref)
            state.pop()
        state.exit_spread()

    def _collapse_tube(self, state: WalkerState):
        if not state.is_spread():
            return
        tags = {
            'collapse_type': 'tube',
            'radius': str(self.radius),
            'segments': str(self.segments),
            'ring_count': str(len(state.spread_buffer)),
            'topology': 'closed' if state.spread_topology == 'CLOSED' else 'open',
        }
        state.emit('__tube__', tags)
        state.exit_spread()


# ===========================================================================
# Phase 6B: Frame Node — Topology → Matrices
# Reference: core/extended_state_algebra.md Section 1
# ===========================================================================

class FrameNode(Node):
    """
    Lift topological features into coordinate frames → Spread.

    Query syntax:
      "vertex[*]"    - All vertices
      "vertex[0:10]" - Vertices 0-9
      "edge[*]"      - All edges (frame at midpoint)
      "face[*]"      - All faces (frame at centroid)
      "curve[n=100]" - 100 samples along curve

    Frame modes:
      DARBOUX   - Surface: N=normal, T=principal curvature, B=T×N
      FRENET    - Curve: T=tangent, N=curvature, B=T×N
      PARALLEL  - Curve with parallel transport (twist-free)
      VERTEX    - Use vertex normal directly

    NOTE: In this reference implementation, geometry resolution is simulated.
    Real implementations must wire to actual mesh queries.
    """

    def __init__(self, geo_ref: str, query: str, mode: str = 'DARBOUX'):
        super().__init__('Frame')
        self.geo_ref = geo_ref
        self.query = query
        self.mode = mode.upper()

    def execute(self, state: WalkerState):
        # Parse query to determine component type and count/range
        frames = self._query_to_frames(state)

        if not frames:
            return

        # Save previous spread state for nesting
        saved = self._save_spread_state(state)

        # Enter spread mode with computed frames
        state.enter_spread(frames)

        # Determine topology from query
        if 'curve' in self.query.lower() or 'edge' in self.query.lower():
            state.spread_topology = 'SEQUENTIAL'
        elif 'face' in self.query.lower():
            state.spread_topology = 'NONE'
        else:
            state.spread_topology = 'NONE'

        # Execute children once per frame
        for i, frame in enumerate(frames):
            state.spread_index = i
            state.push()
            state.transform(frame)
            self.execute_children(state)
            state.pop()

        # Restore spread state
        self._restore_spread_state(state, saved)

    def _save_spread_state(self, state: WalkerState) -> dict:
        return {
            'mode': state.context_mode,
            'buffer': state.spread_buffer.copy() if state.spread_buffer else [],
            'index': state.spread_index,
            'topology': state.spread_topology,
            'grid_nx': state.grid_nx,
            'grid_ny': state.grid_ny,
        }

    def _restore_spread_state(self, state: WalkerState, saved: dict):
        if saved['mode'] == 'SPREAD':
            state.context_mode = 'SPREAD'
            state.spread_buffer = saved['buffer']
            state.spread_index = saved['index']
            state.spread_topology = saved['topology']
            state.grid_nx = saved['grid_nx']
            state.grid_ny = saved['grid_ny']
        else:
            state.exit_spread()

    def _query_to_frames(self, state: WalkerState) -> List[Mat4]:
        """
        Parse query string and generate frames.

        In a real implementation, this would query actual mesh topology.
        Here we simulate with predictable patterns for testing.
        """
        query = self.query.lower()
        frames = []

        # Parse count from query (e.g., "vertex[0:10]", "curve[n=50]")
        count = self._parse_count(query)

        if 'vertex' in query:
            frames = self._generate_vertex_frames(count)
        elif 'edge' in query:
            frames = self._generate_edge_frames(count)
        elif 'face' in query:
            frames = self._generate_face_frames(count)
        elif 'curve' in query:
            frames = self._generate_curve_frames(count)
        else:
            # Default: single frame at origin
            frames = [Mat4.identity()]

        return frames

    def _parse_count(self, query: str) -> int:
        """Extract count from query like 'vertex[0:10]' or 'curve[n=50]'."""
        import re

        # Match [0:N] pattern
        match = re.search(r'\[(\d+):(\d+)\]', query)
        if match:
            return int(match.group(2)) - int(match.group(1))

        # Match [n=N] pattern
        match = re.search(r'\[n=(\d+)\]', query)
        if match:
            return int(match.group(1))

        # Match [*] pattern — default to 8
        if '[*]' in query:
            return 8

        return 4  # Default

    def _generate_vertex_frames(self, count: int) -> List[Mat4]:
        """Simulate vertex frames on a unit sphere."""
        frames = []
        for i in range(count):
            # Distribute on sphere using golden angle
            phi = math.acos(1 - 2 * (i + 0.5) / count)
            theta = math.pi * (1 + math.sqrt(5)) * i

            x = math.sin(phi) * math.cos(theta)
            y = math.sin(phi) * math.sin(theta)
            z = math.cos(phi)

            pos = Vec3(x * 100, y * 100, z * 100)
            normal = Vec3(x, y, z)
            frames.append(self._compute_frame(pos, normal))

        return frames

    def _generate_edge_frames(self, count: int) -> List[Mat4]:
        """Simulate edge loop frames (circle in XY plane)."""
        frames = []
        for i in range(count):
            angle = 2 * math.pi * i / count
            x = math.cos(angle) * 100
            y = math.sin(angle) * 100
            pos = Vec3(x, y, 0)
            normal = Vec3(0, 0, 1)
            tangent = Vec3(-math.sin(angle), math.cos(angle), 0)
            frames.append(self._compute_frame_with_tangent(pos, normal, tangent))

        return frames

    def _generate_face_frames(self, count: int) -> List[Mat4]:
        """Simulate face frames on a subdivided plane."""
        frames = []
        side = int(math.sqrt(count)) or 2
        spacing = 100.0

        for i in range(side):
            for j in range(side):
                if len(frames) >= count:
                    break
                pos = Vec3(i * spacing, j * spacing, 0)
                normal = Vec3(0, 0, 1)
                frames.append(self._compute_frame(pos, normal))

        return frames[:count]

    def _generate_curve_frames(self, count: int) -> List[Mat4]:
        """Simulate Frenet frames along a helix curve."""
        frames = []
        for i in range(count):
            t = i / max(count - 1, 1)
            angle = t * 4 * math.pi  # Two full turns

            # Helix position
            x = math.cos(angle) * 50
            y = math.sin(angle) * 50
            z = t * 200
            pos = Vec3(x, y, z)

            # Tangent (derivative of helix)
            tx = -math.sin(angle)
            ty = math.cos(angle)
            tz = 200 / (4 * math.pi * max(count - 1, 1))
            tangent = Vec3(tx, ty, tz).normalized()

            # Normal (toward center)
            normal = Vec3(-math.cos(angle), -math.sin(angle), 0)

            if self.mode == 'FRENET':
                frames.append(self._compute_frenet_frame(pos, tangent, normal))
            else:
                frames.append(self._compute_frame_with_tangent(pos, normal, tangent))

        return frames

    def _compute_frame(self, pos: Vec3, normal: Vec3) -> Mat4:
        """Compute Darboux-style frame from position and normal."""
        from math_core import align_to_surface
        return align_to_surface(pos, normal)

    def _compute_frame_with_tangent(self, pos: Vec3, normal: Vec3,
                                     tangent: Vec3) -> Mat4:
        """Compute frame with explicit tangent direction."""
        z = normal.normalized()
        x = tangent.normalized()
        y = z.cross(x)

        M = Mat4.identity()
        M.set_row(0, x)
        M.set_row(1, y)
        M.set_row(2, z)
        M.m[3][0] = pos.x
        M.m[3][1] = pos.y
        M.m[3][2] = pos.z
        return M

    def _compute_frenet_frame(self, pos: Vec3, tangent: Vec3,
                               normal: Vec3) -> Mat4:
        """Compute Frenet-Serret frame: T=tangent, N=normal, B=T×N."""
        T = tangent.normalized()
        N = normal.normalized()
        B = T.cross(N)

        M = Mat4.identity()
        M.set_row(0, B)      # X = binormal
        M.set_row(1, T)      # Y = tangent (up along curve)
        M.set_row(2, N)      # Z = normal
        M.m[3][0] = pos.x
        M.m[3][1] = pos.y
        M.m[3][2] = pos.z
        return M


# ===========================================================================
# Phase 6D: Loft Node — Surface from Spread
# Reference: core/extended_state_algebra.md Section 4
# ===========================================================================

class LoftNode(Node):
    """
    Consume a spread of frames and emit a lofted surface.

    Loft modes:
      SEQUENTIAL - Sweep profile along spread (tube-like)
      GRID       - Create surface from NxM grid of frames

    The spread must have connectivity set (via ConnectNode) to determine
    how frames are stitched into a surface.
    """

    def __init__(self, profile: str = 'circle', radius: float = 10.0,
                 segments: int = 8, smooth: bool = True, caps: bool = True):
        super().__init__('Loft')
        self.profile = profile
        self.radius = radius
        self.segments = segments
        self.smooth = smooth
        self.caps = caps

    def execute(self, state: WalkerState):
        if not state.is_spread():
            # Silently skip — Loft is only valid in SPREAD mode.
            # This happens when Loft is a child that gets called after
            # the spread has already been consumed.
            return

        frames = state.spread_buffer.copy()
        topology = state.spread_topology
        nx = state.grid_nx
        ny = state.grid_ny

        if topology == 'GRID' and nx > 0 and ny > 0:
            self._loft_grid(state, frames, nx, ny)
        else:
            self._loft_sequential(state, frames, topology == 'CLOSED')

        state.exit_spread()
        self.execute_children(state)

    def _loft_sequential(self, state: WalkerState, frames: List[Mat4],
                          closed: bool):
        """
        Sweep a profile along the frame sequence.

        Emits metadata for downstream mesh generation. The actual mesh
        generation is platform-specific (UE5 uses ProceduralMesh).
        """
        # Serialize frame positions for mesh generator
        ring_data = []
        for i, M in enumerate(frames):
            pos = M.get_translation()
            ring_data.append(f"{pos.x:.2f},{pos.y:.2f},{pos.z:.2f}")

        tags = {
            'loft_type': 'sequential',
            'profile': self.profile,
            'radius': str(self.radius),
            'segments': str(self.segments),
            'ring_count': str(len(frames)),
            'rings': '|'.join(ring_data),
            'closed': 'true' if closed else 'false',
            'caps': 'true' if self.caps else 'false',
            'smooth': 'true' if self.smooth else 'false',
        }
        state.emit('__loft__', tags)

    def _loft_grid(self, state: WalkerState, frames: List[Mat4],
                    nx: int, ny: int):
        """
        Create a surface from a grid of frames.

        Frames are expected to be ordered row-by-row:
        [row0_col0, row0_col1, ..., row1_col0, row1_col1, ...]
        """
        if len(frames) < nx * ny:
            print(f"Loft grid requires {nx * ny} frames, got {len(frames)}")
            return

        # Serialize grid positions
        grid_data = []
        for i, M in enumerate(frames[:nx * ny]):
            pos = M.get_translation()
            grid_data.append(f"{pos.x:.2f},{pos.y:.2f},{pos.z:.2f}")

        tags = {
            'loft_type': 'grid',
            'nx': str(nx),
            'ny': str(ny),
            'grid_positions': '|'.join(grid_data),
            'smooth': 'true' if self.smooth else 'false',
        }
        state.emit('__loft__', tags)


# ===========================================================================
# Root Node
# ===========================================================================

class RootNode(Node):
    def __init__(self):
        super().__init__('Root')


# ===========================================================================
# JSON → Node Tree Parser
# ===========================================================================

def _read_vec3(obj: dict, key: str, default=(0, 0, 0)) -> Vec3:
    v = obj.get(key, default)
    if isinstance(v, (list, tuple)) and len(v) >= 3:
        return Vec3(v[0], v[1], v[2])
    return Vec3(*default)


def parse_node(obj: dict) -> Optional[Node]:
    """Parse a single JSON node object into a Node."""
    node_type = obj.get('type', '')

    if node_type in ('T', 'Translate'):
        return TranslateNode(_read_vec3(obj, 'v'))

    elif node_type in ('R', 'Rotate'):
        return RotateNode(obj.get('axis', 'Z'), obj.get('deg', 0))

    elif node_type in ('S', 'Scale'):
        return ScaleNode(_read_vec3(obj, 'v', (1, 1, 1)))

    elif node_type == 'Scope':
        return ScopeNode()

    elif node_type == 'Platform':
        pid = obj.get('id', 'P0')
        from_str = obj.get('from', 'Current')
        return PlatformNode(pid, from_str in ('Selection', 'Current'))

    elif node_type == 'Mirror':
        return MirrorNode(obj.get('axis', 'X'))

    elif node_type == 'Radial':
        return RadialNode(obj.get('n', 4), obj.get('axis', 'Z'))

    elif node_type == 'Instance':
        return InstanceNode(obj.get('mesh', ''))

    elif node_type == 'Spread':
        source = obj.get('source', 'Linear')
        params = obj.get('params', {})

        # Convert list params to Vec3 where needed
        for k in ('start', 'end', 'spacing'):
            if k in params and isinstance(params[k], (list, tuple)):
                params[k] = Vec3(*params[k])

        return SpreadNode(source, **params)

    elif node_type == 'Field':
        params = obj.get('params', {})
        target = params.get('target', [0, 0, 0])
        if isinstance(target, (list, tuple)):
            target = Vec3(*target)
        return FieldNode(
            field_type=obj.get('fieldType', params.get('type', 'Noise')),
            operation=obj.get('operation', obj.get('op', 'ADVECT')),
            intensity=params.get('strength', obj.get('intensity', 1.0)),
            scale=params.get('scale', obj.get('scale', 0.01)),
            target=target,
        )

    elif node_type == 'Connect':
        strategy = obj.get('strategy', 'SEQUENTIAL')
        params = obj.get('params', {})
        return ConnectNode(strategy, params.get('nx', 0), params.get('ny', 0))

    elif node_type == 'Collapse':
        return CollapseNode(
            collapse_type=obj.get('collapseType', 'INSTANCES'),
            mesh_ref=obj.get('mesh', ''),
            radius=obj.get('radius', 10.0),
            segments=obj.get('segments', 8),
        )

    elif node_type == 'Frame':
        return FrameNode(
            geo_ref=obj.get('geo', obj.get('geo_ref', '')),
            query=obj.get('query', 'vertex[*]'),
            mode=obj.get('mode', 'DARBOUX'),
        )

    elif node_type == 'Loft':
        return LoftNode(
            profile=obj.get('profile', 'circle'),
            radius=obj.get('radius', 10.0),
            segments=obj.get('segments', 8),
            smooth=obj.get('smooth', True),
            caps=obj.get('caps', True),
        )

    else:
        print(f"Unknown node type: {node_type}")
        return None


def _attach_children(obj: dict, node: Node):
    """Recursively parse and attach children."""
    children = obj.get('children', [])
    for child_obj in children:
        child = parse_node(child_obj)
        if child:
            node.add_child(child)
            _attach_children(child_obj, child)


def parse_tree(json_str: str) -> Optional[Node]:
    """Parse a full JSON SST tree into a Root node."""
    data = json.loads(json_str)

    root_list = data.get('root')
    if root_list is None:
        tree = data.get('tree', {})
        root_list = tree.get('root', [])

    if not root_list:
        return None

    root = RootNode()
    for item in root_list:
        child = parse_node(item)
        if child:
            root.add_child(child)
            _attach_children(item, child)

    return root


def execute_tree(json_str: str) -> List[Dict]:
    """Parse and execute a JSON SST tree, returning the emission buffer."""
    root = parse_tree(json_str)
    if root is None:
        return []
    state = WalkerState()
    root.execute(state)
    return state.buffer


# ===========================================================================
# Self-tests
# ===========================================================================

if __name__ == '__main__':
    print("--- SST Node System Tests ---")

    # Test 1: Simple translate + instance
    tree1 = json.dumps({
        "root": [
            {"type": "T", "v": [100, 0, 0]},
            {"type": "Instance", "mesh": "/Game/Cube"}
        ]
    })
    buf = execute_tree(tree1)
    assert len(buf) == 1, f"Test 1: expected 1 emission, got {len(buf)}"
    pos = buf[0]['transform'].get_translation()
    assert abs(pos.x - 100) < 0.01, f"Test 1: x={pos.x}, expected 100"
    print("  [PASS] Test 1: Translate + Instance")

    # Test 2: Platform + Mirror (M_reflect = P x S x P^-1)
    tree2 = json.dumps({
        "root": [
            {"type": "Platform", "id": "P1"},
            {"type": "Mirror", "axis": "X", "children": [
                {"type": "T", "v": [50, 0, 0]},
                {"type": "Instance", "mesh": "/Game/Arm"}
            ]}
        ]
    })
    buf = execute_tree(tree2)
    assert len(buf) == 2, f"Test 2: expected 2 emissions (real+reflected), got {len(buf)}"
    p_real = buf[0]['transform'].get_translation()
    p_refl = buf[1]['transform'].get_translation()
    assert abs(p_real.x - 50) < 0.01, f"Test 2: real x={p_real.x}"
    assert abs(p_refl.x - (-50)) < 0.01, f"Test 2: reflected x={p_refl.x}"
    assert buf[0]['sym_depth'] == 0
    assert buf[1]['sym_depth'] == 1
    print("  [PASS] Test 2: Platform + Mirror (P x S x P^-1)")

    # Test 3: Radial(4) should produce 4 instances
    tree3 = json.dumps({
        "root": [
            {"type": "Radial", "n": 4, "axis": "Z", "children": [
                {"type": "T", "v": [100, 0, 0]},
                {"type": "Instance", "mesh": "/Game/Pillar"}
            ]}
        ]
    })
    buf = execute_tree(tree3)
    assert len(buf) == 4, f"Test 3: expected 4, got {len(buf)}"
    # First should be at (100, 0, 0), second at roughly (0, 100, 0)
    assert abs(buf[0]['transform'].get_translation().x - 100) < 0.1
    print("  [PASS] Test 3: Radial(4)")

    # Test 4: Spread(Linear) + Collapse(INSTANCES)
    tree4 = json.dumps({
        "root": [
            {"type": "Spread", "source": "Linear",
             "params": {"start": [0, 0, 0], "end": [400, 0, 0], "n": 5},
             "children": [
                {"type": "Collapse", "collapseType": "INSTANCES",
                 "mesh": "/Game/Fence"}
            ]}
        ]
    })
    buf = execute_tree(tree4)
    assert len(buf) == 5, f"Test 4: expected 5, got {len(buf)}"
    xs = [e['transform'].get_translation().x for e in buf]
    assert abs(xs[0]) < 0.01
    assert abs(xs[-1] - 400) < 0.01
    print("  [PASS] Test 4: Spread(Linear, n=5) + Collapse(INSTANCES)")

    # Test 5: Spread(Grid) + Collapse
    tree5 = json.dumps({
        "root": [
            {"type": "Spread", "source": "Grid",
             "params": {"nx": 3, "ny": 3, "spacing": [100, 100, 0]},
             "children": [
                {"type": "Collapse", "collapseType": "INSTANCES",
                 "mesh": "/Game/Tile"}
            ]}
        ]
    })
    buf = execute_tree(tree5)
    assert len(buf) == 9, f"Test 5: expected 9, got {len(buf)}"
    print("  [PASS] Test 5: Spread(Grid, 3x3) -> 9 instances")

    # Test 6: Nested spread (Linear inside Radial) = multiplicative
    tree6 = json.dumps({
        "root": [
            {"type": "Spread", "source": "Linear",
             "params": {"start": [0, 0, 0], "end": [200, 0, 0], "n": 3},
             "children": [
                {"type": "Spread", "source": "Radial",
                 "params": {"n": 4, "radius": 50, "axis": "Z"},
                 "children": [
                    {"type": "Collapse", "collapseType": "INSTANCES",
                     "mesh": "/Game/Bolt"}
                ]}
            ]}
        ]
    })
    buf = execute_tree(tree6)
    assert len(buf) == 12, f"Test 6: expected 3*4=12, got {len(buf)}"
    print("  [PASS] Test 6: Nested Spread (3 x 4) = 12 instances")

    # Test 7: Mirror + Spread (mirrored array)
    tree7 = json.dumps({
        "root": [
            {"type": "Platform", "id": "P1"},
            {"type": "Mirror", "axis": "X", "children": [
                {"type": "Spread", "source": "Linear",
                 "params": {"start": [50, 0, 0], "end": [200, 0, 0], "n": 3},
                 "children": [
                    {"type": "Collapse", "collapseType": "INSTANCES",
                     "mesh": "/Game/Rail"}
                ]}
            ]}
        ]
    })
    buf = execute_tree(tree7)
    assert len(buf) == 6, f"Test 7: expected 6 (3 real + 3 mirrored), got {len(buf)}"
    real_xs = [e['transform'].get_translation().x for e in buf if e['sym_depth'] == 0]
    mirror_xs = [e['transform'].get_translation().x for e in buf if e['sym_depth'] == 1]
    assert all(x > 0 for x in real_xs), "Test 7: real should be positive X"
    assert all(x < 0 for x in mirror_xs), "Test 7: mirrored should be negative X"
    print("  [PASS] Test 7: Mirror + Spread = mirrored array")

    # Test 8: Collapse(TUBE) emits tube metadata
    tree8 = json.dumps({
        "root": [
            {"type": "Spread", "source": "Linear",
             "params": {"start": [0, 0, 0], "end": [0, 0, 500], "n": 10},
             "children": [
                {"type": "Connect", "strategy": "SEQUENTIAL"},
                {"type": "Collapse", "collapseType": "TUBE",
                 "radius": 20, "segments": 12}
            ]}
        ]
    })
    buf = execute_tree(tree8)
    assert len(buf) == 1, f"Test 8: expected 1 tube record, got {len(buf)}"
    assert buf[0]['mesh'] == '__tube__'
    assert buf[0]['tags']['collapse_type'] == 'tube'
    assert buf[0]['tags']['ring_count'] == '10'
    print("  [PASS] Test 8: Spread + Connect + Collapse(TUBE)")

    # Test 9: Field (noise advection) changes position
    tree9 = json.dumps({
        "root": [
            {"type": "T", "v": [100, 100, 100]},
            {"type": "Field", "fieldType": "Noise",
             "operation": "ADVECT",
             "params": {"scale": 1.0, "strength": 50.0},
             "children": [
                {"type": "Instance", "mesh": "/Game/Rock"}
            ]}
        ]
    })
    buf = execute_tree(tree9)
    assert len(buf) == 1
    pos = buf[0]['transform'].get_translation()
    # Position should be offset from (100,100,100) by noise
    assert not (abs(pos.x - 100) < 0.001 and abs(pos.y - 100) < 0.001), \
        "Test 9: Field should displace"
    print("  [PASS] Test 9: Field(Noise) displaces position")

    # Test 10: Scope isolation
    tree10 = json.dumps({
        "root": [
            {"type": "T", "v": [10, 0, 0]},
            {"type": "Scope", "children": [
                {"type": "T", "v": [0, 50, 0]},
                {"type": "Instance", "mesh": "/Game/Inner"}
            ]},
            {"type": "Instance", "mesh": "/Game/Outer"}
        ]
    })
    buf = execute_tree(tree10)
    assert len(buf) == 2
    inner_pos = buf[0]['transform'].get_translation()
    outer_pos = buf[1]['transform'].get_translation()
    assert abs(inner_pos.y - 50) < 0.01, f"Test 10: inner y={inner_pos.y}"
    assert abs(outer_pos.y) < 0.01, f"Test 10: outer y={outer_pos.y} (should be 0)"
    assert abs(outer_pos.x - 10) < 0.01, f"Test 10: outer x={outer_pos.x}"
    print("  [PASS] Test 10: Scope isolation")

    # Test 11: Frame node (simulated vertex frames)
    tree11 = json.dumps({
        "root": [
            {"type": "Frame", "geo": "mesh_001", "query": "vertex[0:4]",
             "mode": "DARBOUX", "children": [
                {"type": "S", "v": [0.5, 0.5, 0.5]},
                {"type": "Instance", "mesh": "/Game/Rivet"}
            ]}
        ]
    })
    buf = execute_tree(tree11)
    assert len(buf) == 4, f"Test 11: expected 4 frame instances, got {len(buf)}"
    # Each instance should be at a different position (sphere distribution)
    positions = [e['transform'].get_translation() for e in buf]
    unique_positions = len(set((round(p.x, 1), round(p.y, 1), round(p.z, 1))
                               for p in positions))
    assert unique_positions == 4, f"Test 11: expected 4 unique positions, got {unique_positions}"
    print("  [PASS] Test 11: Frame(vertex[0:4]) generates 4 unique frames")

    # Test 12: Frame node with curve (Frenet frames)
    tree12 = json.dumps({
        "root": [
            {"type": "Frame", "geo": "curve_001", "query": "curve[n=8]",
             "mode": "FRENET", "children": [
                {"type": "Collapse", "collapseType": "INSTANCES",
                 "mesh": "/Game/RingSegment"}
            ]}
        ]
    })
    buf = execute_tree(tree12)
    assert len(buf) == 8, f"Test 12: expected 8 curve samples, got {len(buf)}"
    # Frames should follow helix pattern (z increases)
    zs = [e['transform'].get_translation().z for e in buf]
    assert zs[-1] > zs[0], f"Test 12: z should increase along curve"
    print("  [PASS] Test 12: Frame(curve, FRENET) generates helix frames")

    # Test 13: Loft (sequential sweep)
    tree13 = json.dumps({
        "root": [
            {"type": "Spread", "source": "Linear",
             "params": {"start": [0, 0, 0], "end": [0, 0, 300], "n": 10},
             "children": [
                {"type": "Connect", "strategy": "SEQUENTIAL"},
                {"type": "Loft", "profile": "circle", "radius": 15,
                 "segments": 8, "caps": True}
            ]}
        ]
    })
    buf = execute_tree(tree13)
    assert len(buf) == 1, f"Test 13: expected 1 loft record, got {len(buf)}"
    assert buf[0]['mesh'] == '__loft__'
    assert buf[0]['tags']['loft_type'] == 'sequential'
    assert buf[0]['tags']['ring_count'] == '10'
    assert buf[0]['tags']['profile'] == 'circle'
    print("  [PASS] Test 13: Loft(sequential) emits surface metadata")

    # Test 14: Loft with grid topology
    tree14 = json.dumps({
        "root": [
            {"type": "Spread", "source": "Grid",
             "params": {"nx": 4, "ny": 5, "spacing": [50, 50, 0]},
             "children": [
                {"type": "Connect", "strategy": "GRID",
                 "params": {"nx": 4, "ny": 5}},
                {"type": "Loft", "smooth": True}
            ]}
        ]
    })
    buf = execute_tree(tree14)
    assert len(buf) == 1, f"Test 14: expected 1 loft record, got {len(buf)}"
    assert buf[0]['tags']['loft_type'] == 'grid'
    assert buf[0]['tags']['nx'] == '4'
    assert buf[0]['tags']['ny'] == '5'
    print("  [PASS] Test 14: Loft(grid) emits NxM surface metadata")

    print("\n--- All 14 SST node tests passed. ---")
