# Extended State Algebra: Topological Frames, Spreads, and Fields

> **From Skeleton to Nervous System**
> Expanding Σ_state for Arrays, Topology, and Volumetric Forces
> Version 1.0 | 2026-01-27

---

## Motivation

The current SST algebra treats the Walker as a **single point** moving through space—a "turtle" carrying a matrix. This is sufficient for:
- Hierarchical transforms (parent-child)
- Bilateral/radial symmetry
- Instance placement

But it cannot express:
- **Arrays of vectors** treated as manifolds
- **Topological features** (edge rings, face loops) as coordinate systems
- **Volumetric fields** that influence placement
- **Lofting** across spreads of frames

To support these, we must expand from **Scalar Matrix Operations** to **Topological Frame Operations** and **Field-Based Transformations**.

---

## Conceptual Leap

```
CURRENT: Single Turtle
┌─────────────────────────────────────────────────────────────────┐
│   State = { M: Matrix4×4 }                                      │
│   Walker traverses tree, accumulating ONE transform             │
│   Emit produces ONE instance per node                           │
└─────────────────────────────────────────────────────────────────┘

EXPANDED: Herd of Turtles + Field Sensitivity
┌─────────────────────────────────────────────────────────────────┐
│   State = {                                                     │
│       mode: SINGLE | SPREAD,                                    │
│       M: Matrix4×4 | Matrix4×4[],    # Singleton OR array       │
│       topology_context: Geo?,         # Optional mesh reference │
│       field_stack: Field[]            # Active force fields     │
│   }                                                             │
│                                                                 │
│   Walker can fork into N parallel contexts                      │
│   Fields continuously perturb the matrix stack                  │
│   Topology provides frames from mesh features                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Topological Frame Binding: `Frame(geo, query, mode)`

### The Problem

`Align(point, normal)` works for a single surface hit. But what if you want to:
- Instance along every face of a mesh
- Array objects down an edge loop
- Place geometry at every vertex with proper tangent space

You need to **lift topological features into matrix frames**.

### The Darboux Frame

For a point on a surface, the **Darboux frame** provides:
- **N** (normal): Perpendicular to surface
- **T** (tangent): Along principal curvature direction
- **B** (binormal): T × N

For curves, the **Frenet-Serret frame** provides:
- **T** (tangent): Curve direction
- **N** (normal): Curvature direction
- **B** (binormal): T × N

### Operator Definition

```
Frame(geo_ref, query, mode) → Spread<Matrix4×4>

Parameters:
    geo_ref : string       # ID of a prior mutation node
    query   : string       # Topology query (see below)
    mode    : DARBOUX | FRENET | PARALLEL | VERTEX_NORMAL

Query Syntax:
    "vertex[*]"           # All vertices
    "vertex[0:10]"        # Vertices 0-9
    "edge[*]"             # All edges (frame at midpoint)
    "edge_loop[0]"        # First detected edge loop
    "face[*]"             # All faces (frame at centroid)
    "face_ring[uv=0.5]"   # Faces along UV coordinate
    "curve[samples=100]"  # 100 samples along curve

Output:
    Converts State from SINGLE to SPREAD mode
    Each component produces one frame in the spread
```

### Frame Computation by Type

| Component | Z (Normal) | Y (Up/Flow) | X (Right) |
|-----------|------------|-------------|-----------|
| Vertex | Vertex normal | Longest edge from vertex | Z × Y |
| Edge | Average of adjacent face normals | Edge direction | Z × Y |
| Face | Face normal | Longest edge of face | Z × Y |
| Curve Point | Frenet normal (or parallel transport) | Tangent | Z × Y |

### Node Implementation

```python
class FrameNode(Node):
    def __init__(self, geo_ref, query, mode='DARBOUX'):
        super().__init__('Frame')
        self.geo_ref = geo_ref
        self.query = query
        self.mode = mode
    
    def execute(self, state):
        # 1. Resolve geometry from prior mutation
        geo = state.resolve_geometry(self.geo_ref)
        
        # 2. Query topological components
        components = geo.query_topology(self.query)
        
        # 3. Compute frame for each component
        frames = [compute_frame(comp, self.mode) for comp in components]
        
        # 4. Fork state into SPREAD mode
        for frame in frames:
            state.push()
            state.transform(frame)
            self.execute_children(state)
            state.pop()
```

### YAML Syntax

```yaml
- Frame:
    geo: "hull_mesh"           # Reference to prior Instance/Mutation
    query: "face[*]"           # All faces
    mode: DARBOUX
  children:
    - S: { v: [0.1, 0.1, 0.1] }
    - Instance: { mesh: "/Game/Rivet" }
```

---

## 2. The Spread Operator: `Spread(matrices)`

### The Problem

The current algebra requires explicit nodes for each instance. To place 1000 objects along a curve, you'd need 1000 nodes. 

The **Spread** operator promotes the state from a single matrix to an **array of matrices**, and all subsequent children execute once per element.

### Algebraic Definition

```
Let S = current state matrix
Let [M₀, M₁, ..., Mₙ] = spread array

S × Spread([M₀, M₁, ..., Mₙ]) → [S·M₀, S·M₁, ..., S·Mₙ]

All children execute N times, once per spread element.
```

### State Schema Extension

```typescript
interface ExtendedState {
    // Existing
    stack: Matrix4x4[];
    platforms: Map<string, Matrix4x4>;
    active_platform: string | null;
    sym_depth: number;
    buffer: EmittedGeometry[];
    
    // NEW: Spread support
    mode: 'SINGLE' | 'SPREAD';
    spread_index: number;           // Current index in spread
    spread_count: number;           // Total spread size
    spread_buffer: Matrix4x4[];     // All matrices in spread
}
```

### Spread Sources

Spreads can be generated from:

| Source | Description | Example |
|--------|-------------|---------|
| `Range(n)` | N copies at origin | 100 instances |
| `Linear(start, end, n)` | N points along line | Array along axis |
| `Curve(curve_ref, n)` | N samples along curve | Spline array |
| `Grid(nx, ny, spacing)` | 2D grid of points | Floor tiles |
| `Radial(n, radius)` | N points in circle | Radial array |
| `Frame(geo, query)` | From topology | Instance on faces |
| `Scatter(geo, n, seed)` | Random on surface | Foliage |

### Node Implementation

```python
class SpreadNode(Node):
    def __init__(self, source, params):
        super().__init__('Spread')
        self.source = source
        self.params = params
    
    def execute(self, state):
        # 1. Generate spread matrices based on source
        matrices = self.generate_spread(state)
        
        # 2. Execute children once per matrix
        state.mode = 'SPREAD'
        state.spread_count = len(matrices)
        
        for i, M in enumerate(matrices):
            state.spread_index = i
            state.push()
            state.transform(M)
            self.execute_children(state)
            state.pop()
        
        state.mode = 'SINGLE'
    
    def generate_spread(self, state):
        if self.source == 'Linear':
            start = Vec3(*self.params['start'])
            end = Vec3(*self.params['end'])
            n = self.params['n']
            return [Mat4.Translate(lerp(start, end, i/(n-1))) for i in range(n)]
        
        elif self.source == 'Curve':
            curve = state.resolve_geometry(self.params['curve'])
            n = self.params['n']
            return [curve.frame_at(i/(n-1)) for i in range(n)]
        
        # ... other sources
```

### YAML Syntax

```yaml
# Linear array
- Spread:
    source: Linear
    params: { start: [0,0,0], end: [1000,0,0], n: 20 }
  children:
    - Instance: { mesh: "/Game/Pillar" }

# Curve array with tangent alignment
- Spread:
    source: Curve
    params: { curve: "rail_spline", n: 50, align: true }
  children:
    - Instance: { mesh: "/Game/TrainCar" }

# Scatter on surface
- Spread:
    source: Scatter
    params: { geo: "terrain", n: 500, seed: 42 }
  children:
    - R: { axis: Y, deg: "random(0, 360)" }
    - Instance: { mesh: "/Game/Tree" }
```

---

## 3. Field-Based Transformation: `Field(type, params, operation)`

### The Problem

Affine transforms (T, R, S) are **position-independent**—they apply the same way everywhere. But natural forms are shaped by **fields**:
- Gravity pulls downward
- Wind displaces laterally
- Attractors pull toward points
- Noise adds organic variation

You need operators that apply **spatially-varying** transformations.

### Field Types

| Type | Formula | Use Case |
|------|---------|----------|
| `Gravity` | F(p) = (0, -g, 0) | Drooping, settling |
| `Attractor` | F(p) = normalize(target - p) / |target - p|² | Magnetism, clustering |
| `Noise` | F(p) = perlin3D(p * scale) | Organic displacement |
| `Vortex` | F(p) = cross(axis, p - center) | Swirling, twisting |
| `Custom` | F(p) = user_function(p) | Arbitrary fields |

### Operations

| Operation | Effect |
|-----------|--------|
| `ADVECT` | Move position along field: P' = P + F(P) × strength |
| `ALIGN` | Rotate to align with field direction |
| `SCALE` | Scale based on field magnitude |
| `COMPOUND` | All of the above |

### Mathematical Formulation

**Advection** (position displacement):
```
P_new = P_old + ∫₀ᵗ F(P) dt

For discrete step:
P_new = P_old + F(P_old) × strength × dt
```

**Alignment** (rotation to field):
```
Given field direction V = F(P):
    Z_new = normalize(V)
    X_new = normalize(cross(Up, Z_new))
    Y_new = cross(Z_new, X_new)
    
M_align = [X_new | Y_new | Z_new | P]
```

### Node Implementation

```python
class FieldNode(Node):
    def __init__(self, field_type, params, operation='ADVECT'):
        super().__init__('Field')
        self.field_type = field_type
        self.params = params
        self.operation = operation
    
    def execute(self, state):
        # Get current position from state matrix
        P = state.M.translation()
        
        # Evaluate field at position
        F = self.evaluate_field(P)
        
        # Apply operation
        if self.operation == 'ADVECT':
            displacement = F * self.params.get('strength', 1.0)
            state.transform(Mat4.Translate(displacement))
        
        elif self.operation == 'ALIGN':
            if F.length() > 0.001:
                align_matrix = build_basis_from_direction(F)
                state.transform(align_matrix)
        
        elif self.operation == 'SCALE':
            magnitude = F.length()
            state.transform(Mat4.Scale(Vec3(magnitude, magnitude, magnitude)))
        
        self.execute_children(state)
    
    def evaluate_field(self, P):
        if self.field_type == 'Noise':
            scale = self.params.get('scale', 1.0)
            return Vec3(
                perlin(P.x * scale, P.y * scale, P.z * scale),
                perlin(P.x * scale + 100, P.y * scale, P.z * scale),
                perlin(P.x * scale, P.y * scale + 100, P.z * scale)
            )
        
        elif self.field_type == 'Attractor':
            target = Vec3(*self.params['target'])
            falloff = self.params.get('falloff', 2.0)
            direction = target - P
            distance = direction.length()
            if distance < 0.001:
                return Vec3(0, 0, 0)
            return direction.normalized() / (distance ** falloff)
        
        # ... other field types
```

### YAML Syntax

```yaml
# Noise displacement for organic variation
- Field:
    type: Noise
    params: { scale: 0.01, strength: 50.0 }
    operation: ADVECT
  children:
    - Instance: { mesh: "/Game/Rock" }

# Align to gravity (drooping branches)
- Field:
    type: Gravity
    params: { strength: 0.3 }
    operation: ALIGN
  children:
    - Instance: { mesh: "/Game/Branch" }

# Cluster toward attractor
- Spread:
    source: Scatter
    params: { geo: "ground", n: 100 }
  children:
    - Field:
        type: Attractor
        params: { target: [0, 0, 0], falloff: 1.5, strength: 20 }
        operation: ADVECT
      children:
        - Instance: { mesh: "/Game/Debris" }
```

---

## 4. Topology-Aware Connectivity: `Connect(strategy)`

### The Problem

When you have a Spread of matrices, they're independent. But geometry often requires **connectivity**:
- Points → Polyline
- Frames → Lofted surface
- Positions → Mesh vertices

The `Connect` operator declares relationships between spread elements.

### Connection Strategies

| Strategy | Result | Use Case |
|----------|--------|----------|
| `SEQUENTIAL` | M₀→M₁→M₂→...→Mₙ | Polyline, tube |
| `LOOP` | M₀→M₁→...→Mₙ→M₀ | Closed curve, ring |
| `PAIRWISE` | (M₀,M₁), (M₂,M₃), ... | Ribbons, bridges |
| `GRID(nx, ny)` | 2D connectivity | Mesh surface |
| `TRIANGULATE` | Delaunay triangulation | Organic surfaces |

### The Loft Pattern

Lofting is not just a mutation—it's a **State Aggregator** that consumes a Spread:

```
1. State Generation:   Curve.sample(100) → Spread<Matrix>[100]
2. State Perturbation: Field(Noise) → Offset frames
3. State Connection:   Connect(SEQUENTIAL) → Declare as polyline
4. State Collapse:     Loft(tube=true) → Emit tube geometry
```

### Node Implementation

```python
class ConnectNode(Node):
    def __init__(self, strategy, params=None):
        super().__init__('Connect')
        self.strategy = strategy
        self.params = params or {}
    
    def execute(self, state):
        if state.mode != 'SPREAD':
            raise Error("Connect requires SPREAD mode")
        
        # Mark connectivity in state
        state.connectivity = self.strategy
        state.connectivity_params = self.params
        
        # Children can now access connectivity info
        self.execute_children(state)
        
        # Clear connectivity after children
        state.connectivity = None
```

### LoftMutation (Topology-Aware)

```python
class LoftNode(Node):
    def __init__(self, params):
        super().__init__('Loft')
        self.params = params
    
    def execute(self, state):
        if state.mode != 'SPREAD':
            raise Error("Loft requires SPREAD mode")
        
        # Collect all matrices from spread
        frames = state.spread_buffer.copy()
        connectivity = state.connectivity or 'SEQUENTIAL'
        
        # Generate lofted geometry
        if connectivity == 'SEQUENTIAL':
            geo = self.loft_sequential(frames)
        elif connectivity == 'GRID':
            nx = state.connectivity_params.get('nx')
            ny = state.connectivity_params.get('ny')
            geo = self.loft_grid(frames, nx, ny)
        
        # Emit to buffer
        state.emit(geo, {'type': 'loft'})
```

### YAML Syntax

```yaml
# Tube along curve
- Spread:
    source: Curve
    params: { curve: "rail", n: 50 }
  children:
    - Connect: { strategy: SEQUENTIAL }
    - Loft:
        profile: "circle"
        radius: 10
        caps: true

# Grid surface from two curves
- Spread:
    source: LoftGrid
    params: { curve_a: "profile", curve_b: "rail", nu: 20, nv: 50 }
  children:
    - Connect: { strategy: GRID, params: { nx: 20, ny: 50 }}
    - Loft: { smooth: true }
```

---

## 5. Updated Alphabet Σ

### Σ_state (Extended)

```
Σ_state = {
    # Original
    T(v)              : Translate
    R(axis, θ)        : Rotate
    S(v)              : Scale
    [ ]               : Push/Pop scope
    Platform(P)       : Set portable world center
    Align(p, n)       : Gram-Schmidt surface adhesion
    
    # NEW: Topological
    Frame(geo, query, mode)  : Lift topology into frames → Spread
    
    # NEW: Array/Spread
    Spread(source, params)   : Generate array of matrices
    Connect(strategy)        : Declare spread connectivity
    
    # NEW: Field
    Field(type, params, op)  : Spatially-varying transformation
}
```

### Σ_mutation (Extended)

```
Σ_mutation = {
    # Original
    Instance(mesh)       : Emit mesh reference
    □ ○ △ ⬡             : Primitives
    ⊕ ⊖ ⊗               : CSG
    
    # NEW: Spread-aware
    Loft(profile, params)    : Consume spread → surface
    Sweep(profile, params)   : Consume spread → swept surface
    Connect(points)          : Consume spread → polyline/curve
    Mesh(params)             : Consume spread → mesh vertices
}
```

---

## 6. Execution Model Update

### Mode Transitions

```
SINGLE mode:
    State.M is a single Matrix4×4
    Children execute once
    
SPREAD mode:
    State.spread_buffer is Matrix4×4[]
    Children execute N times (once per element)
    Can be consumed by Loft/Sweep/Mesh

Transitions:
    SINGLE → SPREAD: Frame(), Spread()
    SPREAD → SINGLE: Loft(), Sweep(), Mesh() (collapse)
    SPREAD → SPREAD: Field(), Transform (applies to all)
```

### Nesting Spreads

Spreads can nest, creating multiplicative instances:

```yaml
- Spread:
    source: Linear
    params: { start: [0,0,0], end: [1000,0,0], n: 10 }  # 10 positions
  children:
    - Spread:
        source: Radial
        params: { n: 8, radius: 50 }  # 8 around each
      children:
        - Instance: { mesh: "/Game/Bolt" }
        # Total: 10 × 8 = 80 instances
```

---

## 7. Implementation Priority

### Phase 1: Spread Foundation (Implement First)
- [ ] Extend State with `mode`, `spread_buffer`, `spread_index`
- [ ] Implement `SpreadNode` with Linear, Radial, Grid sources
- [ ] Modify Walker to iterate spread in execute loop
- [ ] Test: Linear array of cubes

### Phase 2: Frame Binding
- [ ] Implement topology query parser
- [ ] Implement frame computation (vertex, edge, face)
- [ ] Implement `FrameNode`
- [ ] Test: Instance on all faces of a cube

### Phase 3: Field Transformations
- [ ] Implement noise field (Perlin 3D)
- [ ] Implement attractor field
- [ ] Implement `FieldNode` with ADVECT, ALIGN
- [ ] Test: Noise-displaced scatter

### Phase 4: Topology-Aware Mutations
- [ ] Implement `ConnectNode`
- [ ] Implement `LoftNode` (sequential profile sweep)
- [ ] Test: Tube along curve

---

## 8. Integration with Existing SST

The extensions are **additive**—they don't break existing functionality:

```
Existing:
    Mirror, Radial, Instance, Transform nodes work unchanged in SINGLE mode

Extended:
    Frame/Spread nodes transition to SPREAD mode
    All children of a Spread node execute N times
    Loft/Sweep nodes collapse SPREAD back to SINGLE
    Fields apply point-wise to current position

Composition:
    Mirror + Spread = Mirrored array
    Platform + Field = Local field in platform space
    Spread + Field + Loft = Noise-perturbed swept surface
```

---

## 9. Example: Organic Tendril

```yaml
root:
  - Platform: { id: "base" }
  - Mirror: { axis: X }
    children:
      # Generate spine curve samples
      - Spread:
          source: Curve
          params: { curve: "tendril_spine", n: 30 }
        children:
          # Add organic noise
          - Field:
              type: Noise
              params: { scale: 0.02, strength: 15 }
              operation: ADVECT
          
          # Twist along length
          - R: { axis: Y, deg: "spread_index * 12" }
          
          # Scale down toward tip
          - S: { v: "lerp([1,1,1], [0.2,0.2,0.2], spread_index/spread_count)" }
          
          # Mark as connected
          - Connect: { strategy: SEQUENTIAL }
          
          # Loft into tube
          - Loft:
              profile: "circle"
              radius: 5
              segments: 8
```

This produces a mirrored pair of organic, twisting tendrils that taper toward their tips.

---

## 10. Summary

| Extension | What It Enables |
|-----------|-----------------|
| `Frame(geo, query)` | Instance relative to mesh topology |
| `Spread(source)` | Arrays without explicit nodes |
| `Field(type, op)` | Organic, force-influenced placement |
| `Connect(strategy)` | Declare spread relationships |
| `Loft(profile)` | Collapse spread into surface |

The architecture moves from **Skeleton** (rigid hierarchy) to **Nervous System** (topological awareness + field sensitivity), while preserving backward compatibility with the existing node algebra.

---

*The turtle becomes a school of fish, swimming through force fields, leaving geometry in their wake.*
