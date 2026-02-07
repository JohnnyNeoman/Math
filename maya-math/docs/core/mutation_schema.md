# Mutation Schema: Geometric Operations

> **The Substance Layer of Skeletal Singleton Tree**
> Formalizing the "What"

---

## Abstract

Mutation represents **geometric operations that produce or modify geometry**. Mutations are position-agnostic — they define *what* is created, not *where*. The State layer handles placement.

---

## Mutation Categories

```
┌─────────────────────────────────────────────────────────────────┐
│                    MUTATION TAXONOMY                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   PRIMITIVES          CONSTRUCTIVE         DEFORMATIONS         │
│   ┌─────────┐        ┌─────────┐          ┌─────────┐          │
│   │ □ Box   │        │ ⊕ Union │          │ ∿ Bend  │          │
│   │ ○ Sphere│        │ ⊖ Diff  │          │ ∿ Twist │          │
│   │ △ Cone  │        │ ⊗ Inter │          │ ∿ Taper │          │
│   │ ◇ Torus │        │ ↑ Extru │          │ ∿ Noise │          │
│   │ ⬡ Cylin │        │ ⟳ Revol │          │ ∿ FFD   │          │
│   └─────────┘        └─────────┘          └─────────┘          │
│        │                  │                    │                │
│        └──────────────────┴────────────────────┘                │
│                           │                                     │
│                    ┌──────▼──────┐                              │
│                    │    MESH     │                              │
│                    │  (Output)   │                              │
│                    └─────────────┘                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Primitive Operations (Σ_primitive)

### Box: □(w, h, d)

```
□(width, height, depth) → Mesh

Parameters:
    w: float > 0    # X extent
    h: float > 0    # Y extent  
    d: float > 0    # Z extent

Output: Axis-aligned box centered at origin
        Vertices: 8
        Faces: 6
        Bounds: [-w/2, w/2] × [-h/2, h/2] × [-d/2, d/2]
```

### Sphere: ○(r, segments)

```
○(radius, segments) → Mesh

Parameters:
    r: float > 0        # Radius
    segments: int ≥ 4   # Subdivision level

Output: UV sphere centered at origin
        Vertices: segments × (segments/2 + 1)
        Bounds: [-r, r]³
```

### Cone: △(r, h, segments)

```
△(radius, height, segments) → Mesh

Parameters:
    r: float > 0        # Base radius
    h: float > 0        # Height
    segments: int ≥ 3   # Base divisions

Output: Cone with base at origin, apex at (0, h, 0)
```

### Cylinder: ⬡(r, h, segments)

```
⬡(radius, height, segments) → Mesh

Parameters:
    r: float > 0        # Radius
    h: float > 0        # Height
    segments: int ≥ 3   # Circular divisions

Output: Cylinder centered at origin, extends ±h/2 on Y
```

### Torus: ◇(R, r, segments_major, segments_minor)

```
◇(major_radius, minor_radius, seg_major, seg_minor) → Mesh

Parameters:
    R: float > 0        # Distance from center to tube center
    r: float > 0        # Tube radius (r < R)
    seg_major: int ≥ 3  # Around the ring
    seg_minor: int ≥ 3  # Around the tube

Output: Torus in XZ plane, centered at origin
```

---

## Constructive Solid Geometry (Σ_csg)

### Union: ⊕(A, B)

```
⊕(mesh_a, mesh_b) → Mesh

Semantics: Combined volume of A and B
           Points in A OR B are in result

Properties:
    Commutative: ⊕(A, B) = ⊕(B, A)
    Associative: ⊕(⊕(A, B), C) = ⊕(A, ⊕(B, C))
    Identity: ⊕(A, ∅) = A
```

### Difference: ⊖(A, B)

```
⊖(mesh_a, mesh_b) → Mesh

Semantics: A with B carved out
           Points in A AND NOT in B are in result

Properties:
    NOT Commutative: ⊖(A, B) ≠ ⊖(B, A)
    ⊖(A, ∅) = A
    ⊖(A, A) = ∅
```

### Intersection: ⊗(A, B)

```
⊗(mesh_a, mesh_b) → Mesh

Semantics: Overlapping volume only
           Points in A AND B are in result

Properties:
    Commutative: ⊗(A, B) = ⊗(B, A)
    Associative: ⊗(⊗(A, B), C) = ⊗(A, ⊗(B, C))
    ⊗(A, ∅) = ∅
```

---

## Generative Operations (Σ_generative)

### Extrude: ↑(profile, path | distance)

```
↑(profile_curve, path_or_distance) → Mesh

Variants:
    ↑(profile, distance: float)    # Linear extrusion along local Y
    ↑(profile, path: Curve)        # Sweep along arbitrary path

Parameters:
    profile: Curve (closed 2D shape)
    path: Curve | float

Output: Surface swept from profile
```

### Revolve: ⟳(profile, axis, angle)

```
⟳(profile, axis, angle) → Mesh

Parameters:
    profile: Curve (open or closed 2D shape)
    axis: Vector3 (revolution axis)
    angle: float (degrees, default 360)

Output: Surface of revolution
```

---

## Deformation Operations (Σ_deform)

### General Deform: ∿(mesh, function)

```
∿(mesh, deform_func) → Mesh

Parameters:
    mesh: Mesh (input geometry)
    deform_func: (Vector3 → Vector3) | DeformType

DeformType enum:
    BEND(axis, angle, bounds)
    TWIST(axis, angle, bounds)
    TAPER(axis, factor, bounds)
    NOISE(amplitude, frequency, seed)
    FFD(lattice_dims, control_points)

Output: Deformed mesh (same topology, modified positions)
```

---

## Mutation Node Schema (JSONL)

```jsonl
{"type":"mutation_op","id":"m001","op":"box","params":{"w":1,"h":2,"d":1},"result":"mesh"}
{"type":"mutation_op","id":"m002","op":"sphere","params":{"r":0.5,"segments":16},"result":"mesh"}
{"type":"mutation_op","id":"m003","op":"union","operands":["m001","m002"],"result":"mesh"}
{"type":"mutation_op","id":"m004","op":"deform","sub_op":"twist","params":{"axis":[0,1,0],"angle":45},"operand":"m003","result":"mesh"}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"mutation_op"` | Record type identifier |
| `id` | string | Stable unique ID |
| `op` | string | Operation category |
| `sub_op` | string | Specific operation (for deform) |
| `params` | object | Operation parameters |
| `operands` | string[] | Input geometry IDs (for CSG) |
| `operand` | string | Single input geometry ID |
| `result` | `"mesh"\|"curve"\|"points"` | Output type |

---

## Dependency Graph

Mutations can reference other mutations, forming a DAG:

```
m001: □(1,1,1)           ──┐
                           ├──▶ m003: ⊕(m001, m002) ──▶ m004: ∿(m003, twist)
m002: ○(0.5, 16)         ──┘

Evaluation order: m001, m002 → m003 → m004
```

---

## Platform Mapping

| Mutation | Maya | Blender | Unreal |
|----------|------|---------|--------|
| `□` box | `polyCube(w=w,h=h,d=d)` | `primitive_cube_add(size=...)` | `ProceduralMeshComponent` |
| `○` sphere | `polySphere(r=r)` | `primitive_uv_sphere_add(radius=r)` | `ProceduralMeshComponent` |
| `⊕` union | `polyBoolOp(op=1,...)` | `modifier_add(type='BOOLEAN', operation='UNION')` | `UBooleanOperation` |
| `⊖` diff | `polyBoolOp(op=2,...)` | `modifier_add(type='BOOLEAN', operation='DIFFERENCE')` | `UBooleanOperation` |
| `↑` extrude | `polyExtrudeFacet(...)` | `bpy.ops.mesh.extrude_region_move()` | `UProceduralMesh` |
| `∿` deform | `nonLinear(type=...)` | `modifier_add(type='SIMPLE_DEFORM')` | `MeshDeformer` |

---

## Composition Rules

1. **Primitives are leaves**: They take only numeric parameters
2. **CSG requires meshes**: Operands must resolve to mesh type
3. **Deform preserves topology**: Vertex count unchanged
4. **Generative changes topology**: Extrude/revolve add vertices

---

*Mutation is the clay; State is the potter's hands.*
