# Skeletal Singleton Tree

> **The Program Map: Separating State from Mutation in 3D Assembly**
> Version 1.0 | Functional Parametric L-System Architecture

---

## Core Principle

```
┌─────────────────────────────────────────────────────────────────┐
│              SKELETAL SINGLETON TREE (SST)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌───────────────────┐         ┌───────────────────┐          │
│   │      STATE        │         │     MUTATION      │          │
│   │  (Matrix Stack)   │────────▶│    (Geometry)     │          │
│   │                   │         │                   │          │
│   │  • Transform ctx  │         │  • Primitives     │          │
│   │  • Scope stack    │         │  • Booleans       │          │
│   │  • Parameters     │         │  • Deformations   │          │
│   └───────────────────┘         └───────────────────┘          │
│            │                              │                     │
│            │         ┌────────────────────┤                     │
│            ▼         ▼                    ▼                     │
│   ┌─────────────────────────────────────────────────────┐      │
│   │              EVALUATION CONTEXT                      │      │
│   │   SST[node] → (State × Mutation) → Geometry          │      │
│   └─────────────────────────────────────────────────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Separation Theorem

**State** is *where* and *how* (context)
**Mutation** is *what* (substance)

```
Geometry = State(Mutation)
         = MatrixStack.apply(GeometricOperation)
```

This separation enables:
1. **Lazy evaluation** - State computed only when Mutation requires it
2. **Branch isolation** - Each tree branch carries its own transformation scope
3. **Clean transpilation** - State maps to transforms, Mutation maps to mesh ops

---

## Functional Parametric L-System Alphabet

Traditional L-System: `F + - [ ]` (drawing commands)
**Functional Parametric**: High-level geometric functions as axioms

### Σ (Alphabet) Definition

```
Σ_state = {
    𝕋(tx, ty, tz)      : Translate
    ℝ(rx, ry, rz)      : Rotate (Euler)
    𝕊(sx, sy, sz)      : Scale
    [                   : Push matrix (scope begin)
    ]                   : Pop matrix (scope end)
    𝕄(matrix)          : Direct matrix assignment
}

Σ_mutation = {
    □(w, h, d)         : Box primitive
    ○(r, segs)         : Sphere primitive
    △(r, h, segs)      : Cone primitive
    ⊕(A, B)            : Boolean union
    ⊖(A, B)            : Boolean difference
    ⊗(A, B)            : Boolean intersection
    ↑(profile, path)   : Extrude along path
    ⟳(profile, axis)   : Revolve around axis
    ∿(mesh, func)      : Deform by function
}

Σ = Σ_state ∪ Σ_mutation
```

### Production Rules (P)

```
P: Σ* → Σ*

Example rule (branching growth):
    Branch(n, θ) → 
        □(1, n, 1)                      # trunk segment
        𝕋(0, n, 0)                      # move to top
        [ℝ(0, 0, θ) Branch(n*0.7, θ)]   # left branch
        [ℝ(0, 0, -θ) Branch(n*0.7, θ)]  # right branch
```

---

## Node Schema

Each node in the SST carries both state and mutation data:

```typescript
interface SSTNode {
    // Identity
    id: string;              // Stable hash-based ID
    type: 'state' | 'mutation' | 'compound';
    
    // State Component (optional)
    state?: {
        transform: Matrix4x4 | null;
        scope: 'push' | 'pop' | 'none';
        parameters: Record<string, number>;
    };
    
    // Mutation Component (optional)  
    mutation?: {
        operation: keyof typeof Σ_mutation;
        operands: (number | SSTNode)[];
        result_type: 'mesh' | 'curve' | 'point_cloud';
    };
    
    // Tree Structure
    children: SSTNode[];
    parent: string | null;
}
```

---

## Matrix Stack Semantics

```
Stack: [M₀, M₁, M₂, ...]  where M₀ = Identity

Operations:
    push()      : Stack ← Stack ++ [top(Stack)]
    pop()       : Stack ← Stack[:-1]
    transform(T): top(Stack) ← top(Stack) × T

Evaluation:
    WorldMatrix(node) = ∏(Stack) at node's evaluation point
```

### Scope Rules

```
[ ... ]  creates isolated transformation scope

Example:
    𝕋(5, 0, 0)          # Move right 5
    [                    # Push: save state
        ℝ(0, 45, 0)      # Rotate 45° (local)
        □(1, 1, 1)       # Box at rotated position
    ]                    # Pop: restore state
    □(1, 1, 1)           # Box at original (non-rotated) position
```

---

## Transpilation Targets

The SST schema transpiles to platform-specific implementations:

| Symbol | Maya (MEL/Python) | Blender (Python) | Unreal (Blueprint/C++) |
|--------|-------------------|------------------|------------------------|
| `𝕋` | `cmds.move()` | `obj.location =` | `SetActorLocation()` |
| `ℝ` | `cmds.rotate()` | `obj.rotation_euler =` | `SetActorRotation()` |
| `𝕊` | `cmds.scale()` | `obj.scale =` | `SetActorScale3D()` |
| `□` | `cmds.polyCube()` | `bpy.ops.mesh.primitive_cube_add()` | `UProceduralMesh` |
| `⊕` | `cmds.polyBoolOp(op=1)` | `bpy.ops.object.modifier_add(type='BOOLEAN')` | `UBooleanOp` |

---

## Minimal Working Example

```
# L-System: Simple branching structure
Axiom: A
Rules:
    A → □(1, 3, 1) 𝕋(0, 3, 0) [ℝ(0, 0, 30) A] [ℝ(0, 0, -30) A]

# Iteration 0:
A

# Iteration 1:
□(1,3,1) 𝕋(0,3,0) [ℝ(0,0,30) A] [ℝ(0,0,-30) A]

# Iteration 2:
□(1,3,1) 𝕋(0,3,0) 
    [ℝ(0,0,30) □(1,3,1) 𝕋(0,3,0) [ℝ(0,0,30) A] [ℝ(0,0,-30) A]] 
    [ℝ(0,0,-30) □(1,3,1) 𝕋(0,3,0) [ℝ(0,0,30) A] [ℝ(0,0,-30) A]]
```

---

## Design Invariants

1. **Single Source of Truth**: The SST is the canonical representation
2. **State Never Mutates Geometry**: Transforms only affect *placement*
3. **Mutation Never Carries State**: Geometry ops are position-agnostic
4. **Evaluation is Deterministic**: Same SST → Same output geometry
5. **Lazy by Default**: Nothing computed until explicitly requested

---

## File References

```
core/
├── SKELETAL_SINGLETON_TREE.md   # This document (theory)
├── state_schema.md              # Matrix stack formalization
├── mutation_schema.md           # Geometric operations catalog
└── transpiler_spec.md           # Platform-specific mappings
```

---

*The skeleton holds the shape; the flesh fills it in.*
