# SST Quick Reference

> **One-Page Cheat Sheet for the Math Layer**

---

## The Separation

```
Geometry = State(Mutation)
         = MatrixStack.apply(GeometricOperation)

State:    WHERE and HOW  →  Transforms, scopes, platforms
Mutation: WHAT           →  Primitives, CSG, deformations
```

---

## Core Formulas

### Gram-Schmidt Alignment
```
Y = normalize(N)
X = normalize(cross(Up, Y))    # guard singularity
Z = cross(X, Y)
M = [X | Y | Z | P]            # 4×4 affine
```

### Platform Reflection
```
M_reflect = P × S × P⁻¹

where:
    P = platform world matrix
    S = diag(-1,1,1,1) for X-axis flip
```

### Matrix Stack
```
push(): stack.append(copy(top))
pop():  stack.pop()
transform(T): top = top × T
```

---

## Alphabet Σ

| Symbol | Type | Effect |
|--------|------|--------|
| `T(v)` | State | Translate by v |
| `R(axis, θ)` | State | Rotate θ around axis |
| `S(v)` | State | Scale by v |
| `[ ]` | State | Push/Pop scope |
| `Platform(P)` | State | Set portable world center |
| `Align(p, n)` | State | Gram-Schmidt to surface |
| `Mirror(axis)` | Flow | Bifurcate: real + reflected |
| `Radial(n)` | Flow | Replicate n times rotated |
| `Instance(mesh)` | Mutation | Emit geometry |
| `□ ○ △` | Mutation | Primitives |
| `⊕ ⊖ ⊗` | Mutation | CSG ops |

---

## Node Execution

```python
def execute(node, state):
    node.exec_self(state)        # Apply this node's effect
    for child in node.children:
        child.execute(state)     # Recurse
```

**Mirror Node** (bifurcation):
```python
# PATH A: real
for child in children: child.execute(state)

# PATH B: reflected  
state.push()
state.M = state.M @ platform_reflect(P, axis)
for child in children: child.execute(state)
state.pop()
```

---

## Composition Rules

```
Sequential:     A ∘ B = A × B
Scoped:         [ A ] isolates A's effects
Inverse:        (A × B)⁻¹ = B⁻¹ × A⁻¹
Associative:    (A × B) × C = A × (B × C)
NOT Commutative: A × B ≠ B × A
```

---

## YAML → Nodes

```yaml
root:
  - Platform: { id: "P1" }
  - Mirror: { axis: X }
    children:
      - T: { v: [100, 0, 0] }
      - Instance: { mesh: "Arm" }
```

↓ compiles to ↓

```
Root
 └─ Platform("P1")
 └─ Mirror(X)
     └─ Translate([100,0,0])
     └─ Instance("Arm")
```

---

## Output Buffer

```python
state.buffer = [
    { geometry: "Arm", transform: M1, sym_depth: 0 },  # Real
    { geometry: "Arm", transform: M2, sym_depth: 1 },  # Reflected
]
```

---

## Invariants

1. All transforms are 4×4 affine matrices
2. Composition = matrix multiplication (right-to-left)
3. Push/Pop isolates scope
4. Platform reflection: `M = P × S × P⁻¹`
5. Gram-Schmidt produces orthonormal bases
6. System is **closed** under composition

---

## File Map

```
3D_tools_ML_hybrid/
├── core/
│   ├── SKELETAL_SINGLETON_TREE.md  ← Architecture overview
│   ├── math_foundations.md         ← Gram-Schmidt, reflection algebra
│   ├── node_algebra.md             ← Functional alphabet, composition
│   ├── state_schema.md             ← Matrix stack details
│   ├── mutation_schema.md          ← Geometry operations
│   └── transpiler_spec.md          ← Maya/Blender/Unreal mappings
└── index.jsonl                     ← Machine-readable index
```

---

*This math layer plugs into any program logic. The structure IS the code.*
