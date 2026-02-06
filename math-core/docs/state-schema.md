# State Schema: The Matrix Stack

> **Transformation Context for Skeletal Singleton Tree**
> Formalizing the "Where and How"

---

## Abstract

State represents the **accumulated transformation context** at any point in the SST evaluation. It answers: "If I create geometry here, where does it appear in world space?"

---

## Matrix Stack Data Structure

```
┌─────────────────────────────────────────┐
│            MATRIX STACK                 │
├─────────────────────────────────────────┤
│  top →  M₃ = Local×Parent×...×Identity  │
│         M₂                              │
│         M₁                              │
│  base → M₀ = Identity                   │
└─────────────────────────────────────────┘
```

### Formal Definition

```
MatrixStack = List<Matrix4x4>

invariant: |MatrixStack| ≥ 1
invariant: MatrixStack[0] = I₄ₓ₄ (Identity)

current(): Matrix4x4 = MatrixStack[-1]  # top of stack
```

---

## Operations

### Push (Scope Begin)

```
push(): void
    MatrixStack.append(copy(current()))
    
postcondition: |MatrixStack|' = |MatrixStack| + 1
postcondition: current()' = current()
```

**Semantics**: Create a checkpoint. Subsequent transforms are isolated.

### Pop (Scope End)

```
pop(): void
    precondition: |MatrixStack| > 1
    MatrixStack.pop()
    
postcondition: |MatrixStack|' = |MatrixStack| - 1
```

**Semantics**: Discard current scope, return to checkpoint state.

### Transform (Accumulate)

```
transform(T: Matrix4x4): void
    MatrixStack[-1] = MatrixStack[-1] × T
    
postcondition: current()' = current() × T
```

**Semantics**: Compose transform onto current context.

---

## Transform Constructors

### Translate: 𝕋(tx, ty, tz)

```
𝕋(tx, ty, tz) → Matrix4x4:
    ┌                    ┐
    │ 1  0  0  tx │
    │ 0  1  0  ty │
    │ 0  0  1  tz │
    │ 0  0  0   1 │
    └                    ┘
```

### Rotate: ℝ(rx, ry, rz) — Euler angles (degrees)

```
ℝ(rx, ry, rz) → Matrix4x4:
    Rx × Ry × Rz  (composition order configurable)
    
    where Rx, Ry, Rz are rotation matrices about respective axes
```

**Convention Note**: Default order is XYZ. Configurable per-project.

### Scale: 𝕊(sx, sy, sz)

```
𝕊(sx, sy, sz) → Matrix4x4:
    ┌                    ┐
    │ sx  0   0   0 │
    │ 0   sy  0   0 │
    │ 0   0   sz  0 │
    │ 0   0   0   1 │
    └                    ┘
```

---

## State Node Schema (JSONL)

```jsonl
{"type":"state_op","id":"s001","op":"push","depth":1}
{"type":"state_op","id":"s002","op":"translate","params":{"tx":5,"ty":0,"tz":0},"depth":1}
{"type":"state_op","id":"s003","op":"rotate","params":{"rx":0,"ry":45,"rz":0},"depth":1}
{"type":"state_op","id":"s004","op":"pop","depth":0}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `type` | `"state_op"` | Record type identifier |
| `id` | string | Stable unique ID |
| `op` | `"push"\|"pop"\|"translate"\|"rotate"\|"scale"\|"matrix"` | Operation type |
| `params` | object | Operation parameters (optional) |
| `depth` | number | Current stack depth after operation |

---

## Scope Visualization

```
Depth 0: ─────────────────────────────────────────────────────
         │
         [ (push → depth 1)
         │    
Depth 1: │    𝕋(5,0,0)  ℝ(0,45,0)  □(create box)
         │    
         │    [ (push → depth 2)
         │    │
Depth 2: │    │    𝕊(2,2,2)  □(create scaled box)
         │    │
         │    ] (pop → depth 1)
         │
         ] (pop → depth 0)
         │
Depth 0: ─────────────────────────────────────────────────────
```

---

## Evaluation Algorithm

```python
def evaluate_state_to_point(sst_node: SSTNode, nodes: List[SSTNode]) -> Matrix4x4:
    """Compute world matrix at a given node in the SST."""
    
    stack = [Matrix4x4.identity()]
    
    for node in traverse_to(sst_node, nodes):
        if node.state is None:
            continue
            
        match node.state.scope:
            case 'push':
                stack.append(stack[-1].copy())
            case 'pop':
                if len(stack) > 1:
                    stack.pop()
        
        if node.state.transform is not None:
            stack[-1] = stack[-1] @ node.state.transform
    
    return stack[-1]
```

---

## Properties

### Associativity
```
(A × B) × C = A × (B × C)
```
Matrix multiplication is associative, so evaluation order only matters at scope boundaries.

### Non-Commutativity
```
A × B ≠ B × A (in general)
```
Transform order matters: translate then rotate ≠ rotate then translate.

### Scope Isolation
```
[ ... A ... ] B  ⟹  A does not affect B
```
Anything inside brackets is isolated from siblings after the closing bracket.

---

## Platform Mapping

| State Op | Maya | Blender | Unreal |
|----------|------|---------|--------|
| `push` | implicit with `-r` flag | `matrix_world.copy()` | `GetTransform()` |
| `pop` | manual restore | assign cached matrix | `SetActorTransform()` |
| `translate` | `cmds.move(tx,ty,tz,r=True)` | `obj.location += Vector((tx,ty,tz))` | `AddActorLocalOffset()` |
| `rotate` | `cmds.rotate(rx,ry,rz,r=True)` | `obj.rotation_euler.rotate()` | `AddActorLocalRotation()` |
| `scale` | `cmds.scale(sx,sy,sz,r=True)` | `obj.scale = obj.scale * Vector((sx,sy,sz))` | `SetActorScale3D()` |

---

*State is the journey; Mutation is the destination.*
