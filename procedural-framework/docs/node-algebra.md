# Node Algebra: The Functional Alphabet

> **Operations as First-Class Citizens**
> How the SST composes into generative structures

---

## The Core Insight

Traditional L-Systems use a string alphabet: `F + - [ ]`

We use a **functional alphabet** where each symbol is a matrix-producing operation:

```
Traditional:  "F+F-F"  →  string rewriting  →  "F+F-F+F-F-F+F-F"
Functional:   Node(T,R,T)  →  tree traversal  →  Matrix × Matrix × Matrix
```

The strings ARE the expressions. The expressions produce geometry.

---

## 1. The Alphabet Σ

### State Operators (Σ_state)

These modify the traversal state but produce no geometry:

```
Σ_state = {
    T(v)        : Translate by vector v
    R(axis, θ)  : Rotate around axis by θ degrees  
    S(v)        : Scale by vector v
    [           : Push (save state)
    ]           : Pop (restore state)
    Platform(P) : Set active platform matrix P
    Align(p, n) : Gram-Schmidt surface adhesion
}
```

### Mutation Operators (Σ_mutation)

These consume state and produce geometry:

```
Σ_mutation = {
    Emit(mesh)      : Output mesh at current transform
    Instance(ref)   : Reference existing mesh asset
    Lathe(profile)  : Revolve 2D curve
    Sweep(profile, path) : Loft along path
    Bool(op, A, B)  : CSG operation
}
```

### Flow Operators (Σ_flow)

These control traversal structure:

```
Σ_flow = {
    Mirror(axis)    : Bifurcate into real + reflected
    Radial(n, axis) : Replicate n times around axis
    Repeat(n)       : Execute children n times
    Group(name)     : Semantic grouping (no transform effect)
}
```

---

## 2. The Walker (Execution State)

```python
class State:
    def __init__(self):
        self.stack = [identity_4x4()]  # Matrix stack
        self.platforms = {}             # Named platforms
        self.active_platform = None     # Current platform ID
        self.sym_depth = 0              # Symmetry nesting level
        self.buffer = []                # Output geometry
    
    @property
    def M(self) -> Matrix4x4:
        return self.stack[-1]
    
    def push(self):
        self.stack.append(self.M.copy())
    
    def pop(self):
        if len(self.stack) > 1:
            self.stack.pop()
    
    def transform(self, T: Matrix4x4):
        self.stack[-1] = self.stack[-1] @ T
    
    def emit(self, geometry, tags=None):
        self.buffer.append({
            'geometry': geometry,
            'transform': self.M.copy(),
            'sym_depth': self.sym_depth,
            'tags': tags or {}
        })
```

---

## 3. Node Types (The Functional Grammar)

### Base Node

```python
class Node:
    def __init__(self, name='Node'):
        self.name = name
        self.children = []
    
    def add(self, child):
        self.children.append(child)
        return child  # For chaining
    
    def execute(self, state: State):
        self.exec_self(state)
        for child in self.children:
            child.execute(state)
    
    def exec_self(self, state: State):
        pass  # Override in subclasses
```

### Transform Nodes

```python
class TranslateNode(Node):
    def __init__(self, v):
        super().__init__('T')
        self.v = v
    
    def exec_self(self, state):
        state.transform(mat_translate(self.v))

class RotateNode(Node):
    def __init__(self, axis, deg):
        super().__init__('R')
        self.axis = axis
        self.deg = deg
    
    def exec_self(self, state):
        state.transform(mat_rotate(self.axis, self.deg))

class ScaleNode(Node):
    def __init__(self, v):
        super().__init__('S')
        self.v = v
    
    def exec_self(self, state):
        state.transform(mat_scale(self.v))
```

### Scope Node (Push/Pop)

```python
class ScopeNode(Node):
    """Children execute in isolated scope"""
    def __init__(self):
        super().__init__('Scope')
    
    def execute(self, state):
        state.push()
        for child in self.children:
            child.execute(state)
        state.pop()
```

### Align Node (Smart Placement)

```python
class AlignNode(Node):
    def __init__(self, hit_point, normal):
        super().__init__('Align')
        self.p = hit_point
        self.n = normal
    
    def exec_self(self, state):
        M_align = gram_schmidt_align(self.p, self.n)
        state.transform(M_align)
```

### Platform Node

```python
class PlatformNode(Node):
    def __init__(self, platform_id, matrix):
        super().__init__('Platform')
        self.pid = platform_id
        self.matrix = matrix
    
    def exec_self(self, state):
        state.platforms[self.pid] = self.matrix
        state.active_platform = self.pid
```

### Mirror Node (Bifurcation)

```python
class MirrorNode(Node):
    def __init__(self, axis='X'):
        super().__init__('Mirror')
        self.axis = axis
    
    def execute(self, state):
        # PATH A: Real branch
        for child in self.children:
            child.execute(state)
        
        # PATH B: Reflected branch
        if state.active_platform is None:
            return  # No platform, no reflection
        
        P = state.platforms[state.active_platform]
        M_reflect = platform_reflect(P, self.axis)
        
        state.push()
        state.transform(M_reflect)
        state.sym_depth += 1
        
        for child in self.children:
            child.execute(state)
        
        state.sym_depth -= 1
        state.pop()
```

### Radial Node

```python
class RadialNode(Node):
    def __init__(self, n, axis='Z'):
        super().__init__('Radial')
        self.n = n
        self.axis = axis
    
    def execute(self, state):
        angle_step = 360.0 / self.n
        for i in range(self.n):
            state.push()
            state.transform(mat_rotate(self.axis, i * angle_step))
            for child in self.children:
                child.execute(state)
            state.pop()
```

### Instance Node (Geometry Emission)

```python
class InstanceNode(Node):
    def __init__(self, mesh_ref, tags=None):
        super().__init__('Instance')
        self.mesh = mesh_ref
        self.tags = tags or {}
    
    def exec_self(self, state):
        state.emit(self.mesh, {
            **self.tags,
            'is_symmetry': state.sym_depth > 0
        })
```

---

## 4. DSL Mapping (YAML → Nodes)

The YAML is just syntax for constructing the node tree:

```yaml
root:
  - Align: { point: [0,0,0], normal: [0,1,0] }
  - Platform: { id: "shoulder", matrix: World }
  - Mirror: { axis: X }
    children:
      - T: { v: [100, 0, 0] }
      - Instance: { mesh: "/Game/Arm" }
      - Radial: { n: 6, axis: Z }
        children:
          - T: { v: [50, 0, 0] }
          - Instance: { mesh: "/Game/Bolt" }
```

Compiles to:

```python
root = Node('Root')
root.add(AlignNode([0,0,0], [0,1,0]))
root.add(PlatformNode('shoulder', identity()))

mirror = root.add(MirrorNode('X'))
mirror.add(TranslateNode([100, 0, 0]))
mirror.add(InstanceNode('/Game/Arm'))

radial = mirror.add(RadialNode(6, 'Z'))
radial.add(TranslateNode([50, 0, 0]))
radial.add(InstanceNode('/Game/Bolt'))
```

---

## 5. Production Rules (L-System Style)

The node tree can include **parametric production rules**:

```python
class ProductionNode(Node):
    """Expands into child structure based on parameters"""
    def __init__(self, rule_func, params):
        super().__init__('Production')
        self.rule = rule_func
        self.params = params
    
    def execute(self, state):
        # Generate children dynamically
        generated = self.rule(self.params, state)
        for child in generated:
            child.execute(state)
```

Example rule (branching):

```python
def branch_rule(params, state):
    length = params.get('length', 100)
    angle = params.get('angle', 30)
    decay = params.get('decay', 0.7)
    depth = params.get('depth', 0)
    max_depth = params.get('max_depth', 4)
    
    if depth >= max_depth:
        return [InstanceNode('/Game/Leaf')]
    
    nodes = []
    
    # Trunk segment
    nodes.append(InstanceNode('/Game/Branch'))
    nodes.append(TranslateNode([0, length, 0]))
    
    # Left branch (scoped)
    left = ScopeNode()
    left.add(RotateNode('Z', angle))
    left.add(ProductionNode(branch_rule, {
        **params,
        'length': length * decay,
        'depth': depth + 1
    }))
    nodes.append(left)
    
    # Right branch (scoped)
    right = ScopeNode()
    right.add(RotateNode('Z', -angle))
    right.add(ProductionNode(branch_rule, {
        **params,
        'length': length * decay,
        'depth': depth + 1
    }))
    nodes.append(right)
    
    return nodes
```

---

## 6. The Generative Property

The system is **generative** because:

1. **Closure**: Compositions of nodes produce valid nodes
2. **Determinism**: Same tree + same state → same output
3. **Parametric**: Rules can generate structure dynamically
4. **Self-similar**: Trees can contain production rules that generate subtrees

```
Node composition forms a monoid:
    • Identity element: empty Node (no-op)
    • Associative: (A ∘ B) ∘ C = A ∘ (B ∘ C)
    • Closed: Node × Node → Node

This means ANY composition of nodes is valid.
```

---

## 7. Execution Model

```python
def execute_tree(root: Node) -> List[GeometryRecord]:
    state = State()
    root.execute(state)
    return state.buffer

# The buffer contains:
# [
#   { geometry: 'Arm', transform: M1, sym_depth: 0 },
#   { geometry: 'Arm', transform: M2, sym_depth: 1 },  # Mirrored
#   { geometry: 'Bolt', transform: M3, sym_depth: 0 },
#   ...
# ]
```

---

## 8. Platform Integration Points

| Platform | Node → Actor |
|----------|-------------|
| Maya | `cmds.instance()` or `cmds.duplicate()` with `state.M` |
| Blender | `bpy.ops.object.add()` with `obj.matrix_world = state.M` |
| Unreal | `SpawnActor()` or `HISM.AddInstance(FTransform(state.M))` |

The node tree is **platform-agnostic**. Only the emission step (`emit()`) needs platform-specific code.

---

*Nodes are verbs. The tree is a sentence. The geometry is the meaning.*
