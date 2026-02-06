# Mathematical Foundations

> **The Algebraic Core of the Skeletal Singleton Tree**
> Gram-Schmidt | Platform Symmetry | Composition Algebra

---

## Overview

This document defines the **mathematical primitives** that make the SST system compositional. These operations form a closed algebra — any composition of valid operations produces a valid operation.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MATH LAYER ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              PRIMITIVE OPERATIONS                        │  │
│   │   • Gram-Schmidt (surface adhesion)                      │  │
│   │   • Platform Reflection (portable symmetry)              │  │
│   │   • Affine Transforms (T, R, S)                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              COMPOSITION ALGEBRA                         │  │
│   │   • Sequential: A ∘ B = A × B                            │  │
│   │   • Scoped: [ A ] isolates A's effects                   │  │
│   │   • Branching: A splits into {A₁, A₂, ...}               │  │
│   └─────────────────────────────────────────────────────────┘  │
│                           │                                     │
│                           ▼                                     │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              DERIVED OPERATIONS                          │  │
│   │   • Mirror(axis, platform)                               │  │
│   │   • Radial(n, axis)                                      │  │
│   │   • Align(surface, uv)                                   │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Gram-Schmidt Alignment

**Purpose**: Construct an orthonormal basis aligned to an arbitrary surface normal, with a stable "up" direction.

### The Problem

Given:
- `P` — Hit point on surface (position)
- `N` — Surface normal at hit point
- `Up_global` — World up vector, typically `(0, 1, 0)`

Construct: A 4×4 matrix `M` such that:
- Y-axis aligns with `N`
- X and Z are orthogonal to Y and each other
- Origin is at `P`

### The Algorithm

```
Align(P, N, Up = (0,1,0)) → Matrix4×4:

    1. Y = normalize(N)                    # Primary axis = normal
    
    2. if |dot(Y, Up)| > 0.99:             # Handle singularity
           temp = (1, 0, 0)                 # Use alternate up
       else:
           temp = Up
    
    3. X = normalize(cross(temp, Y))       # Gram-Schmidt step
    
    4. Z = cross(X, Y)                     # Complete the basis
    
    5. return [  X.x  X.y  X.z  0  ]       # Column-major
              [  Y.x  Y.y  Y.z  0  ]
              [  Z.x  Z.y  Z.z  0  ]
              [  P.x  P.y  P.z  1  ]
```

### Properties

| Property | Guarantee |
|----------|-----------|
| Orthonormal | X⊥Y⊥Z, all unit length |
| Determinant | det(M) = 1 (right-handed, no scale) |
| Stable | Continuous except at singularity |
| Invertible | M⁻¹ = Mᵀ for rotation block |

### Code

```python
def align_to_surface(hit_point, normal, up=(0,1,0)):
    y = normalize(normal)
    
    # Singularity guard
    if abs(dot(y, up)) > 0.99:
        temp = (1, 0, 0)
    else:
        temp = up
    
    x = normalize(cross(temp, y))
    z = cross(x, y)  # Already unit length
    
    return Matrix4x4(
        x[0], x[1], x[2], 0,
        y[0], y[1], y[2], 0,
        z[0], z[1], z[2], 0,
        hit_point[0], hit_point[1], hit_point[2], 1
    )
```

---

## 2. Platform Symmetry (Portable World Center)

**Purpose**: Define reflection relative to a movable local coordinate system, not just world axes.

### The Insight

A reflection across world X is simple: `S = diag(-1, 1, 1, 1)`

But we want reflection across a **platform's local X**, where the platform can be anywhere and rotated arbitrarily.

### The Formula

```
M_reflect = P × S × P⁻¹
```

Where:
- `P` — Platform's world matrix (4×4)
- `S` — Scale matrix with sign flip: `diag(-1, 1, 1, 1)` for X-reflection
- `P⁻¹` — Inverse of platform matrix

### Derivation

```
To reflect point Q across Platform's local X-plane:

1. Q_local = P⁻¹ × Q       # Transform to platform space
2. Q_flip  = S × Q_local   # Flip in platform space  
3. Q_world = P × Q_flip    # Transform back to world

Combined: Q' = P × S × P⁻¹ × Q
                \_______/
                M_reflect
```

### Multi-Axis Reflection

For combined reflections (XY, XZ, YZ, XYZ):

```
S_X   = diag(-1,  1,  1, 1)
S_Y   = diag( 1, -1,  1, 1)
S_Z   = diag( 1,  1, -1, 1)
S_XY  = diag(-1, -1,  1, 1)
S_XZ  = diag(-1,  1, -1, 1)
S_YZ  = diag( 1, -1, -1, 1)
S_XYZ = diag(-1, -1, -1, 1)   # Point reflection (inversion)
```

### Code

```python
def platform_reflect(platform_matrix, axis='X'):
    P = platform_matrix
    P_inv = inverse(P)
    
    # Build scale matrix based on axis
    s = {'X': (-1,1,1), 'Y': (1,-1,1), 'Z': (1,1,-1),
         'XY': (-1,-1,1), 'XZ': (-1,1,-1), 'YZ': (1,-1,-1),
         'XYZ': (-1,-1,-1)}[axis]
    
    S = diag(s[0], s[1], s[2], 1)
    
    return P @ S @ P_inv
```

### The Mirror Node Pattern

```python
class MirrorNode(Node):
    def execute(self, state):
        # PATH A: Execute children normally (the "real" branch)
        for child in self.children:
            child.execute(state)
        
        # PATH B: Execute children in mirrored space (the "ghost" branch)
        P = state.platforms[state.active_platform]
        M_reflect = platform_reflect(P, self.axis)
        
        state.push()
        state.M = state.M @ M_reflect
        state.symmetry_depth += 1
        
        for child in self.children:
            child.execute(state)
        
        state.symmetry_depth -= 1
        state.pop()
```

---

## 3. Affine Transform Algebra

### Primitive Matrices

**Translation**:
```
T(tx, ty, tz) = [ 1  0  0  tx ]
                [ 0  1  0  ty ]
                [ 0  0  1  tz ]
                [ 0  0  0   1 ]
```

**Rotation** (axis-angle form, here Z-axis):
```
R_z(θ) = [ cos(θ)  -sin(θ)  0  0 ]
         [ sin(θ)   cos(θ)  0  0 ]
         [   0        0     1  0 ]
         [   0        0     0  1 ]
```

**Scale**:
```
S(sx, sy, sz) = [ sx  0   0   0 ]
                [ 0   sy  0   0 ]
                [ 0   0   sz  0 ]
                [ 0   0   0   1 ]
```

### Composition Rules

| Operation | Formula | Note |
|-----------|---------|------|
| Sequential | `A ∘ B = A × B` | Right-to-left application |
| Inverse | `(A × B)⁻¹ = B⁻¹ × A⁻¹` | Reverses order |
| Associative | `(A × B) × C = A × (B × C)` | Grouping doesn't matter |
| NOT Commutative | `A × B ≠ B × A` (generally) | Order matters |

### Identity and Inverse

```
T⁻¹(tx, ty, tz) = T(-tx, -ty, -tz)
R⁻¹(θ)          = R(-θ) = Rᵀ
S⁻¹(sx, sy, sz) = S(1/sx, 1/sy, 1/sz)
```

---

## 4. Composition Algebra (The "Generative" Property)

The system is generative because compositions of valid operations produce valid operations.

### The Walker Algebra

```
State = { M: Matrix4×4, Stack: List<Matrix4×4>, depth: Int }

Operations on State:
    push(s)     : s.Stack.append(copy(s.M))
    pop(s)      : s.M = s.Stack.pop()
    transform(s, T) : s.M = s.M × T
```

### Node Composition

Nodes form a **tree algebra** where:

```
execute: Node × State → State

Leaf nodes:      execute(leaf, s) = transform(s, leaf.matrix)
Branch nodes:    execute(branch, s) = foldl(execute, s, branch.children)
Scope nodes:     execute(scope, s) = pop(foldl(execute, push(s), scope.children))
```

### The Key Insight: Strings → Expressions → Strings

Just as L-Systems turn strings into expressions that generate new strings:

```
L-System:   "A"  →  "AB"  →  "ABA"  →  ...
                rule(A)=AB

Our System: Node →  Matrix  →  Geometry
                execute(node, state) = transformed_mesh
```

The **axiom** is the root node. The **production rules** are the node's execute methods. The **derivation** is the tree traversal.

---

## 5. Derived Operations

These are compositions of primitives that form useful higher-level operations.

### Radial(n, axis)

Generate `n` copies rotated around `axis`:

```
Radial(n, axis) = ∏_{i=0}^{n-1} [ R(axis, i × 360/n) ]
```

As a node:
```python
class RadialNode(Node):
    def execute(self, state):
        angle_step = 360.0 / self.n
        for i in range(self.n):
            state.push()
            state.M = state.M @ R(self.axis, i * angle_step)
            for child in self.children:
                child.execute(state)
            state.pop()
```

### LookAt(target, up)

Construct basis looking at target point:

```
LookAt(eye, target, up) → Matrix4×4:
    Z = normalize(eye - target)    # Or target - eye, convention varies
    X = normalize(cross(up, Z))
    Y = cross(Z, X)
    return [X, Y, Z, eye]
```

### Sweep(profile, path, frames)

Extrude profile along path using parallel transport or Frenet frames:

```
Sweep = ∏_{t ∈ path} [ Align(path(t), tangent(t)) × profile ]
```

---

## 6. Numerical Stability

### Singularity Handling

| Situation | Detection | Resolution |
|-----------|-----------|------------|
| Normal ∥ Up | `|dot(N, Up)| > 0.99` | Use alternate up `(1,0,0)` |
| Zero scale | `|s| < ε` | Clamp to `ε` or skip |
| Degenerate matrix | `|det(M)| < ε` | Fallback to identity |

### Epsilon Values

```
ε_dot    = 0.99    # For parallel detection
ε_scale  = 1e-6    # Minimum scale
ε_det    = 1e-8    # Determinant threshold
```

---

## 7. Summary: The Mathematical Contract

```
┌────────────────────────────────────────────────────────────────┐
│                 MATHEMATICAL INVARIANTS                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. All transforms are 4×4 affine matrices                      │
│  2. Composition is matrix multiplication (right-to-left)        │
│  3. State isolation via push/pop preserves parent context       │
│  4. Platform reflection: M = P × S × P⁻¹                        │
│  5. Gram-Schmidt produces orthonormal bases                     │
│  6. Every operation has a well-defined inverse                  │
│                                                                 │
│  Given these guarantees, ANY tree of operations is valid.       │
│  The system is CLOSED under composition.                        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

*The math is the DNA. The code is just the ribosome.*
