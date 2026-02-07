# Geometric Synthesis Framework

## A Mathematical Compiler for 3D Content Creation

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ███████╗ ██████╗ ███╗   ███╗███████╗████████╗██████╗ ██╗ ██████╗  ║
║  ██╔════╝ ██╔════╝██╔═══██╗████╗ ████║██╔════╝╚══██╔══╝██╔══██╗██║██╔════╝  ║
║  ██║  ███╗█████╗  ██║   ██║██╔████╔██║█████╗     ██║   ██████╔╝██║██║       ║
║  ██║   ██║██╔══╝  ██║   ██║██║╚██╔╝██║██╔══╝     ██║   ██╔══██╗██║██║       ║
║  ╚██████╔╝███████╗╚██████╔╝██║ ╚═╝ ██║███████╗   ██║   ██║  ██║██║╚██████╗  ║
║   ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝ ╚═════╝  ║
║                                                                              ║
║   ███████╗██╗   ██╗███╗   ██╗████████╗██╗  ██╗███████╗███████╗██╗███████╗   ║
║   ██╔════╝╚██╗ ██╔╝████╗  ██║╚══██╔══╝██║  ██║██╔════╝██╔════╝██║██╔════╝   ║
║   ███████╗ ╚████╔╝ ██╔██╗ ██║   ██║   ███████║█████╗  ███████╗██║███████╗   ║
║   ╚════██║  ╚██╔╝  ██║╚██╗██║   ██║   ██╔══██║██╔══╝  ╚════██║██║╚════██║   ║
║   ███████║   ██║   ██║ ╚████║   ██║   ██║  ██║███████╗███████║██║███████║   ║
║   ╚══════╝   ╚═╝   ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚══════╝   ║
║                                                                              ║
║                  A Mathematical Compiler for 3D Modeling                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## Executive Summary

This framework represents a paradigm shift in 3D content creation: **mathematical specification IS the program**. Rather than treating geometry as data to be processed, we treat it as algebraic expressions to be compiled—enabling lossless manipulation, resolution independence, and neural anticipation of artist intent.

---

## The Core Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    GEOMETRIC SYNTHESIS PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ╔═══════════════╗    ╔═══════════════╗    ╔═══════════════╗          │
│   ║    SKETCH     ║ →  ║     LIFT      ║ →  ║    OPERATE    ║ →        │
│   ║   Boundary    ║    ║   to Math     ║    ║   Losslessly  ║          │
│   ╚═══════════════╝    ╚═══════════════╝    ╚═══════════════╝          │
│                                                                          │
│   ╔═══════════════╗    ╔═══════════════╗    ╔═══════════════╗          │
│   ║   COLLAPSE    ║ ←  ║  ANTICIPATE   ║ ←  ║    NEURAL     ║ ←        │
│   ║   to Mesh     ║    ║   Surface     ║    ║   Operators   ║          │
│   ╚═══════════════╝    ╚═══════════════╝    ╚═══════════════╝          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## The Tri-Space Engine

The framework operates in three synchronized mathematical contexts, each preserving different geometric invariants:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          THE TRI-SPACE ENGINE                                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║          ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              ║
║          │   AFFINE    │◄──►│  CONFORMAL  │◄──►│  SPECTRAL   │              ║
║          │   GL(4,ℝ)   │    │   PSL(2,ℂ)  │    │   L²(ℝ³)    │              ║
║          └─────────────┘    └─────────────┘    └─────────────┘              ║
║                │                   │                  │                      ║
║                ▼                   ▼                  ▼                      ║
║          ┌─────────────┐    ┌─────────────┐    ┌─────────────┐              ║
║          │  Position   │    │   Angles    │    │   Fields    │              ║
║          │  Rotation   │    │   Circles   │    │   Fourier   │              ║
║          │   Scale     │    │   Möbius    │    │    FNO      │              ║
║          └─────────────┘    └─────────────┘    └─────────────┘              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

| Layer | Algebra | Preserves | Use Case |
|-------|---------|-----------|----------|
| **Affine** | GL(4,ℝ) | Position, orientation, scale | World transforms, hierarchy |
| **Conformal** | PSL(2,ℂ) | Angles, circles, cross-ratios | Tangent constructions, circle packing |
| **Spectral** | L²(ℝ³) | Frequency content, smoothness | Resolution independence, FNO |

---

## Core Mathematical Structures

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        FUNDAMENTAL MATHEMATICS                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  AFFINE TRANSFORMATION MATRIX                                                │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                               │
│                                                                              │
│       ┌                          ┐                                           │
│       │  R₁₁  R₁₂  R₁₃  Tₓ      │                                           │
│   M = │  R₂₁  R₂₂  R₂₃  Tᵧ      │  ∈ GL(4,ℝ)                                │
│       │  R₃₁  R₃₂  R₃₃  Tᵤ      │                                           │
│       │   0    0    0    1       │                                           │
│       └                          ┘                                           │
│                                                                              │
│  MÖBIUS TRANSFORMATION (CONFORMAL)                                           │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                             │
│                                                                              │
│           az + b                                                             │
│   f(z) = ─────────  where ad - bc ≠ 0                                        │
│           cz + d                                                             │
│                                                                              │
│   • Maps circles to circles                                                  │
│   • Preserves angles at intersection                                         │
│   • Foundation for tangent constructions                                     │
│                                                                              │
│  FOURIER NEURAL OPERATOR (SPECTRAL)                                          │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                           │
│                                                                              │
│   f̂(k) = ∫ f(x) e^(-2πik·x) dx                                              │
│                                                                              │
│   FNO Layer: v(x) → σ(Wv(x) + K(a)v(x))                                     │
│                                                                              │
│   where K operates in Fourier space for                                      │
│   resolution-independent learning                                            │
│                                                                              │
│  BOUNDARY CURVATURE                                                          │
│  ━━━━━━━━━━━━━━━━━                                                          │
│                                                                              │
│        dθ                                                                    │
│   κ = ────  (rate of angle change per arc-length)                           │
│        ds                                                                    │
│                                                                              │
│  GRAM-SCHMIDT SURFACE ALIGNMENT                                              │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                                            │
│                                                                              │
│   Given: hit point P, surface normal N̂                                      │
│                                                                              │
│   ẑ = N̂                                                                     │
│   x̂ = normalize(arbitrary × ẑ)                                              │
│   ŷ = ẑ × x̂                                                                 │
│                                                                              │
│   Platform matrix: [x̂ | ŷ | ẑ | P]                                          │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## The Five Core Frameworks

### 1. Mathematical Compiler (MATHEMATICAL_COMPILER.md)

The core insight: **shift from coordinates to mathematical compilation**.

Traditional pipelines destroy structure:
- Continuous curves → jagged polylines
- Circles → approximated polygons
- Resolution locked at creation time
- Boolean operations create non-manifold edges

Our approach:
- Geometry stored as algebraic expressions
- Operations are expression transformations
- Resolution determined at render time
- Structure preserved through all operations

### 2. Skeletal Singleton Tree (SKELETAL_SINGLETON_TREE.md)

A **functional L-system** that cleanly separates:

```
┌─────────────────────────────────────────────────────────────────┐
│                 SKELETAL SINGLETON TREE                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   STATE NODES (Matrix Stack)      MUTATION NODES (Geometry)     │
│   ━━━━━━━━━━━━━━━━━━━━━━━━       ━━━━━━━━━━━━━━━━━━━━━━━━━━    │
│                                                                  │
│   • Translate(x, y, z)           • Box(w, h, d)                 │
│   • Rotate(θ, axis)              • Sphere(r)                    │
│   • Scale(sx, sy, sz)            • Extrude(profile, path)       │
│   • Push/Pop                     • Boolean(op, A, B)            │
│                                                                  │
│   WHERE geometry lives            WHAT geometry exists           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Benefits:
- **Lazy evaluation**: compute only what's needed
- **Branch isolation**: modifications don't cascade
- **Clean transpilation**: SST → any target platform

### 3. Neural Sketch Field (NEURAL_SKETCH_FIELD.md)

**Generative anticipation**: predict what surface the artist intends from boundary curves.

```
┌─────────────────────────────────────────────────────────────────┐
│                   NEURAL SKETCH FIELD                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│    Boundary Input          FNO Processing         Surface Output │
│    ━━━━━━━━━━━━━━         ━━━━━━━━━━━━━━        ━━━━━━━━━━━━━━ │
│                                                                  │
│    ╭───────────╮          ╭───────────╮         ╭───────────╮   │
│    │  ∂Ω curve │    →     │  Fourier  │    →    │  Surface  │   │
│    │  κ = dθ/ds│          │  Neural   │         │    Ω      │   │
│    ╰───────────╯          │  Operator │         ╰───────────╯   │
│                           ╰───────────╯                          │
│                                                                  │
│    Draw boundary    →    Solve PDE in     →    Generate mesh    │
│    with curvature        spectral domain       from field       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

The "ghost scaffolding" shows the predicted surface in real-time, letting artists confirm or refine before committing.

### 4. Geometric Scaffold Supernode (SUPERNODE_ABSTRACT.md)

The **unified architectural unit** implementing the Tri-Modal pipeline:

```
               LIFT              OPERATE            COLLAPSE
    Sketch ──────────► Math ──────────► Math ──────────► Mesh

    2D input    →    Algebraic    →    Lossless    →    Polygons
    from user        representation    transforms       for render
```

Every modeling operation is expressed as Supernode composition.

### 5. Compositional Closure (ABSTRACT.md)

The philosophical foundation: **mathematical specification IS the program**.

- No impedance mismatch between design and implementation
- Formal verification possible at every step
- Cross-platform by construction (SST transpiles to any target)

---

## Legacy Integration: Maya MEL Procedures

This framework builds on **2,308 MEL procedures** across 68 files, implementing:

| Category | Procedures | Mathematical Layer |
|----------|------------|-------------------|
| Circle Operations | 184 | Conformal PSL(2,ℂ) |
| Tangent Constructions | 71 | Conformal PSL(2,ℂ) |
| Linear Algebra | 240 | Affine GL(4,ℝ) |
| Sketch Modeling | 218 | Projection Camera→World |
| Polygon Operations | 135 | Mesh Topology |
| Array Utilities | 253 | Data Infrastructure |

Key procedures for SST integration:
- `Circle3Point(v1, v2, v3)` - Circle through three points
- `PointToCircleTangents(r, center, point)` - Tangent construction
- `ProjectToCameraPlane(point, camera)` - Sketch projection
- `xyzRotation(x, y, z)` - Quaternion-based rotation matrix

---

## Getting Started

1. **Explore the Documentation**
   - [MATHEMATICAL_COMPILER.md](./docs/MATHEMATICAL_COMPILER.md) - Core philosophy
   - [SKELETAL_SINGLETON_TREE.md](./docs/SKELETAL_SINGLETON_TREE.md) - SST architecture
   - [NEURAL_SKETCH_FIELD.md](./docs/NEURAL_SKETCH_FIELD.md) - Anticipation system

2. **Study the MEL Procedures**
   - [cleaned/](./cleaned/) - Formatted, annotated procedures
   - [circle_procedures.mel](./cleaned/circle_procedures.mel) - Conformal geometry
   - [linear_algebra.mel](./cleaned/linear_algebra.mel) - Matrix operations

3. **Review Python Implementations**
   - [math_core.py](./implementations/python/math_core.py) - Vec3, Mat4, align_to_surface
   - [sst_nodes.py](./implementations/python/sst_nodes.py) - SST Node System

---

## Quick Reference

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           QUICK REFERENCE CARD                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CREATE GEOMETRY                                                             │
│  ───────────────                                                             │
│  Circle3Point(p1, p2, p3)         Create circle through 3 points            │
│  CircleFromCurve(curve, axis)     Extract circle from NURBS                 │
│  TangentCircles(c1, c2)           Tangent lines between circles             │
│                                                                              │
│  TRANSFORM                                                                   │
│  ─────────                                                                   │
│  xyzRotation(x, y, z)             Rotation matrix from Euler angles         │
│  GetRotationFromDirection(d, u)   Matrix from direction + up vector         │
│  ProjectToCameraPlane(pt, cam)    Project 3D point to camera plane          │
│                                                                              │
│  SST NODES                                                                   │
│  ─────────                                                                   │
│  Translate(x, y, z)               State: set position                       │
│  Rotate(angle, axis)              State: set orientation                    │
│  Box(w, h, d)                     Mutation: create box geometry             │
│  Boolean(op, A, B)                Mutation: combine geometries              │
│                                                                              │
│  NEURAL FIELD                                                                │
│  ────────────                                                                │
│  encode_boundary(curve)           Extract κ, tangent, position              │
│  fno_solve(boundary, domain)      Predict interior field                    │
│  extract_surface(field)           Generate mesh from field                  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Diagram

```
                              ┌─────────────────────┐
                              │    USER INPUT       │
                              │  (Sketch Strokes)   │
                              └──────────┬──────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                         GEOMETRIC SCAFFOLD SUPERNODE                        │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────────────────────────┐    ┌────────────┐ │
│  │    LIFT     │    │           OPERATE               │    │  COLLAPSE  │ │
│  │             │    │                                 │    │            │ │
│  │  2D → Math  │───►│  ┌─────────┐  ┌─────────┐      │───►│ Math → Mesh│ │
│  │             │    │  │ Affine  │  │Conformal│      │    │            │ │
│  │  Biarc fit  │    │  │ GL(4,ℝ) │◄►│PSL(2,ℂ) │      │    │ Triangulate│ │
│  │  BFF param  │    │  └────┬────┘  └────┬────┘      │    │ to target  │ │
│  │             │    │       │            │           │    │ resolution │ │
│  └─────────────┘    │       ▼            ▼           │    └────────────┘ │
│                     │  ┌─────────────────────────┐   │                    │
│                     │  │       SPECTRAL          │   │                    │
│                     │  │        L²(ℝ³)           │   │                    │
│                     │  │                         │   │                    │
│                     │  │  Fourier Neural Operator│   │                    │
│                     │  │  Resolution Independence│   │                    │
│                     │  └─────────────────────────┘   │                    │
│                     │                                 │                    │
│                     └─────────────────────────────────┘                    │
│                                                                             │
├────────────────────────────────────────────────────────────────────────────┤
│                      SKELETAL SINGLETON TREE (SST)                          │
│                                                                             │
│    Root ─┬─ Translate(0,1,0) ─┬─ Box(1,1,1)                                │
│          │                    └─ Sphere(0.5)                                │
│          │                                                                  │
│          └─ Rotate(45°, Y) ─── Extrude(profile, path)                      │
│                                                                             │
│    State nodes (WHERE) ──────► Mutation nodes (WHAT)                       │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │    RENDER OUTPUT    │
                              │  (Resolution-free)  │
                              └─────────────────────┘
```

---

## License & Attribution

This framework synthesizes techniques from computational geometry, differential geometry, and neural operators. The MEL procedure library represents over a decade of development for camera-centric sketch-based modeling in Autodesk Maya.

---

*Part of the 3D Tools ML Hybrid framework — bridging legacy Maya tools with modern mathematical architecture*
