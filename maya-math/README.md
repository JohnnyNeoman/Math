# Geometric Synthesis Framework

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
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║                          THE TRI-SPACE ENGINE                                ║
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

> **Mathematical specification IS the program** — geometry as algebraic expressions, not data to be processed.

---

## Listen to the Executive Summary

[**Download Audio Narration (MP3)**](./media/executive_summary_narration.mp3) — 5 minute overview of the complete framework

---

## What Is This?

This framework treats 3D modeling as **mathematical compilation**:

1. **LIFT** — Convert 2D sketch input into algebraic representations
2. **OPERATE** — Transform losslessly in the appropriate mathematical space
3. **COLLAPSE** — Generate polygons only when rendering demands it

Traditional pipelines destroy structure at every step. Continuous curves become jagged polylines. Circles become approximated polygons. Resolution is locked at creation time. **We fix all of this.**

---

## Core Documents

| Document | Description |
|----------|-------------|
| [**EXECUTIVE_SUMMARY.md**](./EXECUTIVE_SUMMARY.md) | Complete overview with diagrams and mathematics |
| [**MATHEMATICAL_COMPILER.md**](./docs/MATHEMATICAL_COMPILER.md) | Tri-Space Engine architecture |
| [**SKELETAL_SINGLETON_TREE.md**](./docs/SKELETAL_SINGLETON_TREE.md) | State/Mutation separation |
| [**NEURAL_SKETCH_FIELD.md**](./docs/NEURAL_SKETCH_FIELD.md) | Generative anticipation system |
| [**SUPERNODE_ABSTRACT.md**](./docs/SUPERNODE_ABSTRACT.md) | Unified architectural unit |

---

## The Tri-Space Engine

Three synchronized mathematical contexts, each preserving different invariants:

| Layer | Algebra | Preserves | Operations |
|-------|---------|-----------|------------|
| **Affine** | GL(4,ℝ) | Position, orientation, scale | Translate, Rotate, Scale |
| **Conformal** | PSL(2,ℂ) | Angles, circles, cross-ratios | Möbius transforms, tangent constructions |
| **Spectral** | L²(ℝ³) | Frequency content, smoothness | Fourier Neural Operators |

---

## Quick Start

### Explore the MEL Procedures

```mel
// Circle through three points
string $circle = Circle3Point(<<0,0,0>>, <<1,0,0>>, <<0,1,0>>);

// Tangent from point to circle
string $tangents[] = PointToCircleTangents(1.0, <<0,0,0>>, <<2,0,0>>);

// Project to camera plane
vector $projected = ProjectToCameraPlane(<<1,2,3>>, "persp");
```

### Browse the Cleaned Scripts

- [`circle_procedures.mel`](./cleaned/circle_procedures.mel) — 184 conformal geometry procedures
- [`linear_algebra.mel`](./cleaned/linear_algebra.mel) — 240 matrix operations
- [`sketch_modeling.mel`](./cleaned/sketch_modeling.mel) — 218 camera-centric tools

### Python Implementation

```python
from math_core import Vec3, Mat4, align_to_surface
from sst_nodes import Translate, Rotate, Box, Boolean

# Build SST
scene = Translate(0, 1, 0) >> Box(1, 1, 1)
scene = scene | (Rotate(45, Vec3.Y) >> Sphere(0.5))

# Collapse to mesh
mesh = scene.collapse(resolution=1024)
```

---

## Repository Structure

```
maya-math/
├── EXECUTIVE_SUMMARY.md          # Complete framework overview
├── media/
│   └── executive_summary_narration.mp3  # Audio narration
│
├── docs/                         # Framework documentation
│   ├── ABSTRACT.md               # Vision and philosophy
│   ├── MATHEMATICAL_COMPILER.md  # Tri-Space Engine
│   ├── SKELETAL_SINGLETON_TREE.md
│   ├── NEURAL_SKETCH_FIELD.md
│   ├── SUPERNODE_ABSTRACT.md
│   └── core/                     # Core module specs
│       ├── math_foundations.md
│       ├── state_schema.md
│       ├── mutation_schema.md
│       └── ...
│
├── cleaned/                      # Annotated MEL procedures
│   ├── circle_procedures.mel     # 184 procedures
│   ├── linear_algebra.mel        # 240 procedures
│   ├── sketch_modeling.mel       # 218 procedures
│   ├── tangent_procedures.mel    # 71 procedures
│   ├── polygon_ops.mel           # 135 procedures
│   └── utility.mel               # 253 procedures
│
└── implementations/
    └── python/
        ├── math_core.py          # Vec3, Mat4, align_to_surface
        └── sst_nodes.py          # SST Node System
```

---

## Core Mathematics

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        FUNDAMENTAL STRUCTURES                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  AFFINE MATRIX                    MÖBIUS TRANSFORM                           │
│                                                                              │
│  ┌                    ┐                az + b                                │
│  │ R₁₁ R₁₂ R₁₃  Tₓ   │        f(z) = ─────────                              │
│  │ R₂₁ R₂₂ R₂₃  Tᵧ   │                cz + d                                │
│  │ R₃₁ R₃₂ R₃₃  Tᵤ   │                                                      │
│  │  0   0   0   1    │        where ad - bc ≠ 0                             │
│  └                    ┘                                                      │
│                                                                              │
│  FOURIER TRANSFORM               CURVATURE                                   │
│                                                                              │
│  f̂(k) = ∫ f(x) e^(-2πik·x) dx         dθ                                   │
│                                   κ = ────                                   │
│                                        ds                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

- **Resolution Independence** — Geometry stored as algebraic expressions, not fixed-resolution meshes
- **Lossless Operations** — Boolean, extrude, bevel without destroying structure
- **Neural Anticipation** — Predict surfaces from boundary curves in real-time
- **Cross-Platform** — SST transpiles to any target (Maya, Blender, WebGL, etc.)
- **Camera-Centric** — 2D sketch input naturally projects to 3D

---

## Statistics

- **2,308** MEL procedures across 68 files
- **1,055** cleaned and annotated procedures
- **5** core mathematical frameworks
- **3** synchronized algebraic layers

---

*Part of the 3D Tools ML Hybrid framework — bridging legacy Maya tools with modern mathematical architecture*
