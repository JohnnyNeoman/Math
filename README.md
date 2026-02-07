```
╔═══════════════════════════════════════════════════════════╗
║   ███╗   ███╗ █████╗ ████████╗██╗  ██╗                   ║
║   ████╗ ████║██╔══██╗╚══██╔══╝██║  ██║                   ║
║   ██╔████╔██║███████║   ██║   ███████║                   ║
║   ██║╚██╔╝██║██╔══██║   ██║   ██╔══██║                   ║
║   ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║                   ║
║   ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝                   ║
║         Geometry · Algebra · Neural Operators             ║
╚═══════════════════════════════════════════════════════════╝
```

> Math utilities, geometry tools, and procedural frameworks for 3D applications

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## Quick Overview

| Package | Status | Description |
|---------|--------|-------------|
| [**math-core**](./math-core) | ✅ Active | Zero-dependency vector/matrix library |
| [**procedural-framework**](./procedural-framework) | 🔨 WIP | Skeletal Singleton Tree system |
| [**ml-geometry**](./ml-geometry) | 📋 Planned | Neural operators for geometry |
| [**maya-math**](./maya-math) | ✅ Active | Maya MEL procedures + Geometric Synthesis Framework |

---

## Explore the Framework

### [Listen to the Executive Summary (5 min)](./maya-math/media/executive_summary_narration.mp3)

A narrated overview covering the Tri-Space Engine, SST architecture, and Neural Sketch Fields.

### [Read the Full Documentation →](./maya-math/EXECUTIVE_SUMMARY.md)

---

## The Tri-Space Engine

Three synchronized mathematical contexts for resolution-independent 3D:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   AFFINE    │◄──►│  CONFORMAL  │◄──►│  SPECTRAL   │
│   GL(4,ℝ)   │    │   PSL(2,ℂ)  │    │   L²(ℝ³)    │
└─────────────┘    └─────────────┘    └─────────────┘
     │                   │                  │
     ▼                   ▼                  ▼
  Position            Angles             Fields
  Rotation            Circles            Fourier
   Scale              Möbius              FNO
```

---

## Packages

### math-core

Pure Python vector and matrix library with zero external dependencies.

```python
from math_core import Vec3, Mat4

v = Vec3(1, 2, 3)
m = Mat4.rotation_y(45)
result = m @ v
```

### procedural-framework

Skeletal Singleton Tree (SST) — functional L-system separating state from mutation.

```python
from sst import Translate, Rotate, Box

scene = Translate(0, 1, 0) >> Box(1, 1, 1)
mesh = scene.collapse(resolution=1024)
```

### maya-math

2,308 MEL procedures for sketch-based 3D modeling:
- **Circle/tangent geometry** — Conformal PSL(2,ℂ) operations
- **Linear algebra utilities** — Matrix/vector operations
- **Camera-centric sketching** — 2D→3D projection

[Browse Cleaned Procedures →](./maya-math/cleaned/)

---

## Installation

```bash
git clone https://github.com/JohnnyNeoman/Math.git
cd Math
pip install -e .
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [EXECUTIVE_SUMMARY.md](./maya-math/EXECUTIVE_SUMMARY.md) | Complete framework overview |
| [MATHEMATICAL_COMPILER.md](./maya-math/docs/MATHEMATICAL_COMPILER.md) | Tri-Space Engine architecture |
| [SKELETAL_SINGLETON_TREE.md](./maya-math/docs/SKELETAL_SINGLETON_TREE.md) | SST state/mutation design |
| [NEURAL_SKETCH_FIELD.md](./maya-math/docs/NEURAL_SKETCH_FIELD.md) | AI surface anticipation |

---

## Philosophy

> *"The specification IS the execution. The tree IS the program. The algebra IS the geometry."*

Traditional 3D pipelines destroy mathematical structure. This framework preserves it through algebraic compilation.

---

## Contact

- GitHub: [@JohnnyNeoman](https://github.com/JohnnyNeoman)

---

*Part of the 3D Tools ML Hybrid framework*
