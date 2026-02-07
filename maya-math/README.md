# Maya MEL Scripts

> Legacy MEL procedures for sketch-based 3D modeling, circle/tangent geometry, and computational tools

## Overview

This collection contains **2,308 MEL procedures** across **68 files**, developed for:
- **Sketch-based modeling** - Camera-plane curve projection and retopology
- **Circle/tangent geometry** - Computational geometry for NURBS
- **Linear algebra utilities** - Vector and matrix operations for Maya

## Quick Navigation

| Document | Purpose |
|----------|---------|
| [PROCEDURE_CATALOG.md](./PROCEDURE_CATALOG.md) | Complete procedure listing with SST annotations |
| [FIXES_SUMMARY.md](./FIXES_SUMMARY.md) | Known issues and applied fixes |
| [Maya_Sketch_script.md](./Maya_Sketch_script.md) | Main sketch modeling script (357 procedures) |

## Folder Structure

```
MAYA/
├── README.md                   # This file
├── PROCEDURE_CATALOG.md        # Complete procedure catalog
├── FIXES_SUMMARY.md            # Error fixes documentation
├── Maya_Sketch_script.md       # Main sketch script (435 KB)
│
├── organized/                  # Source files by category
│   ├── circle-procedures/      # Circle creation, 3-point circles
│   ├── tangent-procedures/     # Point-to-circle tangents
│   ├── sketch-modeling/        # CAM projection, retopology
│   ├── linear-algebra/         # Matrix, vector operations
│   ├── array-utils/            # Array manipulation
│   └── polygon-ops/            # Polygon operations
│
├── cleaned/                    # Formatted, annotated MEL procedures
│   ├── circle_procedures.mel   # 176 cleaned procedures
│   ├── tangent_procedures.mel  # 61 cleaned procedures
│   ├── linear_algebra.mel      # 229 cleaned procedures
│   ├── polygon_ops.mel         # 132 cleaned procedures
│   ├── sketch_modeling.mel     # 214 cleaned procedures
│   ├── array_utils.mel         # 3 cleaned procedures
│   └── utility.mel             # 240 cleaned procedures
│
├── docs/                       # Framework documentation
│   ├── ABSTRACT.md             # SST framework vision
│   ├── MATHEMATICAL_COMPILER.md # Tri-Space Engine
│   ├── NEURAL_SKETCH_FIELD.md  # Anticipation loop
│   └── core/                   # Core module docs
│
├── implementations/            # Python reference implementations
│   └── python/
│       ├── math_core.py        # Universal math library
│       └── sst_nodes.py        # SST node system
│
├── legacy/                     # Original unmodified files
│   └── Maya_MEL_Proc_Scripts/  # Raw procedure scripts
│
├── meta/                       # Organizational tools
│   ├── FindNameOfVariables.mel # Variable cataloging
│   └── list-all-procs.mel      # Procedure listing
│
└── Maya_MEL_Proc_Scripts/      # Source procedure files
    └── ALL MEL SCRIPTS FOR FREEFORM/
```

## Procedure Categories

| Category | Count | SST Layer | Key Procedures |
|----------|-------|-----------|----------------|
| **circle** | 128 | conformal | `Circle3Point`, `CircleFromCurve`, `IsCircle` |
| **tangent** | 106 | conformal | `PointToCircleTangents`, `TangentCircles` |
| **sketch** | 118 | conformal/spectral | `ProjectToCameraPlane`, `MoveZCURVEModelingCAM` |
| **curve** | 389 | conformal | NURBS curves, arc operations |
| **matrix** | 160 | affine | `xyzRotation`, `GetRotationFromDirection` |
| **polygon** | 149 | affine | Face, edge, vertex operations |
| **array** | 565 | affine | Array manipulation, sorting |
| **utility** | 693 | affine | Selection, cleanup, misc |

## Key Procedures for SST Integration

### Circle Family (Conformal Layer)
```mel
// Create circle from 3 points
global proc string Circle3Point(vector $p1, vector $p2, vector $p3)

// Extract circle from NURBS curve
global proc string CircleFromCurve(string $curve, string $axis)

// Test if curve is circular
global proc int IsCircle(string $curve)

// Find intersection of two circles
global proc vector[] IntersectTwoCircles(string $c1, string $c2)
```

### Tangent Family (Conformal Layer)
```mel
// Calculate tangent lines from point to circle
global proc string[] PointToCircleTangents(float $radius, vector $circlePos, vector $pointPos)

// Find tangent lines between two circles
global proc string[] TangentCircles(string $circle1, string $circle2)

// Create circle tangent to two circles
global proc string TangentCircleBetweenCircle(string $c1, string $c2, float $radius)
```

### Sketch Modeling Family (Conformal/Spectral)
```mel
// Project point to camera plane
global proc vector ProjectToCameraPlane(vector $point, string $camera)

// Main CAM curve positioning
global proc MoveZCURVEModelingCAM(string $curve)

// Project curve onto surface
global proc AdvancedCurveMODprojectOnSurface(string $curve, string $surface)
```

### Matrix Family (Affine Layer)
```mel
// Quaternion-based 3D rotation
global proc matrix xyzRotation(float $x, float $y, float $z)

// Rotation matrix from direction vectors
global proc matrix GetRotationFromDirection(vector $dir, vector $up)
```

## Duplicate Analysis

436 procedures appear in multiple files. Top consolidation candidates:

| Procedure | Occurrences | Action |
|-----------|-------------|--------|
| `ArcLengthArray` | 10 | Consolidate to array-utils |
| `AddFloats` | 7 | Merge variants |
| `AppendFloatsZ` | 6 | Single parameterized version |
| `Add_Float_to_3PointFloats` | 6 | Consolidate |
| `appendMultiStringArray` | 6 | Consolidate |

## SST Integration Notes

### Mathematical Context Mapping

| MEL Category | SST Layer | Algebra | Preserves |
|--------------|-----------|---------|-----------|
| circle/tangent | Conformal | PSL(2,C) | Angles, circles |
| matrix | Affine | GL(4,R) | Position, orientation |
| sketch | Conformal/Spectral | Mixed | View-dependent |
| polygon | Affine | GL(4,R) | Topology |

### Python Translation Priority

1. **High** - Core math operations (`xyzRotation`, `Circle3Point`)
2. **Medium** - Geometry operations (`TangentCircles`, `CircleFromCurve`)
3. **Low** - Maya-specific utilities (selection, UI)

## Usage

### Load in Maya
```mel
// Source the main sketch script
source "Maya_Sketch_script.mel";

// Use circle procedures
string $circle = Circle3Point(<<0,0,0>>, <<1,0,0>>, <<0,1,0>>);
```

### Find Procedures
```mel
// Use the organizational script
string $vars[] = FindNameOfVariables("Circle", 1);
print $vars;
```

## Related Documentation

### Core Framework
- [ABSTRACT.md](./docs/ABSTRACT.md) - Project vision and SST framework
- [MATHEMATICAL_COMPILER.md](./docs/MATHEMATICAL_COMPILER.md) - Tri-space engine architecture
- [SKELETAL_SINGLETON_TREE.md](./docs/SKELETAL_SINGLETON_TREE.md) - Core SST architecture

### Neural/Spectral Framework
- [NEURAL_SKETCH_FIELD.md](./docs/NEURAL_SKETCH_FIELD.md) - Anticipation loop, FNO field solving
- [SUPERNODE_ABSTRACT.md](./docs/SUPERNODE_ABSTRACT.md) - Tri-Modal: Affine × Conformal × Spectral
- [SPECTRAL_SKETCH_FIELD.md](./docs/SPECTRAL_SKETCH_FIELD.md) - Harmonic analysis for sketch fields

### Python Reference Implementations
- [math_core.py](./implementations/python/math_core.py) - Vec3, Mat4, align_to_surface
- [sst_nodes.py](./implementations/python/sst_nodes.py) - SST Node System

### Core Module Documentation
- [math_foundations.md](./docs/core/math_foundations.md) - Gram-Schmidt, platform symmetry
- [state_schema.md](./docs/core/state_schema.md) - State machine schema
- [mutation_schema.md](./docs/core/mutation_schema.md) - Mutation operations
- [node_algebra.md](./docs/core/node_algebra.md) - Node-based algebra
- [ml_integration.md](./docs/core/ml_integration.md) - ML integration architecture

---

*Part of the 3D Tools ML Hybrid framework - bridging legacy Maya tools with modern SST architecture*
