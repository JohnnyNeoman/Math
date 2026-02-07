# Comprehensive Abstract: Maya MEL Procedure Framework

> **A Camera-Centric Approach to Sketch-Based 3D Modeling**
> **Legacy MEL Architecture Analysis & Future SST Integration**
> Version 1.0 | 2026-02-06

---

## Executive Summary

This document analyzes a collection of **2,308 MEL procedures** across **68 files**, developed over years of practical 3D modeling work. The framework represents a **camera-centric approach** to sketch-based modeling, where 2D input curves are projected through the camera view to construct 3D geometry. The mathematical foundation combines **linear algebra** (matrix operations, rotations) with **conformal geometry** (circle/tangent constructions) to enable intuitive geometric authoring.

The analysis reveals:
- **749 unique procedures** with **1,559 duplicate instances** (consolidation candidates)
- Heavy reliance on **array-based batch processing** (565 procedures)
- Sophisticated **circle/tangent geometry** operations (234 procedures)
- A **camera-plane projection** architecture that anticipates modern Neural Sketch Field concepts

This legacy codebase provides the foundation for the **Skeletal Singleton Tree (SST)** framework, the **Neural Sketch Field** anticipation system, and the **Geometric Scaffold Supernode**—a mathematical compiler that lifts discrete sketches into continuous, resolution-independent representations.

---

## 1. Project Framework Overview

### 1.1 The Core Vision

The framework treats **sketch input as a first-class geometric primitive**. Unlike traditional modeling where artists manipulate vertices directly, this system interprets 2D strokes as constraints on 3D form:

```
Traditional Pipeline:    Sketch → Trace → Extrude → Edit vertices
This Framework:          Sketch → Project → Solve → Generate
```

The fundamental insight is that a **camera-plane projection** creates a natural mapping between 2D input and 3D intent. The artist draws in screen space; the system infers depth and form.

### 1.2 Procedure Distribution

| Category | Count | Percentage | Primary Purpose |
|----------|-------|------------|-----------------|
| **utility** | 693 | 30.0% | Selection, cleanup, Maya API wrappers |
| **array** | 565 | 24.5% | Batch processing, data transformation |
| **curve** | 389 | 16.9% | NURBS curve manipulation, arc operations |
| **matrix** | 160 | 6.9% | Rotation, transformation, linear algebra |
| **polygon** | 149 | 6.5% | Face, edge, vertex operations |
| **circle** | 128 | 5.5% | Circle creation, 3-point circles, packing |
| **sketch** | 118 | 5.1% | Camera projection, retopology |
| **tangent** | 106 | 4.6% | Point-to-circle tangents, tangent circles |

This distribution reveals the framework's priorities:
1. **Data processing infrastructure** (utility + array = 54.5%)
2. **Curve-based geometry** (curve + circle + tangent = 27%)
3. **Transformation mathematics** (matrix = 6.9%)
4. **Sketch-to-3D pipeline** (sketch = 5.1%)

---

## 2. Mathematical Foundations

### 2.1 Linear Algebra Layer (Affine Space)

The matrix operations form the **skeletal structure** of the framework, handling position, orientation, and scale in 3D space.

#### Core Algebra: GL(4, ℝ)

The General Linear Group of 4×4 invertible matrices provides:

```
Transformation matrix M ∈ GL(4, ℝ):

    ┌                          ┐
    │  R₁₁  R₁₂  R₁₃  Tₓ      │
M = │  R₂₁  R₂₂  R₂₃  Tᵧ      │
    │  R₃₁  R₃₂  R₃₃  Tᵤ      │
    │   0    0    0    1       │
    └                          ┘

Where:
- R ∈ SO(3): Rotation submatrix (orthonormal, det = 1)
- T ∈ ℝ³: Translation vector
```

#### Key Procedures

| Procedure | Signature | Mathematical Operation |
|-----------|-----------|------------------------|
| `xyzRotation` | `(float $x, $y, $z) → matrix` | Euler → Rotation matrix via quaternion |
| `GetRotationFromDirection` | `(vector $dir, $up) → matrix` | Gram-Schmidt orthonormalization |
| `MovePointDirectionAndDistance` | `(float[] $dir, $dist, $point) → float[]` | Vector addition: **p' = p + d·n̂** |
| `MirrorFloatXYZ` | `(float[] $point, $plane) → float[]` | Reflection: **p' = p - 2(p·n̂)n̂** |

#### Rotation Implementation

The `xyzRotation` procedure implements Euler angle composition:

```
R(α, β, γ) = Rz(γ) · Ry(β) · Rx(α)

Where:
        ┌                      ┐
Rx(α) = │  1     0       0     │
        │  0   cos α  -sin α   │
        │  0   sin α   cos α   │
        └                      ┘

        ┌                      ┐
Ry(β) = │  cos β   0   sin β   │
        │    0     1     0     │
        │ -sin β   0   cos β   │
        └                      ┘

        ┌                      ┐
Rz(γ) = │  cos γ  -sin γ   0   │
        │  sin γ   cos γ   0   │
        │    0       0     1   │
        └                      ┘
```

### 2.2 Conformal Geometry Layer

The circle and tangent procedures operate in **conformal space**, where the fundamental invariant is **angle preservation**.

#### Core Algebra: PSL(2, ℂ)

Möbius transformations form the Projective Special Linear Group:

```
f(z) = (az + b) / (cz + d)

Where a, b, c, d ∈ ℂ and ad - bc ≠ 0

Properties:
- Circles map to circles (including lines as infinite-radius circles)
- Angles are preserved at all points
- Cross-ratio is invariant
```

#### The Three-Point Circle (Fundamental Operation)

The `Circle3Point` family computes the unique circle through three non-collinear points:

```
Given points P₁, P₂, P₃ ∈ ℝ²:

1. Compute perpendicular bisectors:
   - L₁₂: perpendicular bisector of P₁P₂
   - L₂₃: perpendicular bisector of P₂P₃

2. Find center C = L₁₂ ∩ L₂₃

3. Compute radius r = |C - P₁|

Algebraically:
    |x - C.x|² + |y - C.y|² = r²

Where C solves:
    ┌                                          ┐ ┌     ┐   ┌                      ┐
    │ 2(P₂.x - P₁.x)  2(P₂.y - P₁.y)          │ │ C.x │   │ |P₂|² - |P₁|²        │
    │                                          │ │     │ = │                      │
    │ 2(P₃.x - P₂.x)  2(P₃.y - P₂.y)          │ │ C.y │   │ |P₃|² - |P₂|²        │
    └                                          ┘ └     ┘   └                      ┘
```

#### Point-to-Circle Tangent (Key Geometric Construction)

The `PointToCircleTangents` procedure computes tangent lines from an external point to a circle:

```
Given:
- Circle C with center O, radius r
- External point P where |OP| > r

Tangent points T₁, T₂ satisfy:
1. |OT| = r (on circle)
2. OT ⊥ PT (tangent condition)

Construction:
1. Compute d = |OP|
2. Compute tangent length: t = √(d² - r²)
3. Compute angle: θ = arctan(r/t)
4. Rotate OP by ±θ, scale to length t

Tangent points:
    T₁ = O + r · rotate(normalize(P - O), +θ)
    T₂ = O + r · rotate(normalize(P - O), -θ)
```

#### Circle Intersection

The `IntersectTwoCircles` procedure finds intersection points of two circles:

```
Given circles C₁(O₁, r₁) and C₂(O₂, r₂):

Let d = |O₂ - O₁|

Intersection exists when: |r₁ - r₂| ≤ d ≤ r₁ + r₂

Intersection points:
    a = (r₁² - r₂² + d²) / (2d)
    h = √(r₁² - a²)

    P = O₁ + a · normalize(O₂ - O₁)

    I₁ = P + h · perpendicular(normalize(O₂ - O₁))
    I₂ = P - h · perpendicular(normalize(O₂ - O₁))
```

### 2.3 The Camera Projection Layer

The sketch procedures implement **camera-centric projection**, the core innovation of this framework.

#### Projection Mathematics

```
Given:
- Camera position C ∈ ℝ³
- View direction V̂ ∈ S² (unit sphere)
- Up vector Û ∈ S²
- Sketch point p ∈ ℝ² (screen space)

Camera basis:
    X̂ = normalize(V̂ × Û)    (right)
    Ŷ = normalize(X̂ × V̂)    (up)
    Ẑ = V̂                    (forward)

Screen-to-world (at depth d):
    P₃D = C + d·Ẑ + p.x·X̂ + p.y·Ŷ

For orthographic projection:
    P₃D = C + p.x·X̂ + p.y·Ŷ  (fixed depth)
```

#### Key Procedures

| Procedure | Purpose | Mathematical Basis |
|-----------|---------|-------------------|
| `PointToCameraPlane` | Project 3D point to screen | **p₂D = (P - C) · [X̂, Ŷ]ᵀ** |
| `MoveZCURVEModelingCAM` | Position curve on camera plane | Sets Z-depth in camera space |
| `VecPointsToCameraPlane` | Batch projection | Vectorized screen projection |
| `ProjectToCameraPlane` | Full pipeline | Camera → World → Screen |
| `nurbsViewDirectionVectorCam` | Get camera direction | Extract V̂ from camera node |

#### The Retopology Pipeline

The `StartofCurveScriptRetopo` procedures implement surface-constrained curve drawing:

```
1. User draws curve in screen space
2. Project curve onto camera plane
3. Ray-cast to target surface
4. Snap curve CVs to surface
5. Optionally smooth along surface normals
```

---

## 3. The Camera-Centric Approach

### 3.1 Philosophy

Traditional 3D modeling requires the artist to think in three dimensions simultaneously. The camera-centric approach **collapses the problem to 2D** by:

1. **Fixing the view**: The camera defines a reference frame
2. **Drawing in screen space**: Natural 2D input
3. **Inferring depth**: From context, constraints, or surface projection
4. **Generating 3D**: Mathematical projection from 2D to 3D

This is precisely the paradigm that modern **Neural Sketch Field** systems adopt—the framework anticipated these developments by over a decade.

### 3.2 Advantages

| Advantage | Traditional Modeling | Camera-Centric |
|-----------|---------------------|----------------|
| **Input modality** | 3D manipulation | 2D sketching |
| **Learning curve** | Steep | Natural |
| **Speed** | Vertex-by-vertex | Stroke-based |
| **Iteration** | Destructive | Constructive |
| **Symmetry** | Manual | Automatic (bilateral mode) |

### 3.3 Implementation in MEL

The framework implements camera-centric modeling through:

```mel
// Core workflow
global proc MoveZCURVEModelingCAM(string $EdgeCurves[], string $ConeLocator[])
{
    // 1. Get camera parameters
    float $camPos[] = `xform -q -ws -t $camera`;
    float $camDir[] = nurbsViewDirectionVectorCam($camera, 1);

    // 2. For each curve point
    for ($curve in $EdgeCurves) {
        // Project to camera plane
        vector $projected = VecPointsToCameraPlane($curvePoints);

        // Set depth based on reference
        float $depth = computeDepthFromContext();

        // Reconstruct 3D
        vector $final = projectBackTo3D($projected, $depth);

        // Update curve
        curve -r -p $final[0] $final[1] $final[2];
    }
}
```

---

## 4. Key Findings from Procedure Analysis

### 4.1 Array-Centric Architecture

**565 procedures (24.5%)** are dedicated to array manipulation, revealing a fundamental design decision: **geometry is processed in batches**.

#### Why Arrays Dominate

| Pattern | Count | Example Procedures |
|---------|-------|-------------------|
| Batch transformation | 120+ | `VecPointsToCameraPlane`, `TransformFloatArray` |
| Sorting/filtering | 80+ | `SortFloatArrayAndString`, `FilterArrayByCondition` |
| Type conversion | 60+ | `FloatArrayToStringArray`, `VectorToFloatArray` |
| Index manipulation | 100+ | `RemoveVecAtIndex`, `IncludeStringAtIndex` |

#### MEL Array Patterns

```mel
// Pattern 1: Synchronized parallel arrays
float $distances[];
string $objects[];
// Sort both by distance
SortFloatArrayAndString($distances, $objects);

// Pattern 2: Index-based filtering
int $indices[] = getValidIndices($data);
float $filtered[] = extractByIndices($data, $indices);

// Pattern 3: Type bridging
string $nodes[] = `ls -sl`;
float $positions[] = getPositionsAsFloats($nodes);
vector $vectors[] = floatsToVectors($positions);
```

### 4.2 Duplicate Analysis

**1,559 duplicate procedure names** across files reveal:

| Top Duplicates | Occurrences | Consolidation Strategy |
|----------------|-------------|------------------------|
| `ArcLengthArray` | 10 | Single version in array-utils |
| `AddFloats` | 7 | Merge into parameterized function |
| `AppendFloatsZ` | 6 | Consolidate Z-variants |
| `Circle3Pt*` | 5 variants | Single `Circle3Point(mode)` |
| `TangentPointCircles*` | 4 variants | Single `TangentPointToCircle(returnType)` |

### 4.3 Naming Conventions Discovered

| Suffix | Meaning | Example |
|--------|---------|---------|
| `*Z` | Z-plane focused | `Circle3PtZFloats` |
| `*Vec` | Vector-based I/O | `TangentPointCirVectors` |
| `*Float` | Float array input | `PointsGetDistanceFLOAT` |
| `*String` | String array input | `FloatPointsToCamPlane` |
| `*2`, `*3` | Iterative improvements | `MoveZCURVEModelingCAM2010` |
| `*TF` | Returns True/False | `IScircleTF` |
| `*B` | Variant B implementation | `Circle3PtZB` |

### 4.4 Procedural vs. Global State Architecture

The framework uses a **hybrid state model**:

#### Procedures with Explicit State (Recommended)
```mel
global proc float[] Circle3PtZFloats(float $p1[], float $p2[], float $p3[])
{
    // All state is passed as parameters
    // Pure function: same inputs → same outputs
    return computeCircle($p1, $p2, $p3);
}
```

#### Procedures with Global State (Legacy Pattern)
```mel
global float $g_CameraPosition[];
global string $g_ActiveSurface;

global proc void PointToCameraPlane(string $ObjectLocZx)
{
    // Depends on global state
    // Must be set externally before calling
    float $projected[] = projectToPlane($ObjectLocZx, $g_CameraPosition);
    // ...
}
```

---

## 5. Global Variable Usage Analysis

### 5.1 Patterns Observed

The codebase uses global variables for:

| Pattern | Example | Purpose |
|---------|---------|---------|
| **Session state** | `$g_CurrentCamera` | Active viewport camera |
| **Cached computation** | `$g_LastProjectionMatrix` | Avoid recomputation |
| **Cross-procedure communication** | `$g_SelectedCurves[]` | Share selection between procs |
| **Configuration** | `$g_Tolerance` | User-adjustable thresholds |

### 5.2 Arguments For Global Variables

| Advantage | Explanation |
|-----------|-------------|
| **Performance** | Avoid passing large arrays through call stack |
| **Simplicity** | Reduce parameter count in complex pipelines |
| **Session persistence** | State survives between user actions |
| **MEL limitation workaround** | MEL lacks structs; globals simulate fields |
| **Tool integration** | Maya's architecture expects global state |

### 5.3 Arguments Against Global Variables

| Disadvantage | Impact |
|--------------|--------|
| **Hidden dependencies** | Procedure behavior depends on unseen state |
| **Testing difficulty** | Cannot isolate procedures for unit tests |
| **Reentrancy failure** | Nested calls corrupt shared state |
| **Namespace pollution** | Risk of name collisions across scripts |
| **Thread unsafety** | Maya's multi-threaded evaluation breaks |
| **Debugging complexity** | "Spooky action at a distance" bugs |

### 5.4 Recommended Refactoring

For SST integration, global state should be **encapsulated in context objects**:

```python
# Instead of global variables
class SketchContext:
    def __init__(self, camera: Camera, surface: Surface = None):
        self.camera = camera
        self.surface = surface
        self.tolerance = 0.01
        self.cached_projection = None

    def project_to_camera_plane(self, points: List[Vec3]) -> List[Vec2]:
        # State is explicit and encapsulated
        return self.camera.project(points)

# Usage
ctx = SketchContext(get_active_camera())
projected = ctx.project_to_camera_plane(curve_points)
```

---

## 6. Meta-Organizational Tools

### 6.1 FindNameOfVariables

The `FindNameOfVariables` procedure is a **meta-tool** that catalogs MEL scripts:

```mel
global proc string[] FindNameOfVariables(string $pattern, int $sortByFirst)
{
    // Scans loaded procedures for naming patterns
    // Returns sorted list of matching procedure names

    // This procedure was used to create the organizational
    // structure that this abstract documents
}
```

This represents an early form of **program introspection**—the code analyzing itself for organizational purposes.

### 6.2 Extraction and Cataloging Pipeline

The modern Python extraction tools (`extract_procedures.py`, `generate_catalog.py`) extend this concept:

```
MEL Files → Regex Extraction → JSON Database → Markdown Catalog
                                    ↓
                            Category Classification
                                    ↓
                            SST Layer Mapping
                                    ↓
                            Duplicate Detection
```

### 6.3 Automated Organization

The extraction revealed:
- **68 files** containing MEL procedures
- **8 categories** based on naming patterns
- **3 SST layers** (affine, conformal, spectral)
- **436 consolidation candidates** (procedures appearing in multiple files)

---

## 7. Comparative Analysis to Industry

### 7.1 vs. ZBrush Curve Mode

| Feature | This Framework | ZBrush |
|---------|---------------|--------|
| **Curve source** | Camera-projected strokes | Surface-constrained strokes |
| **Depth inference** | Camera plane + surface projection | Always on surface |
| **Circle/tangent** | Full conformal geometry toolkit | Limited |
| **Programmability** | MEL scripts | ZScript (limited math) |

### 7.2 vs. Blender Grease Pencil

| Feature | This Framework | Grease Pencil |
|---------|---------------|---------------|
| **Primary use** | Geometry construction | Annotation + 2D animation |
| **Mathematical basis** | Conformal geometry | Bezier curves |
| **3D projection** | Camera plane with depth | Layer-based |
| **Extensibility** | MEL procedures | Python + C |

### 7.3 vs. Modern Neural Sketch Systems

| Feature | This Framework (Legacy) | Neural Sketch Field (Modern) |
|---------|------------------------|------------------------------|
| **Surface prediction** | Rule-based projection | FNO inference |
| **Uncertainty** | None (deterministic) | Confidence-weighted ghosts |
| **Learning** | None | Adapts to user style |
| **Resolution** | Fixed (polygon count) | Resolution-independent |

The key insight: **this framework's camera-centric approach anticipated the Neural Sketch Field paradigm**. The modern system extends it with:
- Learned surface prediction (vs. explicit projection)
- Uncertainty quantification (vs. deterministic output)
- Resolution independence (vs. fixed tessellation)

---

## 8. Evolution Visible in the Code

### 8.1 Versioning Through Naming

The codebase shows evolution through procedure naming:

| Generation | Pattern | Example | Era |
|------------|---------|---------|-----|
| First | Simple names | `MakeCIRCLE` | Early development |
| Second | Z-plane variants | `Circle3PtZ` | Z-focused workflow |
| Third | Type-specific | `Circle3PtZFloats` | Float array optimization |
| Fourth | Iterative | `Circle3PtZFloatsI` | Performance iteration |
| Fifth | Year-stamped | `MoveZCURVEModelingCAM2010` | Maya version compatibility |

### 8.2 Optimization Trajectory

```
String arrays → Float arrays → Vector arrays
     ↓               ↓              ↓
  Slow          Faster         Fastest
(type conversion) (direct) (Maya native)
```

The migration from string-based to vector-based procedures shows performance optimization over time.

### 8.3 Consolidation Patterns

Multiple implementations of the same algorithm reveal:
1. **Initial implementation** (often verbose)
2. **Optimized variant** (reduced operations)
3. **Specialized variant** (specific use case)
4. **Final version** (marked "True" or "the real")

---

## 9. Future Direction: SST Integration

### 9.1 Mapping to SST Architecture

The legacy MEL procedures map directly to SST node types:

| MEL Category | SST Node Type | Transformation |
|--------------|---------------|----------------|
| matrix | `TransformNode` | Direct translation |
| circle | `MutationNode` (conformal) | Lift to Möbius |
| tangent | `MutationNode` (conformal) | Lift to Möbius |
| sketch | `HybridNode` (conformal/spectral) | Camera → Field |
| curve | `MutationNode` (conformal) | NURBS → Biarc |
| polygon | `MutationNode` (affine) | Mesh operations |

### 9.2 The Tri-Space Engine

Legacy procedures become **layers in the Tri-Space Engine**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRI-SPACE ENGINE MAPPING                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   AFFINE LAYER (GL(4,ℝ))                                                   │
│   ──────────────────────                                                    │
│   Legacy: matrix procedures (xyzRotation, GetRotationFromDirection)        │
│   SST: Matrix stack, Platform transforms                                   │
│                                                                             │
│   CONFORMAL LAYER (PSL(2,ℂ))                                               │
│   ────────────────────────                                                  │
│   Legacy: circle/tangent procedures (Circle3Point, PointToCircleTangents)  │
│   SST: Möbius composition, Biarc curves, Conformal maps                    │
│                                                                             │
│   SPECTRAL LAYER (L²(ℝ³))                                                  │
│   ─────────────────────                                                     │
│   Legacy: None (anticipated by sketch procedures)                          │
│   SST: FNO inference, Neural implicits, Spectral fields                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.3 Neural Sketch Field Integration

The legacy camera-centric approach becomes the **input layer** for Neural Sketch Field:

```
Legacy Pipeline:
    Sketch → Camera Projection → Surface Constraint → Polygons

Neural Sketch Field Pipeline:
    Sketch → Camera Projection → FNO Prediction → Spectral Field → Mesh
                                      ↑
                            Latent Priming (from legacy surface)
```

The `PointToCameraPlane` family becomes the **encoder** for the neural anticipator.

### 9.4 Supernode Compilation

Legacy procedures become **compilation targets** in the Supernode:

```python
class LegacyMELToSupernode:
    """
    Compile legacy MEL procedures into Supernode operations.
    """

    def compile_circle3point(self, p1, p2, p3) -> SupernodeOp:
        """
        Legacy: Circle3PtZFloats(p1[], p2[], p3[])
        Supernode: BiarcApprox with circle constraint
        """
        # Compute circle in conformal space
        center, radius = solve_circle_3pt(p1, p2, p3)

        # Return as Supernode operation
        return BiarcCircle(center, radius, plane=self.active_plane)

    def compile_tangent(self, circle, point) -> SupernodeOp:
        """
        Legacy: PointToCircleTangents(radius, circlePos, pointPos)
        Supernode: ConformalWarp with tangent constraint
        """
        # Solve in conformal space
        t1, t2 = solve_tangent_points(circle, point)

        # Return as Supernode operations
        return [BiarcLine(point, t1), BiarcLine(point, t2)]
```

---

## 10. Conclusion

### 10.1 What This Framework Achieved

1. **Camera-centric modeling** ahead of its time
2. **Conformal geometry** as a practical tool (not just theory)
3. **Batch processing** architecture for performance
4. **Meta-organizational** tools for code management

### 10.2 What It Anticipated

1. **Neural Sketch Field**: The camera-projection paradigm
2. **Resolution independence**: The desire to edit "math, not meshes"
3. **Topological awareness**: Circle/tangent constraints as topology hints
4. **Anticipatory modeling**: The retopology pipeline as proto-prediction

### 10.3 The Path Forward

```
Legacy MEL → Python Translation → SST Integration → Neural Enhancement
     ↓              ↓                   ↓                  ↓
  2,308 procs   math_core.py      HybridState        FNO + G-CNN
```

The legacy procedures become **training data** and **reference implementations** for the mathematical compiler. The camera-centric approach becomes the **input modality** for neural anticipation. The conformal geometry becomes the **constraint language** for topological guarantees.

---

## Appendix A: Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| GL(4, ℝ) | General Linear Group (4×4 invertible real matrices) |
| SO(3) | Special Orthogonal Group (3×3 rotation matrices) |
| PSL(2, ℂ) | Projective Special Linear Group (Möbius transforms) |
| L²(ℝ³) | Square-integrable functions on ℝ³ |
| S² | Unit 2-sphere |
| ⊥ | Perpendicular |
| · | Dot product |
| × | Cross product |
| ∘ | Function composition |
| n̂ | Unit normal vector |
| |v| | Vector magnitude |

## Appendix B: Key Procedure Reference

### Circle Operations

| Procedure | Input | Output | Complexity |
|-----------|-------|--------|------------|
| `Circle3PtZFloats` | 3 points (floats) | center + radius | O(1) |
| `IntersectTwoCircles` | 2 circles | 0-2 points | O(1) |
| `PointInCircle` | point, circle | boolean | O(1) |
| `CurvatureIsCircle` | curve, steps | bool + data | O(n) |

### Tangent Operations

| Procedure | Input | Output | Complexity |
|-----------|-------|--------|------------|
| `PointToCircleTangents` | point, circle | 2 tangent lines | O(1) |
| `TangentCircles` | 2 circles | 4 tangent lines | O(1) |
| `TangentCircleBetweenCircle` | 2 circles, radius | tangent circle | O(1) |

### Camera Operations

| Procedure | Input | Output | Complexity |
|-----------|-------|--------|------------|
| `PointToCameraPlane` | 3D point | 2D screen point | O(1) |
| `VecPointsToCameraPlane` | point array | 2D array | O(n) |
| `MoveZCURVEModelingCAM` | curves, camera | positioned curves | O(n) |

---

*This abstract documents a decade of practical geometric computation, translated into formal mathematics for integration with modern computational frameworks. The legacy code is not obsolete—it is a blueprint.*

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-06 | Initial comprehensive abstract |

