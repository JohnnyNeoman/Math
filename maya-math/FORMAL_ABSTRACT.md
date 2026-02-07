# Batch-Centric Procedural Geometry: A Camera-Projection Framework for Sketch-Based 3D Modeling

> **A Formal Analysis of Procedural Geometry Architecture**
> **Technical Report on Development Methodology and Mathematical Foundations**

---

## Abstract

Three-dimensional modeling presents a persistent cognitive barrier: artists must simultaneously reason about form in all spatial dimensions while manipulating discrete geometric primitives. This paper presents a procedural geometry framework comprising **2,308 procedures** across **68 implementation files**, developed to address this challenge through a **camera-centric projection architecture**. The framework treats sketch input as a first-class geometric primitive, collapsing the 3D authoring problem to 2D screen-space interaction through systematic camera-plane projection.

Analysis of the procedural corpus reveals a **batch-processing paradigm** where 54.5% of procedures (1,258 of 2,308) constitute data transformation infrastructure, enabling geometry to be processed as collections rather than individual primitives. The mathematical foundation integrates **linear algebra** (GL(4,ℝ) matrix operations) with **conformal geometry** (PSL(2,ℂ) circle and tangent constructions), providing 27% of the procedural vocabulary for geometric computation.

The framework anticipates contemporary developments in neural sketch-based modeling, with its camera-projection pipeline corresponding directly to the encoder stage of modern Fourier Neural Operator systems. We present a formal analysis of the architectural decisions, development patterns observable through procedural naming archaeology, and the mapping from legacy categories to the proposed Tri-Space Engine (Affine × Conformal × Spectral). The 749 unique procedures represent a coherent computational geometry library where infrastructure enables transformation, mathematics defines primitives, and projection bridges 2D input to 3D output.

**Keywords**: procedural geometry, sketch-based modeling, conformal geometry, batch processing, camera projection, computational geometry

---

## 1. Introduction

### 1.1 The Cognitive Barrier in 3D Modeling

Traditional three-dimensional modeling tools require artists to simultaneously maintain mental models of form across all spatial axes while manipulating discrete geometric elements—vertices, edges, and faces. This cognitive load creates a barrier to entry and limits the speed at which experienced artists can iterate on designs. The fundamental tension lies between the **continuous nature of artistic intent** and the **discrete nature of polygonal representation**.

### 1.2 The Camera-Centric Insight

The framework presented in this analysis emerges from a fundamental insight: **geometric authoring is most natural when performed in screen space**. By fixing the viewpoint through a camera reference frame, the 3D modeling problem collapses to a 2D sketching interaction. The system then infers depth and form through a combination of explicit projection mathematics and surface constraints.

This insight manifests in procedure design:

```
Traditional Pipeline:
    Artist Intent → 3D Manipulation → Vertex-by-Vertex Editing → Mesh

Camera-Centric Pipeline:
    Artist Intent → 2D Sketch → Camera Projection → Depth Inference → Mesh
```

### 1.3 Contribution

This paper presents a formal analysis of the procedural geometry framework, examining:

1. The **batch-processing paradigm** that dominates the architectural design
2. The **mathematical foundations** spanning linear algebra and conformal geometry
3. The **development patterns** observable through naming conventions and procedure evolution
4. The **anticipatory architecture** that aligns with modern neural sketch systems

---

## 2. The Batch-Processing Paradigm

### 2.1 Empirical Distribution

Analysis of 2,308 procedures across 68 implementation files reveals the following categorical distribution:

| Category | Procedure Count | Percentage | Functional Role |
|----------|-----------------|------------|-----------------|
| utility | 693 | 30.0% | Maya API wrappers, selection management, cleanup |
| array | 565 | 24.5% | Data structure manipulation, type conversion, sorting |
| curve | 389 | 16.9% | NURBS curve operations, arc length computation |
| matrix | 160 | 6.9% | Rotation, transformation, linear algebra |
| polygon | 149 | 6.5% | Face, edge, and vertex operations |
| circle | 128 | 5.5% | Circle construction, intersection, containment |
| sketch | 118 | 5.1% | Camera projection, surface constraint |
| tangent | 106 | 4.6% | Tangent line and circle construction |

The dominance of infrastructure categories (utility + array = 54.5%) reveals a fundamental architectural decision: **geometry is processed as collections, not individual elements**.

### 2.2 Rationale for Batch Architecture

The batch-processing paradigm emerges from multiple convergent pressures:

**Language Constraints**: The MEL scripting environment lacks native support for compound data types (structures, classes). This limitation necessitates the use of parallel arrays to represent multi-attribute entities:

```mel
// Synchronized parallel arrays maintain implicit relationships
float $positions[];      // Position data
string $objectNames[];   // Corresponding Maya node names
int $indices[];          // Original ordering
```

**Performance Optimization**: Each call crossing the boundary between MEL script and Maya's internal API incurs overhead. Batch processing minimizes these crossings by operating on collections:

```mel
// Inefficient: N API calls
for ($i = 0; $i < size($objects); $i++) {
    float $pos[] = `xform -q -ws -t $objects[$i]`;
    // process single element
}

// Efficient: 1 API call + MEL iteration
float $allPositions[] = `xform -q -ws -t $objects`;
// process collection
```

**Geometric Coherence**: Geometric operations naturally apply to sets of related primitives. A curve is not a single point but a collection of control vertices; a surface is not a single face but a mesh of interconnected elements.

### 2.3 The Synchronized Array Pattern

A recurring architectural pattern emerges from the constraint of parallel arrays: **synchronized ordering**. When multiple arrays represent different attributes of the same entities, their ordering must remain consistent through all transformations.

This requirement produces a family of specialized sorting procedures:

| Procedure | Function |
|-----------|----------|
| `SortFloatArrayAndString` | Sort float array, apply same permutation to string array |
| `NewArrayOrderWithIndexKey` | Reorder array by explicit index mapping |
| `ReverseStringArray` | Reverse ordering while maintaining correspondence |

The frequency of these procedures (appearing in multiple files with variations) indicates the centrality of synchronized arrays to the framework's operation.

---

## 3. Mathematical Foundations

The procedural corpus encodes two primary mathematical frameworks: **affine geometry** for spatial transformations and **conformal geometry** for shape-preserving curve operations.

### 3.1 Affine Layer: GL(4,ℝ)

Matrix operations constitute 160 procedures (6.9% of the corpus), implementing the General Linear Group of 4×4 invertible real matrices.

**Transformation Matrix Structure**:

```
M ∈ GL(4, ℝ):

    ┌                          ┐
    │  R₁₁  R₁₂  R₁₃  Tₓ      │
M = │  R₂₁  R₂₂  R₂₃  Tᵧ      │
    │  R₃₁  R₃₂  R₃₃  Tᵤ      │
    │   0    0    0    1       │
    └                          ┘

Where R ∈ SO(3) (rotation) and T ∈ ℝ³ (translation)
```

**Euler Rotation Composition**: The `xyzRotation` procedure family implements rotation through Euler angle composition:

```
R(α, β, γ) = Rz(γ) · Ry(β) · Rx(α)
```

Where each component rotation matrix follows the standard form:

```
        ┌                      ┐
Rx(α) = │  1     0       0     │
        │  0   cos α  -sin α   │
        │  0   sin α   cos α   │
        └                      ┘
```

**Gram-Schmidt Orthonormalization**: The `GetRotationFromDirection` procedure constructs a rotation matrix from a direction vector and up reference using Gram-Schmidt orthonormalization, enabling geometry to "adhere" to arbitrary surfaces by computing stable local coordinate frames from hit points and surface normals.

### 3.2 Conformal Layer: PSL(2,ℂ)

Circle and tangent operations constitute 234 procedures (10.1%), operating in the conformal geometric framework where the fundamental invariant is **angle preservation**.

**Möbius Transformations**: The Projective Special Linear Group provides the algebraic structure:

```
f(z) = (az + b) / (cz + d)

Where a, b, c, d ∈ ℂ and ad - bc ≠ 0

Properties:
- Circles map to circles (including lines as infinite-radius circles)
- Angles are preserved at intersection points
- Cross-ratio is invariant
```

**Three-Point Circle Construction**: The `Circle3Point` procedure family computes the unique circle through three non-collinear points via perpendicular bisector intersection:

Given points P₁, P₂, P₃ ∈ ℝ²:

1. Construct perpendicular bisector L₁₂ of segment P₁P₂
2. Construct perpendicular bisector L₂₃ of segment P₂P₃
3. Compute center C = L₁₂ ∩ L₂₃
4. Compute radius r = |C - P₁|

The algebraic solution reduces to a 2×2 linear system:

```
┌                                    ┐ ┌     ┐   ┌                  ┐
│ 2(P₂.x - P₁.x)  2(P₂.y - P₁.y)    │ │ C.x │   │ |P₂|² - |P₁|²    │
│                                    │ │     │ = │                  │
│ 2(P₃.x - P₂.x)  2(P₃.y - P₂.y)    │ │ C.y │   │ |P₃|² - |P₂|²    │
└                                    ┘ └     ┘   └                  ┘
```

**Point-to-Circle Tangent Construction**: The `PointToCircleTangents` procedure computes tangent lines from an external point P to a circle C(O, r):

Given external point P where |OP| > r:

1. Compute distance d = |OP|
2. Compute tangent length t = √(d² - r²)
3. Compute tangent angle θ = arctan(r/t)
4. Tangent points: T₁,₂ = O + r · rotate(normalize(P - O), ±θ)

**Circle-Circle Intersection**: The `IntersectTwoCircles` procedure computes intersection points via the radical axis:

Given circles C₁(O₁, r₁) and C₂(O₂, r₂) with d = |O₂ - O₁|:

Intersection exists when |r₁ - r₂| ≤ d ≤ r₁ + r₂

```
a = (r₁² - r₂² + d²) / (2d)
h = √(r₁² - a²)
P = O₁ + a · normalize(O₂ - O₁)
I₁,₂ = P ± h · perpendicular(normalize(O₂ - O₁))
```

### 3.3 Camera Projection Layer

The sketch procedures (118, 5.1%) implement the core camera-centric architecture through screen-to-world transformation.

**Camera Basis Construction**:

Given camera position C ∈ ℝ³, view direction V̂ ∈ S², and up vector Û ∈ S²:

```
X̂ = normalize(V̂ × Û)    (right axis)
Ŷ = normalize(X̂ × V̂)    (up axis)
Ẑ = V̂                    (forward axis)
```

**Screen-to-World Projection**:

For sketch point p ∈ ℝ² (screen space) at depth d:

```
P₃D = C + d·Ẑ + p.x·X̂ + p.y·Ŷ
```

Key procedures in the projection pipeline:

| Procedure | Mathematical Operation |
|-----------|------------------------|
| `nurbsViewDirectionVectorCam` | Extract V̂ from camera node |
| `PointToCameraPlane` | p₂D = (P - C) · [X̂, Ŷ]ᵀ |
| `VecPointsToCameraPlane` | Batch projection over point array |
| `MoveZCURVEModelingCAM` | Set curve depth in camera space |

---

## 4. Development Patterns

Analysis of the procedural corpus reveals systematic development patterns encoded in naming conventions and procedure evolution.

### 4.1 Iterative Refinement Through Naming

Procedure names encode evolutionary history through consistent suffix conventions:

| Generation | Pattern | Interpretation |
|------------|---------|----------------|
| 1st | `MakeCIRCLE` | Initial implementation, general purpose |
| 2nd | `Circle3PtZ` | Axis-specific variant (Z-plane operation) |
| 3rd | `Circle3PtZFloats` | Type-optimized (float array input) |
| 4th | `Circle3PtZFloatsI` | Performance iteration |
| 5th | `MoveZCURVEModelingCAM2010` | Version-stamped for compatibility |

This naming archaeology reveals an optimization trajectory driven by practical use:

```
Initial implementation (correctness)
    ↓
Axis specialization (workflow alignment)
    ↓
Type optimization (performance)
    ↓
Further iteration (refinement)
    ↓
Version stamping (compatibility maintenance)
```

### 4.2 Type Migration Trajectory

A clear migration pattern emerges in data representation:

```
String arrays (maximum flexibility, minimum performance)
    ↓
Float arrays (direct numeric access, moderate performance)
    ↓
Vector arrays (Maya-native type, maximum performance)
```

This trajectory reflects the tension between development convenience (strings are easily printed and debugged) and runtime efficiency (vectors are Maya's native geometric type).

### 4.3 Naming Convention Semantics

Systematic suffix conventions encode procedure characteristics:

| Suffix | Semantic Meaning | Example |
|--------|------------------|---------|
| `*Z` | Z-plane operation | `Circle3PtZFloats` |
| `*Vec` | Vector-based I/O | `TangentPointCirVectors` |
| `*Float` | Float array input | `PointsGetDistanceFLOAT` |
| `*String` | String array input | `FloatPointsToCamPlane` |
| `*TF` | Boolean return | `IScircleTF` |
| `*B`, `*2`, `*3` | Variant implementations | `Circle3PtZB` |
| `*I` | Iterative/improved version | `Circle3PtZFloatsI` |

These conventions function as a **type system encoded in names**, compensating for MEL's weak typing through explicit naming discipline.

### 4.4 Consolidation Archaeology

The presence of 1,559 duplicate procedure instances across 749 unique names reveals a development pattern of **local copy and modification**:

| Duplicate Family | Occurrences | Interpretation |
|------------------|-------------|----------------|
| `ArcLengthArray` | 10 | Frequently needed, copied to avoid dependencies |
| `AddFloats` | 7 | Utility function replicated for local modification |
| `AppendFloatsZ` | 6 | Z-axis variant proliferated across files |

This pattern emerges from the absence of a formal module system: when a procedure is needed, the pragmatic solution is to copy it locally rather than manage cross-file dependencies.

---

## 5. State Management Architecture

The framework exhibits a **hybrid state model** balancing functional purity with session awareness.

### 5.1 Explicit State (Functional Procedures)

A subset of procedures follow functional design principles:

```mel
global proc float[] Circle3PtZFloats(float $p1[], float $p2[], float $p3[])
{
    // All state is passed as parameters
    // Deterministic: same inputs → same outputs
    return computeCircle($p1, $p2, $p3);
}
```

Characteristics:
- All inputs as parameters
- No external dependencies
- Composable and testable
- Suitable for library functions

### 5.2 Implicit State (Session-Aware Procedures)

Other procedures depend on externally configured state:

```mel
global float $g_CameraPosition[];
global string $g_ActiveSurface;

global proc void PointToCameraPlane(string $ObjectLocZx)
{
    // Depends on global configuration
    float $projected[] = projectToPlane($ObjectLocZx, $g_CameraPosition);
    // ...
}
```

Characteristics:
- Reduced parameter count
- Session persistence between operations
- Tool-like interaction model
- State must be configured before invocation

### 5.3 Global Variable Categories

Analysis reveals four primary categories of global state:

| Category | Example | Purpose |
|----------|---------|---------|
| **Session** | `$g_CurrentCamera` | Active viewport reference |
| **Cache** | `$g_LastProjectionMatrix` | Avoid redundant computation |
| **Communication** | `$g_SelectedCurves[]` | Cross-procedure data sharing |
| **Configuration** | `$g_Tolerance` | User-adjustable thresholds |

### 5.4 Tradeoff Analysis

The hybrid model represents a deliberate tradeoff:

**Advantages of Global State**:
- Performance (avoid stack passing of large arrays)
- Necessity (MEL lacks struct/class constructs)
- Integration (Maya's architecture expects persistent state)
- Convenience (reduces procedure signature complexity)

**Disadvantages of Global State**:
- Hidden dependencies (behavior depends on unseen configuration)
- Testing complexity (cannot isolate procedures without environment setup)
- Reentrancy failure (nested calls may corrupt shared state)
- Namespace collision (risk increases with codebase size)

The prevalence of both patterns suggests an evolved architecture where **core mathematical operations remain pure** while **tool integration layers manage session state**.

---

## 6. Anticipatory Architecture

The framework's design anticipates developments in neural sketch-based modeling that emerged subsequently.

### 6.1 Correspondence to Neural Sketch Field

The camera-centric pipeline corresponds directly to modern neural approaches:

| Framework Component | Neural Sketch Field Equivalent |
|---------------------|-------------------------------|
| Camera projection | Encoder (sketch → latent) |
| Rule-based depth inference | FNO surface prediction |
| Surface constraint | Latent priming from base geometry |
| Deterministic output | Confidence-weighted ghost scaffolding |

The critical insight—that sketch input should be processed through camera-plane projection—appears in both systems, validating the architectural approach.

### 6.2 Mapping to Tri-Space Engine

The procedural categories map directly to the proposed Tri-Space Engine architecture:

| Legacy Category | SST Layer | Algebraic Structure |
|-----------------|-----------|---------------------|
| matrix, utility, polygon | Affine | GL(4,ℝ) |
| circle, tangent, curve | Conformal | PSL(2,ℂ) |
| (anticipated) | Spectral | L²(ℝ³) |

The absence of spectral-layer procedures in the legacy corpus identifies the precise gap that neural operators (FNO, GNO) would fill: **resolution-independent surface prediction from boundary constraints**.

### 6.3 The Supernode Compilation Target

Legacy procedures become compilation targets for the Geometric Scaffold Supernode:

```
Legacy MEL Procedure → SST Node Operation

Circle3PtZFloats(p1, p2, p3) → BiarcCircle(center, radius, plane)
PointToCircleTangents(...)   → ConformalWarp(tangent_constraint)
MoveZCURVEModelingCAM(...)   → HybridNode(camera_projection)
```

This mapping preserves the mathematical semantics while enabling integration with neural anticipation systems.

---

## 7. Quantitative Summary

### 7.1 Corpus Statistics

| Metric | Value |
|--------|-------|
| Total procedures | 2,308 |
| Unique procedures | 749 |
| Duplicate instances | 1,559 |
| Implementation files | 68 |
| Unique rate | 32.4% |

### 7.2 Functional Distribution

| Functional Layer | Procedure Count | Percentage |
|------------------|-----------------|------------|
| Infrastructure (utility + array) | 1,258 | 54.5% |
| Geometric computation (curve + circle + tangent) | 623 | 27.0% |
| Transformation (matrix) | 160 | 6.9% |
| Mesh operations (polygon) | 149 | 6.5% |
| Projection pipeline (sketch) | 118 | 5.1% |

### 7.3 Mathematical Coverage

| Domain | Procedure Count | Key Operations |
|--------|-----------------|----------------|
| Linear Algebra | 160 | Rotation, transformation, reflection |
| Conformal Geometry | 234 | Circle, tangent, intersection |
| Projective Geometry | 118 | Camera projection, depth inference |

---

## 8. Conclusion

The procedural geometry framework analyzed in this paper represents a coherent computational approach to sketch-based 3D modeling. The architecture emerges from the intersection of practical constraints (MEL language limitations, Maya API design) and geometric insight (camera-centric projection collapses 3D to 2D).

The framework demonstrates:

1. **Batch-centric design** as a systematic response to scripting environment constraints, with 54.5% of procedures dedicated to collection-based data transformation

2. **Mathematical rigor** through integration of linear algebra (GL(4,ℝ)) and conformal geometry (PSL(2,ℂ)), providing 27% of the procedural vocabulary for geometric computation

3. **Camera-centric authoring** as a natural input modality, enabling artists to work in screen space while the system infers 3D form

4. **Iterative refinement** visible through naming archaeology, with clear optimization trajectories from initial implementation through type-specific variants

5. **Anticipatory architecture** that aligns with modern neural sketch systems, validating the core projection insight

The 2,308 procedures form a system where infrastructure enables transformation, mathematics defines primitives, and projection bridges 2D input to 3D output. This corpus serves not as legacy code to be deprecated but as a **blueprint for the integration of procedural geometry with neural anticipation**—the foundation upon which the Skeletal Singleton Tree framework, Neural Sketch Field, and Geometric Scaffold Supernode are constructed.

---

## References

### Mathematical Foundations
- **GL(4,ℝ)**: General Linear Group of 4×4 invertible real matrices
- **SO(3)**: Special Orthogonal Group (rotation matrices)
- **PSL(2,ℂ)**: Projective Special Linear Group (Möbius transformations)
- **L²(ℝ³)**: Square-integrable functions on ℝ³

### Framework Documents
- COMPREHENSIVE_ABSTRACT.md: Detailed procedure analysis
- PROCEDURE_CATALOG.md: Complete procedure listing with signatures
- NEURAL_SKETCH_FIELD.md: Anticipation loop architecture
- SUPERNODE_ABSTRACT.md: Tri-Modal mathematical compiler

---

*This analysis documents a computational geometry framework developed through practical application, formalized through mathematical analysis, and positioned for integration with neural anticipation systems.*
