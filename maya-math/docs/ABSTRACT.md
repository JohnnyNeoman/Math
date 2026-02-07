# Abstract: Generative Geometry as Compositional Code

> **A Unified Framework for Algebraic 3D Assembly with Anticipatory Intelligence**
> Version 1.0 | Living Document | 2026-01-27

---

## Executive Summary

We present a framework for 3D geometry construction that treats **the mathematical specification as the program itself**. By establishing a minimal algebraic core—matrix stack traversal, portable symmetry platforms, and a functional L-system alphabet—we create a system where any valid composition of operations produces valid geometry. This compositional closure enables geometry to be "written" as strings, transpiled across platforms (Maya, Blender, Unreal), and extended with machine learning without architectural disruption.

The framework is designed for **degrees of freedom at the foundation**, enabling components to adhere to arbitrary surfaces, reference prior constructions, and encode complex assembly patterns as reusable rules. Future integration with neural operators, equivariant networks, and probabilistic geometry anticipation will transform this algebraic substrate into an anticipatory design partner.

---

## 1. The Core Insight: Math Layer as Generative Code

Traditional procedural geometry systems separate "code" from "data"—programs manipulate meshes as passive objects. Our approach inverts this relationship:

```
The specification IS the execution.
The tree IS the program.
The algebra IS the geometry.
```

We define a **Skeletal Singleton Tree (SST)**—a single hierarchical data structure that serves as both the scene graph and the generative grammar. The SST cleanly separates:

- **State** (the Matrix Stack): Accumulated transformation context—*where* and *how*
- **Mutation** (Geometry Operations): Position-agnostic operations—*what*

This separation yields a **closed algebra**: any composition of valid state and mutation operations produces valid geometry. The system is deterministic, invertible, and platform-agnostic until the final emission step.

---

## 2. Design Philosophy: Degrees of Freedom at the Foundation

The architecture prioritizes **adaptability over specificity**:

### 2.1 Surface Adhesion (Gram-Schmidt Alignment)
Any axiom can "grow" from any surface. The alignment operation constructs a stable orthonormal basis from a hit point and surface normal, enabling geometry to adhere to arbitrary topology without manual pivot adjustment.

### 2.2 Portable World Centers (Platform Symmetry)
Symmetry is not bound to world axes. A "Platform" establishes a local coordinate system that can be positioned and rotated arbitrarily. Reflection operations compose through the platform: `M = P × S × P⁻¹`. This enables nested, hierarchical symmetries—mirror-of-mirror resolves cleanly by algebraic composition.

### 2.3 Bifurcation as First-Class Pattern
The Mirror node doesn't duplicate geometry; it **bifurcates the execution stream**. Children execute once in the "real" branch and again in the reflected branch. This pattern generalizes to radial instancing, array-along-curve, and any replication scheme—all encoded as tree structure rather than imperative loops.

---

## 3. Compositional Power: Strings as Expressions

Because the node tree is algebraically closed, we can encode assembly patterns as **serializable rules**:

```yaml
root:
  - Platform: { id: "shoulder" }
  - Mirror: { axis: X }
    children:
      - Align: { surface: "torso", uv: [0.5, 0.8] }
      - Instance: { mesh: "arm_segment" }
      - Radial: { n: 6, axis: Z }
        children:
          - Instance: { mesh: "armor_plate" }
```

This YAML compiles to a node tree, executes via the Walker, and emits platform-specific geometry. The same specification runs in Maya, Blender, or Unreal—only the `emit()` binding changes.

### 3.1 Rule Hardcoding
Once a pattern proves useful (e.g., "bilateral limb with radial armor"), it can be **hardcoded as a named production rule**. The rule becomes a first-class node that expands into its constituent operations. This is the L-system's rewriting mechanism applied to 3D assembly.

### 3.2 Transpilation as Serialization
The SST is the canonical representation. Transpilers for Maya/Blender/Unreal are thin mappings from node operations to platform APIs. Adding a new platform means writing one emitter—the math layer remains untouched.

---

## 4. Future Trajectory: Anticipatory Intelligence

The algebraic foundation is designed to accept machine learning integration without architectural disruption. We envision three convergent capabilities:

### 4.1 Neural Operators for Sketch-Based Modeling
**Fourier Neural Operators (FNO)**, **Graph Neural Operators (GNO)**, and **Geometry-aware Neural Processes (GNP)** operate on continuous function spaces rather than discrete samples. Integrated into a sketch interface, they can:

- Infer 3D form from 2D strokes by learning the mapping from silhouette to volume
- Propagate edits through the SST by predicting downstream geometry changes
- Generalize across topology—trained on one class of shapes, applicable to others

The SST provides the **structural scaffold** these operators need: they predict mutations; the matrix stack places them.

### 4.2 Epipolar Geometry ML: Sculpt/Retopo Hybrid
Classical epipolar geometry constrains 3D reconstruction from multiple views. We propose a **fuzzy probabilistic extension**:

- The user sketches from multiple implicit viewpoints (front, side, ¾)
- ML models trained on epipolar constraints infer a **probability distribution over 3D form**
- The distribution is represented as a **Gaussian Tube** (for curves) or **Gaussian Surface** (for patches)
- As the user adds strokes, the distribution sharpens—uncertainty collapses into geometry

This "**Stereo-Photogrammetric Sketcher**" treats drawing as evidence accumulation. The system doesn't reconstruct a single mesh; it maintains a belief over possible meshes, allowing the user to guide convergence.

### 4.3 Ghost Scaffolding: Predictive Symmetry via G-CNNs
**Group-equivariant Convolutional Neural Networks (G-CNNs)** respect symmetry transformations by construction. Applied to the SST:

- When the user works in a Mirror node, the G-CNN predicts the **probability of the next operation** based on what's been placed so far
- High-probability predictions render as "ghost" scaffolding—translucent guides showing anticipated geometry
- The user can accept, reject, or refine the prediction

This transforms symmetry from a constraint into a **generative prior**. The system anticipates bilateral, radial, and hierarchical patterns because it has learned them from the structure of the SST itself.

### 4.4 Topological Guardrails via Persistent Homology
As geometry accumulates in the buffer, we can compute **Betti numbers** (topological invariants) to detect:

- Unexpected holes (disconnected components that should merge)
- Self-intersections (invalid manifold topology)
- Symmetry breaks (one side has features the other lacks)

These guardrails run as a background "Cruncher" thread, flagging issues before they propagate.

---

## 5. The Vision: Orchestrating Geometry

The end state is a system where:

1. **The math layer** provides deterministic, composable, platform-agnostic operations
2. **The rule layer** encodes reusable assembly patterns as production rules
3. **The ML layer** anticipates user intent, proposes scaffolding, and sharpens uncertainty into form
4. **The topological layer** validates output, ensuring manifold integrity and symmetry coherence

The user operates at the level of **intent**—"I want a bilateral arm with radial armor"—and the system handles placement, symmetry, and even suggests detail based on learned priors. The SST remains the single source of truth; ML predictions are just proposed nodes awaiting acceptance.

```
User Intent  →  SST (algebraic core)  →  ML Anticipation  →  Geometry Buffer
                      ↑                        │
                      └────── feedback ────────┘
```

---

## 6. Summary of Contributions

| Layer | Contribution | Status |
|-------|--------------|--------|
| **Math Foundation** | Gram-Schmidt alignment, platform reflection algebra | ✅ Documented |
| **State Schema** | Matrix stack with push/pop scope isolation | ✅ Documented |
| **Mutation Schema** | Primitives, CSG, deformations | ✅ Documented |
| **Node Algebra** | Functional Σ, compositional closure, generative property | ✅ Documented |
| **Transpiler Spec** | Maya/Blender/Unreal mappings | 🔲 Skeleton |
| **Neural Operators** | FNO/GNO/GNP for sketch→form | 🔲 Planned |
| **Epipolar ML** | Probabilistic multi-view reconstruction | 🔲 Planned |
| **G-CNN Scaffolding** | Predictive symmetry, ghost geometry | 🔲 Planned |
| **Topological Validation** | Persistent homology guardrails | 🔲 Planned |

---

## 7. Closing Remark

This framework began as a question: *What is the minimal algebraic core that enables arbitrary 3D assembly?* The answer—matrix stacks, portable platforms, and a functional alphabet—turned out to be small enough to fit on one page, yet powerful enough to compose into any geometry.

The beauty is in the closure: **strings become expressions that become geometry that become new strings**. The system is self-similar, compositional, and ready to accept intelligence. When we add neural operators and predictive symmetry, we're not bolting on features—we're filling in slots the algebra always had room for.

The math layer is the DNA. The nodes are the ribosomes. The geometry is the organism. And soon, the ML will be the environment that guides its growth.

---

*This document will be updated as the framework evolves. Each section marked 🔲 Planned will be promoted to ✅ Documented as implementation proceeds.*

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-27 | Initial abstract: core architecture + ML vision |

