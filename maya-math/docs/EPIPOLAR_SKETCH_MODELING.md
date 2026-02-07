# Geometry-Aware Latent Diffusion for Epipolar Sketch-Based Modeling

> **Bridging the Geometric Scaffold Supernode to Semantically-Aware 3D Synthesis**
> A Dirac-Informed Latent Space with Cross-Attention Guidance and Wasserstein Coherence
> Version 1.0 | 2026-01-28

---

## Abstract

We present a unified framework that bridges the **Geometric Scaffold Supernode** (GSS)—a tri-modal mathematical compiler for lossless geometry synthesis—with a **geometry-aware latent diffusion model** trained on intrinsic geometric representations derived from the **Dirac operator**. The core insight is that the Dirac operator's spectrum encodes intrinsic shape information (curvature, topology, symmetry) in a coordinate-free manner; a latent space trained on Dirac-spectral embeddings of a shape library thus becomes *semantically aware of geometry itself*—it "knows" what valid, coherent shapes look like at a structural level, not merely as point clouds or voxels.

We extend the GSS tri-modal state `X = X_aff × X_conf × X_spec` with a fourth **Semantic-Geometric context** `X_sem` residing in this Dirac-informed latent space. User intent—expressed through natural language, procedural rules, or partial sketches—is encoded as a **prior distribution** over `X_sem`. As the user draws, each stroke acts as an **observation** that updates this distribution via Bayesian conditioning, progressively collapsing the latent toward compositional structures consistent with both the sketch geometry and the learned geometric grammar.

The bridge between user input and latent geometry is realized through a **multi-scale U-Net with cross-attention**, where:
- **Queries** derive from spatial features of the user's sketch (stroke positions, curvatures, implied surfaces)
- **Keys/Values** derive from semantic tokens (intent descriptors, procedural rule names, geometric primitives)
- **Attention maps** at each resolution perform **spatial-semantic alignment**, binding tokens like "cylinder," "fillet," "symmetry-8" to specific regions of the evolving geometry

To ensure **coherence** across multiple constraint sources (semantic intent, epipolar consistency, symmetry, physical plausibility), we formulate latent updates as **Wasserstein barycenter** problems: each constraint induces a target distribution over latents, and the optimal update minimizes squared W₂ distance to all targets simultaneously. This yields globally coherent trajectories through latent space—the "transport skeleton"—which diffusion then refines into on-manifold geometry.

**Epipolar constraints** are introduced for multi-view sketch-to-3D inference: when the user draws corresponding strokes across two or more viewports, an **F-matrix head** enforces that lifted 3D curves satisfy the fundamental matrix constraint, triangulating consistent 3D structure from 2D observations. This enables a sketch-based workflow where rough strokes from multiple angles converge to a single, geometrically valid 3D model.

The resulting system implements **generative anticipation**: as the user sketches, the model continuously predicts likely completions, displays ghost scaffolds (biarc curves, conformal grids, predicted surfaces), and allows the user to "steer" toward preferred structures by accepting, rejecting, or refining suggestions. The compositional design grammar emerges from the Dirac-trained latent space—shapes are not generated from noise, but *navigated to* through a structured manifold of valid geometries.

This reframes interactive 3D authoring as **latent space navigation with geometric guarantees**: user strokes define a path through a semantically-organized geometry manifold, cross-attention binds intent to spatial regions, optimal transport ensures coherent transitions, and the GSS Lift–Operate–Collapse pipeline compiles the result into resolution-independent, topologically valid output.

---

## 1. The Four-Context Hybrid State

We extend the GSS tri-modal state to include a semantic-geometric context:

```
X := X_aff × X_conf × X_spec × X_sem
```

| Context | Mathematical Basis | Data Type | Role |
|---------|-------------------|-----------|------|
| **Affine** | SE(3), GL(4,ℝ) | Matrix4×4 | World placement, instancing |
| **Conformal** | PSL(2,ℂ) | Möbius transform | Angle-preserving maps, UVs |
| **Spectral** | L²(ℝ³), Fourier basis | Field coefficients | Resolution-independent physics |
| **Semantic-Geometric** | Dirac spectral embedding | Latent vector z ∈ ℝᵈ | Learned shape manifold |

The semantic-geometric context `X_sem` is not computed from the current geometry—it is a **latent code** that *generates* geometry when decoded. The relationship is:

```
z ∈ X_sem  →  (Decode)  →  u ∈ X_spec  →  (Collapse)  →  Mesh
```

But crucially, the mapping is **bidirectional**: user strokes can be *encoded* into `X_sem`, allowing navigation rather than pure generation.

---

## 2. The Dirac Operator: Geometry-Aware Latent Space

### 2.1 Why Dirac?

The **Dirac operator** `D` on a Riemannian manifold is a first-order differential operator acting on spinor fields. Its spectrum {λᵢ} encodes intrinsic geometric information:

- **Curvature**: The Lichnerowicz formula relates `D²` to scalar curvature
- **Topology**: The Atiyah-Singer index theorem links `ker(D)` to topological invariants
- **Symmetry**: Isometries of the manifold induce symmetries of `spec(D)`

Unlike point clouds (extrinsic, coordinate-dependent) or voxels (resolution-locked), the Dirac spectrum is:
- **Intrinsic**: Depends only on the shape, not its embedding
- **Global**: Captures both local curvature and global topology
- **Spectral**: Naturally compatible with Fourier-based neural operators

### 2.2 Training on a Geometry Library

Given a library of 3D shapes `{M₁, M₂, ..., Mₙ}`:

1. **Compute Dirac spectra**: For each shape, compute the first k eigenvalues and eigenfunctions of the Dirac operator (or its discrete approximation via finite elements)

2. **Build spectral embeddings**: Represent each shape as a vector of spectral invariants:
   ```
   e(M) = [λ₁, λ₂, ..., λₖ, ∫φ₁², ∫φ₂², ..., higher-order invariants]
   ```

3. **Train latent encoder/decoder**: A variational autoencoder maps shapes to latent codes:
   ```
   Encoder: M → z ∈ ℝᵈ
   Decoder: z → SDF (spectral coefficients)
   ```
   
4. **Semantic alignment**: Align latent space with semantic labels via contrastive learning:
   ```
   z("cylinder") close to z(actual cylinders in library)
   z("organic") close to z(organic shapes)
   ```

The result: a latent space where **geometric similarity ≈ latent proximity**, and where semantic concepts have consistent geometric meaning.

### 2.3 The Latent Manifold

The trained latent space `Z` is not a structureless blob—it has geometry:

- **Clusters**: Semantic categories (primitives, organic forms, mechanical parts)
- **Interpolation paths**: Geodesics in Z correspond to smooth shape morphs
- **Compositional structure**: Additive operations in Z may correspond to boolean unions

This manifold becomes the "prior" that makes generation coherent: random walks through Z produce valid shapes, not noise.

---

## 3. Cross-Attention: Spatial-Semantic Alignment

### 3.1 The U-Net Architecture

We employ a multi-resolution U-Net operating on the spectral field `u`:

```
Resolution levels: 64³ → 32³ → 16³ → 8³ → 16³ → 32³ → 64³
                   (encoder)              (decoder)
```

At each level ℓ, we insert **cross-attention** between:
- **Spatial features** `x^(ℓ)` from the current field/sketch
- **Semantic tokens** `T` from user intent (text, rule names, primitive labels)

### 3.2 Attention Mechanism

```
Queries:   Q = W_Q · x^(ℓ)       (from spatial patches)
Keys:      K = W_K · T           (from semantic tokens)
Values:    V = W_V · T           (from semantic tokens)

Attention: A^(ℓ) = softmax(QK^T / √d)
Update:    Δx^(ℓ) = A^(ℓ) · V
```

**Interpretation**:
- `A^(ℓ)` is a **token-to-patch correspondence map**
- Each token distributes influence over spatial regions
- Coarse levels (8³, 16³): tokens bind to global structure ("a sphere on the left")
- Fine levels (32³, 64³): tokens bind to local detail ("smooth fillet", "sharp edge")

### 3.3 Hierarchical Mosaic

The multi-resolution structure creates a **hierarchical mosaic**:

```
8³  level: Global layout, object count, rough positions
16³ level: Object shapes, major features, relations
32³ level: Surface details, edges, connections
64³ level: Fine texture, micro-features, crispness
```

Semantic guidance enters at ALL levels, but:
- **High-level concepts** (composition, semantics) dominate coarse levels
- **Low-level concepts** (texture, sharpness) dominate fine levels

This matches human design intent: "I want a mechanical assembly" (coarse) with "brushed metal finish" (fine).

---

## 4. Wasserstein Barycenters: Coherent Multi-Constraint Blending

### 4.1 The Coherence Problem

Multiple constraints compete to guide generation:
- **Semantic**: Text/intent tokens → target distribution μ_semantic
- **Geometric**: Symmetry/rules → target distribution μ_geometry  
- **Epipolar**: Multi-view consistency → target distribution μ_epipolar
- **Physical**: Plausibility prior → target distribution μ_physics
- **User sketch**: Stroke observations → target distribution μ_sketch

Naive blending (weighted average) produces incoherent results—features from different constraints conflict.

### 4.2 Barycentric Blending

Define the **Wasserstein barycenter** as:

```
μ* = argmin_μ Σᵢ wᵢ · W₂²(μ, μᵢ)
```

This finds the distribution that minimizes total squared optimal transport distance to all constraint targets.

**Key property**: The barycenter represents the **least-violent rearrangement** that satisfies all constraints—coherence in the strongest possible sense.

### 4.3 Transport Skeleton

Rather than denoising arbitrary intermediate states, we construct a **transport skeleton**:

1. **Plan**: Compute barycentric waypoints at coarse resolution
2. **Refine**: Diffusion projects each waypoint onto the learned manifold
3. **Verify**: Check topological consistency (Hauptvermutung-like constraint)

```
μ₀ (current) → μ_λ (barycentric interpolants) → μ₁ (target)
     ↓               ↓                              ↓
   x₀ (mesh)     x_λ (valid shapes)            x₁ (final mesh)
```

The transport skeleton ensures that the path through latent space passes only through **valid geometry**.

### 4.4 Computational Efficiency

Full OT barycenters are expensive. We make them viable by:
- **Low-dim projection**: Compute OT in PCA/learned subspace of Z
- **Entropic regularization**: Sinkhorn iterations for approximate barycenters
- **Coarse-scale only**: Barycenters at 8³/16³, standard attention at 32³/64³
- **Planning layer**: Compute skeleton once, not per-denoising-step

---

## 5. Epipolar Sketch-Based Modeling

### 5.1 The Multi-View Setup

User draws strokes in 2+ viewports with known (or estimated) camera matrices:
- **View 1**: Front view, stroke s₁
- **View 2**: Side view, stroke s₂
- **View 3**: (Optional) Perspective view, stroke s₃

### 5.2 Fundamental Matrix Constraint

For corresponding points (x₁, x₂) in two views, the **fundamental matrix** F encodes:

```
x₂ᵀ · F · x₁ = 0
```

This constrains where a point in View 1 can appear in View 2 (along an epipolar line).

### 5.3 F-Matrix Head

We add an **F-matrix prediction head** to the U-Net:

```
Input: Latent features from both views
Output: Predicted fundamental matrix F̂

Loss: Σ (x₂ᵀ · F̂ · x₁)² over corresponding stroke points
    + ||F̂ - F_GT||² if ground truth available
```

### 5.4 Triangulation to 3D

Given strokes s₁, s₂ and fundamental matrix F:

1. **Lift strokes to biarcs** (GSS Lift phase)
2. **Find correspondences**: Match biarc points across views using epipolar constraint
3. **Triangulate**: Compute 3D points from corresponding 2D points + camera matrices
4. **Fit 3D biarc**: Lift triangulated points to 3D G1-continuous curve
5. **Generate surface**: FNO predicts surface from 3D boundary (GSS Operate phase)

### 5.5 Epipolar Ghost Scaffolding

In real-time, the system displays:
- **Epipolar lines**: For each stroke point in View 1, show corresponding line in View 2
- **Predicted correspondences**: Highlight likely matching points
- **3D preview**: Show triangulated 3D structure as ghost mesh
- **Ambiguity regions**: Highlight where more strokes would help

This guides the user toward strokes that maximally constrain the 3D interpretation.

---

## 6. The Complete Pipeline: Intent to Geometry

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTENT → GEOMETRY PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STAGE 1: PRIME THE LATENT SPACE                                          │
│   ─────────────────────────────────                                         │
│   User provides intent: "mechanical assembly with cylindrical parts"        │
│   Intent → tokens → cross-attention keys/values                            │
│   Tokens activate regions of Dirac-trained latent manifold                 │
│   Result: Prior distribution μ_prior over semantic-geometric context       │
│                                                                             │
│   STAGE 2: SKETCH OBSERVATIONS                                              │
│   ────────────────────────────                                              │
│   User draws strokes in multiple viewports                                  │
│   Strokes → biarcs (GSS Lift)                                              │
│   Biarcs → spatial features → cross-attention queries                      │
│   Attention maps bind strokes to semantic tokens                           │
│   Epipolar constraint triangulates 3D structure                            │
│   Result: Observation likelihood p(strokes | z)                            │
│                                                                             │
│   STAGE 3: BARYCENTRIC UPDATE                                               │
│   ────────────────────────────                                              │
│   Multiple constraints produce target distributions:                        │
│   - μ_semantic (from intent tokens)                                        │
│   - μ_sketch (from stroke observations)                                    │
│   - μ_epipolar (from F-matrix consistency)                                 │
│   - μ_symmetry (from detected/specified symmetries)                        │
│   - μ_physics (from learned plausibility prior)                            │
│                                                                             │
│   Compute barycenter: μ* = argmin Σ wᵢ W₂²(μ, μᵢ)                         │
│   Result: Coherent latent distribution balancing all constraints           │
│                                                                             │
│   STAGE 4: DIFFUSION REFINEMENT                                             │
│   ──────────────────────────────                                            │
│   Denoise μ* to obtain on-manifold latent code z*                          │
│   Diffusion U-Net with cross-attention at each scale                       │
│   Result: Valid latent code z* in Dirac-trained manifold                   │
│                                                                             │
│   STAGE 5: GSS COLLAPSE                                                     │
│   ──────────────────────                                                    │
│   Decode z* → spectral field u                                             │
│   Verify topology: β(u) matches expected Betti numbers                     │
│   Extract isosurface: Dual contouring at user-specified resolution        │
│   Apply conformal UV: BFF parameterization                                 │
│   Apply affine transform: Place in world coordinates                       │
│   Result: Final mesh with UVs, topologically valid                         │
│                                                                             │
│   CONTINUOUS FEEDBACK                                                       │
│   ───────────────────                                                       │
│   Throughout, display ghost scaffolds:                                      │
│   - Smoothed biarcs (stroke interpretation)                                │
│   - Conformal grid (geometry flow)                                         │
│   - Predicted surface (current best guess)                                 │
│   - Epipolar lines (multi-view guidance)                                   │
│   - Suggested completions (likely next strokes)                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Compositional Design Grammar

### 7.1 Emergence from Latent Structure

The Dirac-trained latent space is not unstructured—it has learned a **geometry grammar**:

- **Primitives**: Regions of Z correspond to basic shapes (sphere, cylinder, box, torus)
- **Operations**: Directions in Z correspond to operations (scale, smooth, sharpen)
- **Compositions**: Paths through Z can implement booleans, blends, arrays

### 7.2 Rule-Based Navigation

SST procedural rules can be interpreted as **navigation instructions** in Z:

```yaml
# SST Rule
root:
  - Primitive: { type: CYLINDER, id: "base" }
  - Mirror: { axis: Y }
  - Fillet: { radius: 5 }
```

**Latent interpretation**:
1. Start at z_cylinder (primitive cluster)
2. Apply symmetry transformation (reflection in Z)
3. Navigate toward z_fillet (smoothing direction)

### 7.3 Sketch as Navigation

User strokes don't generate from scratch—they **navigate** the latent manifold:

- **First stroke**: Collapses ambiguity about global shape class
- **Second stroke**: Refines within that class
- **Detail strokes**: Fine-tune local features
- **Correction strokes**: Redirect toward alternative interpretations

This is why the system can "anticipate"—it knows the manifold structure and predicts likely destinations.

---

## 8. Mathematical Formulation

### 8.1 The Four-Context State

```
s = (A, φ, û, z) ∈ SE(3) × Conf(Ω) × L²(ℝ³) × Z
```

where:
- `A`: Affine transform (4×4 matrix)
- `φ`: Conformal map (Möbius transform or BFF parameterization)
- `û`: Spectral field coefficients (Fourier or neural implicit)
- `z`: Semantic-geometric latent code (Dirac-trained)

### 8.2 The Lift-Operate-Collapse-Decode Pipeline

```
C := Collapse ∘ Operate ∘ Lift           (GSS original)
G := Collapse ∘ Decode ∘ Navigate ∘ Encode ∘ Lift   (with latent)
```

where:
- `Encode`: Stroke features → latent observation
- `Navigate`: Barycentric update in latent space
- `Decode`: Latent code → spectral field

### 8.3 The Objective Function

End-to-end training minimizes:

```
min_{θ} E_data + λ₁E_smooth + λ₂E_frame + λ₃E_topo + λ₄E_epipolar + λ₅E_transport
```

where:
- `E_data`: User intent / sketch matching
- `E_smooth`: Curvature regularity (biarc fitting)
- `E_frame`: Darboux frame consistency
- `E_topo`: Persistent homology match (β_p = β*_p)
- `E_epipolar`: Fundamental matrix constraint (x₂ᵀFx₁ = 0)
- `E_transport`: Wasserstein transport cost (coherence)

### 8.4 The Cross-Attention Mechanism

At U-Net level ℓ:

```
Q^(ℓ) = W_Q x^(ℓ)           Queries from spatial features
K^(ℓ) = W_K T               Keys from semantic tokens
V^(ℓ) = W_V T               Values from semantic tokens

A^(ℓ) = softmax(Q^(ℓ) K^(ℓ)ᵀ / √d)    Attention weights

x^(ℓ) ← x^(ℓ) + A^(ℓ) V^(ℓ)           Semantic-guided update
```

### 8.5 The Barycentric Update

Given constraint distributions {μᵢ} with weights {wᵢ}:

```
μ* = argmin_μ Σᵢ wᵢ W₂²(μ, μᵢ)

z* = sample(μ*) or z* = mean(μ*) for deterministic output
```

Computed via entropic optimal transport (Sinkhorn iterations).

---

## 9. Research Directions

### 9.1 Dirac Operator Approximation

**Key papers**:
- Bär, "The Dirac Operator on Hyperbolic Manifolds" (1991)
- Cipriani et al., "Spectral Geometry with Applications" (2017)
- Sharp & Crane, "A Discrete Dirac Operator" (2021)

**Implementation**: Discrete Dirac via finite elements on triangle meshes; GPU-accelerated eigensolvers for large shape libraries.

### 9.2 Geometry-Aware Diffusion

**Key papers**:
- Huang et al., "Riemannian Diffusion Models" (2022)
- De Bortoli et al., "Diffusion Schrödinger Bridge" (2021)
- Chen et al., "Geodesic Diffusion for Shape Generation" (2023)

**Implementation**: Diffusion on the Riemannian manifold Z, not Euclidean; geodesic interpolation for latent paths.

### 9.3 Optimal Transport for Coherence

**Key papers**:
- Cuturi, "Sinkhorn Distances" (2013)
- Peyré & Cuturi, "Computational Optimal Transport" (2019)
- Chizat et al., "Scaling Algorithms for Unbalanced OT" (2018)

**Implementation**: Entropic OT for computational efficiency; unbalanced OT for partial constraints.

### 9.4 Multi-View Sketch-to-3D

**Key papers**:
- Delanoy et al., "3D Sketching using Multi-View Deep Volumetric Prediction" (2018)
- Li et al., "Sketch2CAD: Sequential CAD Modeling" (2020)
- Seff et al., "SketchGraphs: A Large-Scale Dataset for Modeling Relational Geometry" (2020)

**Implementation**: F-matrix prediction from learned features; differentiable triangulation.

---

## 10. Summary: The Vision

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GEOMETRY-AWARE LATENT DIFFUSION                          │
│                    FOR EPIPOLAR SKETCH-BASED MODELING                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   THE INSIGHT                                                               │
│   ───────────                                                               │
│   The Dirac operator's spectrum encodes intrinsic geometry.                │
│   A latent space trained on Dirac embeddings "knows" what shapes are.      │
│   User sketches navigate this manifold, not generate from noise.           │
│                                                                             │
│   THE ARCHITECTURE                                                          │
│   ────────────────                                                          │
│   Four-context state: Affine × Conformal × Spectral × Semantic-Geometric   │
│   U-Net with cross-attention: Spatial-semantic alignment at all scales     │
│   Wasserstein barycenters: Coherent multi-constraint blending              │
│   Epipolar head: Multi-view consistency for sketch-to-3D                   │
│                                                                             │
│   THE WORKFLOW                                                              │
│   ────────────                                                              │
│   1. Prime latent space with intent (text, rules, examples)                │
│   2. Draw strokes → observations that collapse latent distribution         │
│   3. System displays predictions, user steers toward preferred result      │
│   4. Multi-view strokes triangulate 3D structure                           │
│   5. GSS collapses to topologically valid mesh                             │
│                                                                             │
│   THE RESULT                                                                │
│   ──────────                                                                │
│   Sketch-based modeling where:                                              │
│   - Strokes navigate a learned geometry manifold                           │
│   - Intent primes likely shape classes                                     │
│   - Multiple views constrain 3D interpretation                             │
│   - Coherence is guaranteed by optimal transport                           │
│   - Output is resolution-independent and topologically valid               │
│                                                                             │
│   This is GENERATIVE ANTICIPATION:                                         │
│   The tool knows geometry, predicts intent, and guides the user            │
│   toward valid, coherent structures through a mathematically               │
│   principled navigation of shape space.                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix: Connection to GSS

This framework **extends** the Geometric Scaffold Supernode:

| GSS Component | Latent Extension |
|---------------|------------------|
| Affine context A | Unchanged — world placement |
| Conformal context φ | Unchanged — UV parameterization |
| Spectral context û | Now **decoded** from latent z |
| Lift (discrete→continuous) | Now also **encodes** to latent |
| Operate (math→math) | Now includes **latent navigation** |
| Collapse (continuous→discrete) | Unchanged — isosurfacing |
| Guardrails (topology) | Now includes **coherence constraint** |

The GSS remains the **backend compiler**. The latent diffusion model is the **frontend interface** that translates user intent into the mathematical representations GSS operates on.

```
USER INTENT
     ↓
┌─────────────────────────────────┐
│  LATENT DIFFUSION FRONTEND     │
│  (Dirac manifold navigation)    │
│  (Cross-attention guidance)     │
│  (Wasserstein coherence)        │
│  (Epipolar constraints)         │
└─────────────────────────────────┘
     ↓
     z (latent code)
     ↓
┌─────────────────────────────────┐
│  GSS BACKEND                    │
│  (Decode to spectral)           │
│  (BFF conformal map)            │
│  (Topology guardrails)          │
│  (Isosurface collapse)          │
└─────────────────────────────────┘
     ↓
VALID MESH
```

---

*We don't generate shapes from noise. We navigate a manifold of valid geometries, guided by intent, constrained by observation, coherent by construction.*
