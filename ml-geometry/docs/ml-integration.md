# ML Integration Layer

> **Neural Operators, Predictive Scaffolding, and Probabilistic Geometry**
> Plugging Intelligence into the SST
> Version 1.0 | 2026-01-27

---

## Overview

The SST provides a **compositional algebra** for geometry. This document specifies how machine learning components integrate without disrupting that algebra. The key insight:

```
ML predicts NODES, not geometry.
The Walker still executes.
The algebra stays closed.
```

ML components are **proposal generators**. They suggest node trees, the user accepts/rejects, and the deterministic SST handles execution.

---

## 1. Integration Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ML INTEGRATION LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   USER INPUT                                                                │
│   ──────────                                                                │
│   Sketch strokes ──┐                                                        │
│   Voice commands ──┼──▶ ENCODER ──▶ Latent ──▶ DECODER ──▶ Node Tree       │
│   Partial geometry─┘                                                        │
│                                                                             │
│   NODE TREE (proposed)                                                      │
│   ────────────────────                                                      │
│   ┌─────────────────────────────────────────┐                              │
│   │ root:                                    │                              │
│   │   - Platform: { id: "P1" }              │  ◀── ML Output               │
│   │   - Mirror: { axis: X }                 │      (YAML/JSON)             │
│   │     children:                           │                              │
│   │       - T: { v: [?, ?, ?] }  ◀─ params  │                              │
│   │       - Instance: { mesh: "?" } ◀─ ref  │                              │
│   └─────────────────────────────────────────┘                              │
│                          │                                                  │
│                          ▼                                                  │
│   ┌─────────────────────────────────────────┐                              │
│   │           GHOST RENDERER                 │                              │
│   │   (Translucent preview of proposal)     │                              │
│   └─────────────────────────────────────────┘                              │
│                          │                                                  │
│            User: Accept / Reject / Refine                                   │
│                          │                                                  │
│                          ▼                                                  │
│   ┌─────────────────────────────────────────┐                              │
│   │           SST WALKER                     │                              │
│   │   (Deterministic execution)             │                              │
│   └─────────────────────────────────────────┘                              │
│                          │                                                  │
│                          ▼                                                  │
│                    GEOMETRY BUFFER                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Neural Operator Integration (FNO/GNO/GNP)

### Purpose

Neural operators learn mappings between **function spaces**, not discrete samples. For geometry:

- **Input**: 2D sketch strokes (as continuous curves)
- **Output**: 3D form parameters (as continuous fields)

### Architecture: Sketch-to-SST

```
┌─────────────────────────────────────────────────────────────────┐
│                    SKETCH → SST PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   SKETCH INPUT              NEURAL OPERATOR           SST OUT   │
│   ────────────              ───────────────           ───────   │
│                                                                 │
│   Stroke₁(t) ──┐            ┌─────────────┐                    │
│   Stroke₂(t) ──┼──▶ Encode ─│ FNO Layers  │─▶ Decode ──▶ Nodes │
│   Stroke₃(t) ──┘            └─────────────┘                    │
│                                                                 │
│   Encoding:                                                     │
│   • Strokes → SDF (signed distance to stroke curves)           │
│   • Multi-view → Epipolar features                             │
│   • Temporal → Sequence of intent                              │
│                                                                 │
│   Decoding:                                                     │
│   • Latent → Node type probabilities                           │
│   • Latent → Transform parameters                              │
│   • Latent → Topology hints (symmetry, repetition)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Neural Operator Types

| Operator | Input Domain | Output Domain | Use Case |
|----------|--------------|---------------|----------|
| **FNO** (Fourier) | Regular grid | Regular grid | Volumetric SDF prediction |
| **GNO** (Graph) | Mesh/graph | Mesh/graph | Topology-aware deformation |
| **GNP** (Geometry-aware) | Point cloud | Point cloud | Unstructured scatter |
| **DeepONet** | Function + query | Function value | Continuous field evaluation |

### Integration Point: `MLPredictor` Node

```python
class MLPredictorNode(Node):
    """
    Meta-node that invokes ML model and injects predicted subtree.
    """
    def __init__(self, model_id: str, input_type: str):
        super().__init__('MLPredictor')
        self.model_id = model_id      # e.g., "sketch_to_sst_v1"
        self.input_type = input_type  # e.g., "strokes", "partial_geo"
    
    def execute(self, state: ExtendedState):
        # 1. Gather input from state
        input_data = self.gather_input(state)
        
        # 2. Invoke ML model (async, cached)
        prediction = ml_service.predict(self.model_id, input_data)
        
        # 3. Parse predicted node tree
        predicted_tree = parse_yaml(prediction['tree'])
        
        # 4. Store as "ghost" (not yet committed)
        state.ghost_buffer.append({
            'tree': predicted_tree,
            'confidence': prediction['confidence'],
            'origin': 'ml_predictor'
        })
        
        # 5. If auto-accept enabled, execute immediately
        if state.ml_auto_accept and prediction['confidence'] > 0.9:
            predicted_tree.execute(state)
```

### Training Data: SST as Ground Truth

The SST format becomes **training data**:

```yaml
# training_sample_0042.yaml
input:
  strokes:
    - [[0,0], [100,0], [100,100], [0,100]]  # Square-ish
    - [[50,50], [50,150]]                     # Vertical line
  view: "front"
  
output:
  tree:
    - Platform: { id: "base" }
    - T: { v: [50, 50, 0] }
    - Box: { size: [100, 100, 20] }
    - T: { v: [0, 0, 50] }
    - Cylinder: { r: 10, h: 100 }
```

**Key insight**: Every manually-created SST becomes a training example.

---

## 3. G-CNN Ghost Scaffolding (Predictive Symmetry)

### Purpose

**Group-equivariant CNNs** respect symmetry by construction. When the user works inside a Mirror node, the G-CNN predicts what comes next based on symmetry priors.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    G-CNN SCAFFOLDING                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   CURRENT STATE              G-CNN                 PREDICTION   │
│   ─────────────              ─────                 ──────────   │
│                                                                 │
│   Active Platform ──┐        ┌─────────┐                       │
│   Symmetry Depth ───┼──▶     │ G-CNN   │ ──▶ Next Node Probs   │
│   Recent Nodes ─────┤        │ (E(2))  │     Next Params Dist  │
│   Partial Geometry ─┘        └─────────┘                       │
│                                                                 │
│   Equivariance Groups:                                          │
│   • E(2): 2D Euclidean (rotation + translation)                │
│   • SE(3): 3D rigid (rotation + translation)                   │
│   • Bilateral: Z₂ (mirror symmetry)                            │
│   • Radial: Cₙ (n-fold rotational)                             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Prediction Output

```python
@dataclass
class GCNNPrediction:
    # Node type probabilities
    node_probs: Dict[str, float]  # {"Instance": 0.7, "T": 0.2, "Mirror": 0.1}
    
    # Parameter distributions (Gaussian)
    param_dists: Dict[str, Distribution]  # {"T.v": N([10,0,0], [2,1,1])}
    
    # Confidence
    confidence: float
    
    # Symmetry context
    symmetry_group: str  # "bilateral", "radial_6", etc.
    
    # Suggested completion
    suggested_tree: Optional[NodeTree]
```

### Ghost Rendering

```python
class GhostRenderer:
    """
    Renders ML predictions as translucent geometry.
    """
    def render_prediction(self, prediction: GCNNPrediction, state: State):
        # Clone state for preview
        preview_state = state.clone()
        preview_state.ghost_mode = True
        
        # Execute predicted tree
        if prediction.suggested_tree:
            prediction.suggested_tree.execute(preview_state)
        
        # Render with transparency based on confidence
        for record in preview_state.buffer:
            self.render_ghost_mesh(
                record['geometry'],
                record['transform'],
                alpha=prediction.confidence * 0.5
            )
```

### User Interaction

```
┌─────────────────────────────────────────────────────────────────┐
│                    GHOST INTERACTION                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Ghost appears as user works...                                │
│                                                                 │
│   [TAB]     → Accept ghost, commit to SST                      │
│   [ESC]     → Dismiss ghost                                    │
│   [SCROLL]  → Cycle through alternative predictions            │
│   [DRAG]    → Refine predicted parameters                      │
│   [SHIFT]   → Hold to see confidence visualization             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Epipolar Geometry ML (Stereo-Photogrammetric Sketcher)

### Purpose

Classical epipolar geometry constrains 3D reconstruction from multiple views. We extend this with **probabilistic inference**:

- User sketches from multiple implicit viewpoints
- System maintains **probability distribution over 3D form**
- Distribution sharpens as more strokes are added

### The Gaussian Tube/Surface Representation

Instead of reconstructing a single mesh, maintain uncertainty:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PROBABILISTIC GEOMETRY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   GAUSSIAN TUBE (for curves)                                    │
│   ──────────────────────────                                    │
│                                                                 │
│   Each point on curve has position uncertainty:                 │
│                                                                 │
│   P(x,y,z) ~ N(μ, Σ)                                           │
│                                                                 │
│   μ = mean position (center of tube)                           │
│   Σ = covariance (ellipsoid of uncertainty)                    │
│                                                                 │
│   Visualization: Translucent tube, radius = uncertainty        │
│                                                                 │
│   ─────────────────────────────────────────────────────────    │
│   │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│    │
│   │░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░│    │
│   │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│    │
│   ─────────────────────────────────────────────────────────    │
│         Low confidence               High confidence            │
│         (wide tube)                  (narrow tube)              │
│                                                                 │
│   GAUSSIAN SURFACE (for patches)                                │
│   ──────────────────────────────                                │
│                                                                 │
│   Each point on surface: P(x,y,z) ~ N(μ(u,v), Σ(u,v))         │
│   Rendered as translucent surface with thickness = σ           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Multi-View Stroke Integration

```python
@dataclass
class ViewStroke:
    points_2d: List[Vec2]      # Screen-space stroke
    view_matrix: Mat4          # Camera transform
    view_type: str             # "front", "side", "perspective", "inferred"

class EpipolarReconstructor:
    def __init__(self):
        self.strokes: List[ViewStroke] = []
        self.distribution: GaussianField = None
    
    def add_stroke(self, stroke: ViewStroke):
        """Add a stroke and update the probability distribution."""
        self.strokes.append(stroke)
        
        # Compute epipolar constraints
        if len(self.strokes) >= 2:
            constraints = self.compute_epipolar_constraints()
            
            # Update distribution via Bayesian inference
            self.distribution = self.bayesian_update(
                prior=self.distribution,
                evidence=constraints
            )
    
    def compute_epipolar_constraints(self) -> List[EpipolarConstraint]:
        """
        For each pair of strokes from different views,
        compute the epipolar lines and their intersections.
        """
        constraints = []
        for s1, s2 in combinations(self.strokes, 2):
            if s1.view_matrix != s2.view_matrix:
                # Fundamental matrix between views
                F = compute_fundamental_matrix(s1.view_matrix, s2.view_matrix)
                
                # For each point in s1, compute epipolar line in s2's view
                for p1 in s1.points_2d:
                    epipolar_line = F @ p1.homogeneous()
                    
                    # Find closest point on s2 to epipolar line
                    p2, distance = closest_point_to_line(s2.points_2d, epipolar_line)
                    
                    # Triangulate 3D point with uncertainty
                    p3d, uncertainty = triangulate_with_uncertainty(
                        p1, s1.view_matrix,
                        p2, s2.view_matrix,
                        reprojection_error=distance
                    )
                    
                    constraints.append(EpipolarConstraint(p3d, uncertainty))
        
        return constraints
    
    def to_sst_nodes(self, confidence_threshold: float = 0.8) -> NodeTree:
        """
        Convert the probability distribution to SST nodes.
        High-confidence regions become concrete geometry.
        Low-confidence regions become suggestions.
        """
        # Extract iso-surface at confidence threshold
        high_conf_points = self.distribution.extract_high_confidence(confidence_threshold)
        
        # Fit SST primitives to the point cloud
        fitted_tree = self.fit_primitives(high_conf_points)
        
        return fitted_tree
```

### Visualization

```yaml
# Epipolar visualization modes

mode: UNCERTAINTY_TUBES
  # Render curves as tubes where radius = σ
  # Color gradient: green (certain) → red (uncertain)

mode: PROBABILITY_VOLUME
  # Render 3D probability field as volumetric fog
  # Density = probability

mode: CONFIDENCE_MESH
  # Render mesh with vertex colors = confidence
  # User can scrub threshold to see form "crystallize"

mode: EPIPOLAR_LINES
  # Debug: show epipolar constraints as lines in 3D
```

---

## 5. Integration Points in SST

### New Node Types

```yaml
Σ_ml:
  MLPredict:
    model: string          # Model identifier
    input: string          # Input source ("strokes", "partial", "voice")
    auto_accept: bool      # Commit if confidence > threshold
    
  GhostScope:
    # Children render as ghosts until accepted
    children: [...]
    confidence_threshold: float
    
  ProbabilisticPrimitive:
    type: string           # "tube", "surface", "volume"
    distribution: GaussianField
    
  ConfidenceFilter:
    threshold: float       # Only emit geometry above this confidence
    children: [...]
```

### State Extensions

```python
class MLState:
    # Pending predictions (ghosts)
    ghost_buffer: List[GhostPrediction] = []
    
    # Active probability distributions
    distributions: Dict[str, GaussianField] = {}
    
    # ML model cache
    model_cache: Dict[str, MLModel] = {}
    
    # User preference: auto-accept threshold
    auto_accept_threshold: float = 0.9
    
    # Stroke history for epipolar reconstruction
    stroke_history: List[ViewStroke] = []
```

---

## 6. Training Pipeline

### Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   1. USER CREATES SST                                           │
│      ────────────────                                           │
│      Manual node composition → Saved as YAML                    │
│                                                                 │
│   2. AUTOMATIC AUGMENTATION                                     │
│      ──────────────────────                                     │
│      • Render from multiple views → Synthetic sketches          │
│      • Add noise, incomplete strokes                            │
│      • Vary parameters within valid ranges                      │
│                                                                 │
│   3. TRAINING PAIRS                                             │
│      ──────────────                                             │
│      Input: Augmented sketches/partial geometry                 │
│      Output: Original SST node tree                             │
│                                                                 │
│   4. MODEL TRAINING                                             │
│      ──────────────                                             │
│      FNO/GNO for sketch→form                                    │
│      G-CNN for symmetry prediction                              │
│      Transformer for node sequence prediction                   │
│                                                                 │
│   5. DEPLOYMENT                                                 │
│      ──────────                                                 │
│      Models served via MCP endpoint                             │
│      Claude can invoke models as tools                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### MCP Endpoints for ML

```json
{
    "method": "ml.predict",
    "params": {
        "model": "sketch_to_sst_v2",
        "input": {
            "strokes": [...],
            "view": "front",
            "context": { "active_platform": "P1", "sym_depth": 1 }
        }
    }
}

{
    "method": "ml.train_on_session",
    "params": {
        "session_id": "abc123",
        "feedback": "accepted"  // or "rejected", "modified"
    }
}

{
    "method": "ml.get_ghost",
    "params": {
        "state_snapshot": {...},
        "prediction_type": "next_node"
    }
}
```

---

## 7. Implementation Phases

### Phase 7A: Ghost Infrastructure
- [ ] Add `ghost_buffer` to State
- [ ] Implement GhostRenderer (translucent preview)
- [ ] Add keyboard shortcuts (TAB/ESC/SCROLL)
- [ ] **TEST**: Manual ghost injection, accept/reject

### Phase 7B: G-CNN Symmetry Predictor
- [ ] Train G-CNN on SST corpus
- [ ] Implement real-time prediction during Mirror node
- [ ] Confidence visualization
- [ ] **TEST**: Symmetry completion suggestions

### Phase 7C: Sketch-to-SST (FNO)
- [ ] Build training data generator
- [ ] Train FNO encoder-decoder
- [ ] Integrate with stroke input
- [ ] **TEST**: Draw square → get Box node

### Phase 7D: Epipolar Reconstructor
- [ ] Implement multi-view stroke collection
- [ ] Bayesian distribution update
- [ ] Gaussian tube rendering
- [ ] **TEST**: Front + side sketch → 3D form

---

## 8. Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    ML INTEGRATION SUMMARY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   KEY PRINCIPLE: ML proposes, SST disposes.                     │
│                                                                 │
│   • Neural operators predict NODE TREES, not raw geometry       │
│   • Predictions rendered as GHOSTS until accepted               │
│   • G-CNNs provide SYMMETRY-AWARE suggestions                   │
│   • Epipolar ML maintains PROBABILITY DISTRIBUTIONS             │
│   • User feedback becomes TRAINING DATA                         │
│   • Claude can invoke ML via MCP ENDPOINTS                      │
│                                                                 │
│   The algebra stays closed. ML is just another node source.     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

*Intelligence proposes. Algebra executes. The user decides.*
