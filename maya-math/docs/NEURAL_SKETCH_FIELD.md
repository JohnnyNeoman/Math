# Neural Sketch-Field Framework

> **A Minimalist Test Bed for Generative Anticipation in Sketch-Based Modeling**
> Field-Solving Neural Operators on Anticipated Surfaces
> Version 1.0 | 2026-01-29

---

## Vision

A **focused, elegant test** of generative anticipation principles:

> **Draw a boundary → System anticipates the surface → Neural operator solves the field → Artist refines at any resolution**

The goal is not a complete modeling tool, but a **proof-of-concept** that validates:
1. Neural operators can predict surfaces from boundary curvatures
2. Latent priming (base mesh, description) guides anticipation
3. Cross-attention binds strokes to geometric intent
4. The system suggests, the artist confirms

---

## 1. Core Concept: The Anticipation Loop

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE ANTICIPATION LOOP                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐                                                          │
│   │   PRIME     │  Base mesh, description, or empty canvas                 │
│   │   (Latent)  │  → Activates regions of geometry manifold                │
│   └──────┬──────┘                                                          │
│          ↓                                                                  │
│   ┌─────────────┐                                                          │
│   │   SKETCH    │  User draws boundary curves                              │
│   │   (Stroke)  │  → Parameterize, sample curvature, classify             │
│   └──────┬──────┘                                                          │
│          ↓                                                                  │
│   ┌─────────────┐                                                          │
│   │  ANTICIPATE │  System predicts likely surface                          │
│   │   (Field)   │  → Ghost scaffold appears as suggestion                  │
│   └──────┬──────┘                                                          │
│          ↓                                                                  │
│   ┌─────────────┐                                                          │
│   │   CONFIRM   │  Artist accepts, rejects, or refines                     │
│   │   (Commit)  │  → Field bakes to geometry at chosen resolution          │
│   └──────┬──────┘                                                          │
│          ↓                                                                  │
│   ┌─────────────┐                                                          │
│   │   EXTEND    │  New surface becomes context for next sketch             │
│   │   (Chain)   │  → Boundaries can be extruded, new patches added        │
│   └─────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Two Sketch Modes

### Mode A: Free-Form Bilateral Symmetry

**Use case**: Character sculpting, vehicle design, organic forms

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BILATERAL SYMMETRY MODE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User draws on ONE side → System mirrors to OTHER side                    │
│                                                                             │
│   ┌─────────────────┬─────────────────┐                                    │
│   │     LEFT        │      RIGHT      │                                    │
│   │    (drawn)      │    (mirrored)   │                                    │
│   │                 │                 │                                    │
│   │    ╭───╮        │        ╭───╮    │                                    │
│   │   ╱     ╲       │       ╱     ╲   │                                    │
│   │  │       │      │      │       │  │                                    │
│   │   ╲     ╱       │       ╲     ╱   │                                    │
│   │    ╰───╯        │        ╰───╯    │                                    │
│   │                 │                 │                                    │
│   └─────────────────┴─────────────────┘                                    │
│                     ↑                                                       │
│              Symmetry Plane                                                 │
│                                                                             │
│   The "dual curve" (left + mirrored right) defines a 3D boundary.          │
│   FNO predicts the surface that spans this boundary.                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation**:
```python
class BilateralSketchMode:
    def __init__(self, symmetry_plane: Plane = Plane.YZ):
        self.symmetry_plane = symmetry_plane
        self.strokes_left = []
        self.strokes_right = []  # Auto-mirrored
    
    def on_stroke(self, stroke: Stroke):
        # Parameterize and fit
        curve = self.fit_curve(stroke)
        
        # Mirror across symmetry plane
        mirrored = curve.reflect(self.symmetry_plane)
        
        self.strokes_left.append(curve)
        self.strokes_right.append(mirrored)
        
        # Check if boundary is closed
        if self.forms_closed_boundary():
            self.trigger_anticipation()
    
    def trigger_anticipation(self):
        boundary = self.combine_to_boundary()
        anticipated_surface = self.neural_field.predict(boundary)
        self.display_ghost(anticipated_surface)
```

### Mode B: Reef Topology (Surface Drawing)

**Use case**: Retopology, detail sculpting, patch extension

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REEF TOPOLOGY MODE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User draws ON an existing surface → Curves project to surface            │
│                                                                             │
│        ╭─────────────────────╮                                             │
│       ╱                       ╲                                            │
│      │    ┌───────────┐        │   ← Existing surface                     │
│      │    │  (drawn   │        │                                          │
│      │    │   patch)  │        │   ← User sketches boundary on surface    │
│       ╲   └───────────┘       ╱                                            │
│        ╰─────────────────────╯                                             │
│                                                                             │
│   The bounded region inherits curvature from the base surface.             │
│   FNO predicts how to EXTRUDE or MODIFY within the patch.                  │
│                                                                             │
│   Operations:                                                               │
│   • EXTRUDE: Push patch outward (add detail)                               │
│   • INSET: Create sub-patch with offset                                    │
│   • BRIDGE: Connect two patches                                            │
│   • SMOOTH: Relax patch to base surface curvature                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Implementation**:
```python
class ReefTopologyMode:
    def __init__(self, base_surface: Surface):
        self.base_surface = base_surface
        self.patches = []
    
    def on_stroke(self, stroke: Stroke):
        # Project stroke onto base surface
        projected = self.project_to_surface(stroke, self.base_surface)
        
        # Fit to surface-constrained curve (geodesic or projected)
        curve = self.fit_surface_curve(projected)
        
        # Check if forms closed patch
        if curve.is_closed():
            patch = self.create_patch(curve)
            self.patches.append(patch)
            self.trigger_anticipation(patch)
    
    def trigger_anticipation(self, patch: Patch):
        # Sample curvature from base surface at patch boundary
        boundary_curvature = self.sample_boundary_curvature(patch)
        
        # Predict likely extrusion/modification
        anticipated = self.neural_field.predict_patch_operation(
            patch=patch,
            base_curvature=boundary_curvature,
            context=self.base_surface
        )
        
        self.display_ghost(anticipated)
```

---

## 3. Curve Fitting & Classification

### The Parameterization Pipeline

When user draws a stroke, immediately:

```
Raw Stroke → Smooth → Parameterize → Sample Curvature → Classify → Fit
```

**Step 1: Parameterize by Arc Length**
```python
def parameterize_curve(points: List[Vec2]) -> Curve:
    """
    Convert raw points to arc-length parameterized curve.
    """
    # Compute cumulative arc length
    arc_lengths = [0.0]
    for i in range(1, len(points)):
        arc_lengths.append(arc_lengths[-1] + distance(points[i], points[i-1]))
    
    total_length = arc_lengths[-1]
    
    # Normalize to [0, 1]
    t_values = [s / total_length for s in arc_lengths]
    
    return Curve(points, t_values)
```

**Step 2: Sample Curvature**
```python
def sample_curvature(curve: Curve, n_samples: int = 100) -> List[float]:
    """
    Sample signed curvature along the curve.
    
    Curvature κ = dθ/ds where θ is tangent angle, s is arc length.
    """
    curvatures = []
    
    for i in range(n_samples):
        t = i / (n_samples - 1)
        
        # Compute tangent angle
        tangent = curve.tangent_at(t)
        theta = atan2(tangent.y, tangent.x)
        
        # Compute curvature via finite difference
        if i > 0 and i < n_samples - 1:
            t_prev = (i - 1) / (n_samples - 1)
            t_next = (i + 1) / (n_samples - 1)
            
            theta_prev = atan2(*curve.tangent_at(t_prev)[::-1])
            theta_next = atan2(*curve.tangent_at(t_next)[::-1])
            
            ds = curve.arc_length(t_prev, t_next)
            kappa = (theta_next - theta_prev) / ds
        else:
            kappa = 0.0
        
        curvatures.append(kappa)
    
    return curvatures
```

**Step 3: Classify & Fit**

```python
class CurveClassifier:
    """
    Classify closed curves and fit to canonical primitives.
    """
    
    CLASSES = ['circle', 'ellipse', 'rounded_rectangle', 'capsule', 'freeform']
    
    def classify_and_fit(self, curve: Curve) -> FittedCurve:
        if not curve.is_closed():
            return FittedCurve(curve, class_='open', primitive=None)
        
        # Sample curvature signature
        curvatures = sample_curvature(curve)
        
        # Compute statistics
        kappa_mean = np.mean(np.abs(curvatures))
        kappa_std = np.std(curvatures)
        kappa_range = max(curvatures) - min(curvatures)
        
        # Classification heuristics
        if kappa_std < 0.1 * kappa_mean:
            # Nearly constant curvature → Circle
            return self.fit_circle(curve)
        
        elif self.has_four_corners(curvatures):
            # Four curvature peaks → Rounded rectangle
            return self.fit_rounded_rectangle(curve)
        
        elif self.is_bimodal(curvatures):
            # Two curvature modes → Ellipse or Capsule
            if self.has_flat_sections(curvatures):
                return self.fit_capsule(curve)
            else:
                return self.fit_ellipse(curve)
        
        else:
            # Freeform
            return FittedCurve(curve, class_='freeform', primitive=None)
    
    def fit_circle(self, curve: Curve) -> FittedCurve:
        """Fit to perfect circle."""
        # Least-squares circle fit
        center, radius = least_squares_circle(curve.points)
        
        primitive = Circle(center, radius)
        fitted_curve = primitive.to_curve(n_points=len(curve.points))
        
        return FittedCurve(fitted_curve, class_='circle', primitive=primitive)
    
    def fit_rounded_rectangle(self, curve: Curve) -> FittedCurve:
        """Fit to rounded rectangle."""
        # Find corner positions (curvature peaks)
        corners = self.find_curvature_peaks(curve, n=4)
        
        # Compute bounding box
        bbox = compute_bbox(curve.points)
        
        # Estimate corner radius from curvature at corners
        corner_kappa = np.mean([curve.curvature_at(c) for c in corners])
        corner_radius = 1.0 / corner_kappa if corner_kappa > 0 else 0
        
        primitive = RoundedRectangle(
            center=bbox.center,
            width=bbox.width,
            height=bbox.height,
            corner_radius=corner_radius
        )
        fitted_curve = primitive.to_curve(n_points=len(curve.points))
        
        return FittedCurve(fitted_curve, class_='rounded_rectangle', primitive=primitive)
    
    def fit_ellipse(self, curve: Curve) -> FittedCurve:
        """Fit to ellipse."""
        # Least-squares ellipse fit
        center, axes, rotation = least_squares_ellipse(curve.points)
        
        primitive = Ellipse(center, axes[0], axes[1], rotation)
        fitted_curve = primitive.to_curve(n_points=len(curve.points))
        
        return FittedCurve(fitted_curve, class_='ellipse', primitive=primitive)
    
    def fit_capsule(self, curve: Curve) -> FittedCurve:
        """Fit to capsule (stadium shape)."""
        # Find the two semicircular ends
        # ... implementation
        pass
```

---

## 4. Neural Field Anticipation

### The Field Solver

Given a boundary curve (from user sketch), predict the surface that fills it.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEURAL FIELD ANTICIPATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT                                                                     │
│   ─────                                                                     │
│   • Boundary curve (parameterized, with curvature samples)                 │
│   • Context (base surface curvature, if reef mode)                         │
│   • Priming (latent from base mesh or description)                         │
│                                                                             │
│   PROCESS                                                                   │
│   ───────                                                                   │
│   1. Encode boundary as sparse indicator function                          │
│   2. Encode curvature as boundary condition                                │
│   3. FNO predicts interior field (SDF or height field)                     │
│   4. Apply latent prior (cross-attention with priming)                     │
│                                                                             │
│   OUTPUT                                                                    │
│   ──────                                                                    │
│   • Spectral field coefficients (resolution-independent)                   │
│   • Can be evaluated at any resolution for preview/render                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class NeuralFieldAnticipator:
    """
    Predicts surface from boundary using FNO with latent priming.
    """
    
    def __init__(self, fno_model: FNO, latent_encoder: LatentEncoder):
        self.fno = fno_model
        self.encoder = latent_encoder
    
    def predict(self, 
                boundary: Curve, 
                context: Optional[Surface] = None,
                priming: Optional[LatentPrior] = None) -> SpectralField:
        """
        Predict surface field from boundary curve.
        
        Args:
            boundary: Closed curve defining the patch boundary
            context: Optional base surface (for reef mode)
            priming: Optional latent prior (from base mesh or description)
        
        Returns:
            SpectralField that can be evaluated at any resolution
        """
        # ─────────────────────────────────────────────────────────────────
        # STEP 1: Encode boundary as sparse grid
        # ─────────────────────────────────────────────────────────────────
        
        # Rasterize boundary to coarse grid (32x32 for 2D, 32³ for 3D)
        boundary_grid = self.rasterize_boundary(boundary, resolution=(32, 32))
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 2: Encode curvature as boundary condition
        # ─────────────────────────────────────────────────────────────────
        
        # Sample curvature along boundary
        curvatures = sample_curvature(boundary, n_samples=boundary_grid.shape[0] * 4)
        
        # Create curvature field (values at boundary, zero elsewhere)
        curvature_grid = self.rasterize_scalar(boundary, curvatures, resolution=(32, 32))
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 3: Encode context (if reef mode)
        # ─────────────────────────────────────────────────────────────────
        
        if context is not None:
            # Sample base surface curvature in patch region
            context_grid = self.sample_surface_curvature(context, boundary, resolution=(32, 32))
        else:
            context_grid = torch.zeros_like(boundary_grid)
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 4: Apply latent priming (cross-attention)
        # ─────────────────────────────────────────────────────────────────
        
        if priming is not None:
            # Cross-attention: boundary features attend to latent prior
            priming_features = self.cross_attend(boundary_grid, priming.latent)
        else:
            priming_features = torch.zeros_like(boundary_grid)
        
        # ─────────────────────────────────────────────────────────────────
        # STEP 5: FNO inference
        # ─────────────────────────────────────────────────────────────────
        
        # Stack inputs
        input_tensor = torch.stack([
            boundary_grid,      # Where is the boundary?
            curvature_grid,     # What curvature at boundary?
            context_grid,       # What's the base surface like?
            priming_features    # What does the prior suggest?
        ], dim=0)
        
        # Run FNO
        spectral_coeffs = self.fno(input_tensor)
        
        # ─────────────────────────────────────────────────────────────────
        # RETURN: Spectral field (resolution-independent)
        # ─────────────────────────────────────────────────────────────────
        
        return SpectralField(
            coefficients=spectral_coeffs,
            boundary=boundary,
            domain=self.compute_domain(boundary)
        )
    
    def cross_attend(self, spatial_features: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        """
        Cross-attention: spatial features attend to latent prior.
        
        This is where the "priming" happens — the latent code biases
        the predicted surface toward geometries consistent with the prior.
        """
        # Flatten spatial features to sequence
        B, H, W = spatial_features.shape
        queries = spatial_features.view(B, H * W, 1)  # (B, HW, 1)
        
        # Latent as keys/values
        keys = self.key_proj(latent)    # (B, D, K)
        values = self.value_proj(latent)  # (B, D, V)
        
        # Attention
        attn_weights = torch.softmax(queries @ keys.transpose(-1, -2) / sqrt(keys.shape[-1]), dim=-1)
        attended = attn_weights @ values
        
        # Reshape back to spatial
        return attended.view(B, H, W)
```

### Physics Modes

The FNO can be trained for different "physics" interpretations:

```python
class PhysicsMode(Enum):
    MINIMAL_SURFACE = "minimal"      # Soap film (minimize area)
    INFLATION = "inflate"            # Balloon (uniform pressure)
    CURVATURE_FLOW = "flow"          # Smooth toward base curvature
    EXTRUSION = "extrude"            # Push outward uniformly
    BLEND = "blend"                  # Smooth interpolation to neighbors
```

---

## 5. Latent Priming

### Priming from Base Mesh

```python
class MeshPrimer:
    """
    Encode a base mesh to latent prior that guides anticipation.
    """
    
    def __init__(self, encoder: MeshEncoder):
        self.encoder = encoder
    
    def prime(self, mesh: Mesh) -> LatentPrior:
        """
        Encode mesh to latent code.
        
        The latent captures:
        - Overall shape category (organic, mechanical, architectural)
        - Curvature distribution (smooth, sharp, mixed)
        - Symmetry patterns
        - Typical feature scales
        """
        # Extract geometric features
        vertices = mesh.vertices
        normals = mesh.vertex_normals
        curvatures = mesh.principal_curvatures
        
        # Encode via graph neural network or PointNet++
        latent = self.encoder(vertices, normals, curvatures)
        
        return LatentPrior(
            latent=latent,
            source='mesh',
            metadata={'vertex_count': len(vertices)}
        )
```

### Priming from Description

```python
class DescriptionPrimer:
    """
    Encode text description to latent prior.
    """
    
    def __init__(self, text_encoder: TextEncoder, geometry_aligner: GeometryAligner):
        self.text_encoder = text_encoder
        self.aligner = geometry_aligner
    
    def prime(self, description: str) -> LatentPrior:
        """
        Encode description to geometry-aligned latent.
        
        Examples:
        - "smooth organic form" → bias toward low curvature variation
        - "mechanical part with sharp edges" → bias toward high curvature at corners
        - "cylindrical with rounded ends" → bias toward capsule-like shapes
        """
        # Encode text
        text_embedding = self.text_encoder(description)
        
        # Align to geometry space (CLIP-like alignment)
        geometry_latent = self.aligner(text_embedding)
        
        return LatentPrior(
            latent=geometry_latent,
            source='description',
            metadata={'text': description}
        )
```

### Priming from Symmetry Assumptions

```python
class SymmetryPrimer:
    """
    Generate latent prior from symmetry specification.
    """
    
    def prime(self, symmetry: SymmetrySpec) -> LatentPrior:
        """
        Encode symmetry constraints as latent bias.
        
        Examples:
        - Bilateral: Expect mirrored features across plane
        - Radial(8): Expect 8-fold rotational symmetry
        - Translational: Expect repeating pattern
        """
        # Encode symmetry type
        symmetry_embedding = self.symmetry_encoder(symmetry)
        
        return LatentPrior(
            latent=symmetry_embedding,
            source='symmetry',
            metadata={'type': symmetry.type, 'params': symmetry.params}
        )
```

---

## 6. Ghost Scaffolding

### Real-Time Preview

As the user sketches, display anticipated geometry as translucent "ghost":

```python
class GhostScaffold:
    """
    Manages real-time display of anticipated geometry.
    """
    
    def __init__(self, renderer: Renderer):
        self.renderer = renderer
        self.current_ghost = None
        self.confidence = 0.0
    
    def update(self, anticipated_field: SpectralField, confidence: float):
        """
        Update ghost display.
        
        confidence: How certain is the anticipation (0 to 1)
        - Low confidence: Faint, multiple alternatives shown
        - High confidence: Solid, single suggestion
        """
        self.confidence = confidence
        
        # Extract mesh at preview resolution
        preview_resolution = self.compute_preview_resolution(confidence)
        mesh = anticipated_field.extract_mesh(preview_resolution)
        
        # Compute display parameters
        alpha = 0.2 + 0.3 * confidence  # More opaque when confident
        color = self.confidence_color(confidence)  # Blue→Green as confidence increases
        
        # Update display
        self.current_ghost = GhostMesh(
            mesh=mesh,
            alpha=alpha,
            color=color,
            wireframe=confidence < 0.5  # Show wireframe when uncertain
        )
        
        self.renderer.display_ghost(self.current_ghost)
    
    def show_alternatives(self, alternatives: List[SpectralField]):
        """
        Show multiple possible interpretations.
        
        Used when classification is ambiguous (e.g., circle vs. ellipse).
        """
        for i, field in enumerate(alternatives[:3]):  # Max 3 alternatives
            mesh = field.extract_mesh(resolution=(16, 16, 16))
            
            self.renderer.display_ghost(GhostMesh(
                mesh=mesh,
                alpha=0.15,
                color=self.alternative_color(i),
                label=field.metadata.get('class', f'Option {i+1}')
            ))
    
    def confirm(self) -> Mesh:
        """
        User confirms the current ghost → bake to real geometry.
        """
        if self.current_ghost is None:
            return None
        
        # Extract at full resolution
        field = self.current_ghost.source_field
        final_mesh = field.extract_mesh(resolution=self.render_resolution)
        
        # Clear ghost
        self.renderer.clear_ghosts()
        self.current_ghost = None
        
        return final_mesh
```

### Confidence Estimation

```python
class ConfidenceEstimator:
    """
    Estimate how confident the anticipation is.
    """
    
    def estimate(self, 
                 boundary: Curve, 
                 classification: CurveClassification,
                 priming: Optional[LatentPrior]) -> float:
        """
        Confidence based on:
        - How well the curve fits a known primitive
        - How consistent with priming
        - How closed/complete the boundary is
        """
        scores = []
        
        # Fit quality
        if classification.primitive is not None:
            fit_error = classification.fit_error
            fit_score = 1.0 / (1.0 + fit_error * 10)  # Sigmoid-like
            scores.append(fit_score)
        
        # Priming consistency
        if priming is not None:
            consistency = self.compute_priming_consistency(boundary, priming)
            scores.append(consistency)
        
        # Closure completeness
        if boundary.is_closed():
            closure_score = 1.0
        else:
            # How close to closing?
            gap = distance(boundary.start, boundary.end)
            total_length = boundary.arc_length()
            closure_score = 1.0 - min(gap / total_length, 1.0)
        scores.append(closure_score)
        
        # Weighted average
        return np.mean(scores) if scores else 0.5
```

---

## 7. Surface Extension (Patch Chaining)

### Boundary Extrusion

Once a patch is confirmed, its boundaries become candidates for extension:

```python
class PatchExtender:
    """
    Extend confirmed patches by extruding boundaries.
    """
    
    def __init__(self, neural_field: NeuralFieldAnticipator):
        self.neural_field = neural_field
        self.patches = []
    
    def add_patch(self, patch: ConfirmedPatch):
        """Register a confirmed patch."""
        self.patches.append(patch)
        self.update_extrudable_boundaries()
    
    def update_extrudable_boundaries(self):
        """
        Identify boundaries that can be extruded.
        
        A boundary is extrudable if:
        - It's on the edge of the current geometry
        - It's not shared with another patch (unless bridging)
        """
        self.extrudable = []
        
        for patch in self.patches:
            for edge in patch.boundary_edges:
                if not self.is_shared(edge):
                    self.extrudable.append(ExtrudableBoundary(
                        edge=edge,
                        patch=patch,
                        normal=self.compute_extrusion_direction(edge, patch)
                    ))
    
    def anticipate_extrusion(self, boundary: ExtrudableBoundary, 
                            sketch: Optional[Stroke] = None) -> SpectralField:
        """
        Anticipate what surface would result from extruding this boundary.
        
        If sketch is provided: Use sketch as guide for extrusion shape
        If no sketch: Predict based on boundary curvature + priming
        """
        # Get curvature at boundary
        boundary_curvature = boundary.edge.curvature_samples
        
        # Get patch surface curvature for context
        patch_curvature = boundary.patch.surface_curvature_at_boundary(boundary.edge)
        
        if sketch is not None:
            # User provided guide curve for extrusion
            extrusion_profile = self.fit_curve(sketch)
        else:
            # Predict profile from curvature continuity
            extrusion_profile = self.predict_continuation(
                boundary_curvature, 
                patch_curvature
            )
        
        # Generate new boundary (original + extruded profile + closing)
        new_boundary = self.construct_extrusion_boundary(
            boundary.edge,
            extrusion_profile,
            boundary.normal
        )
        
        # Predict surface
        return self.neural_field.predict(
            new_boundary,
            context=boundary.patch.surface,
            priming=self.current_priming
        )
```

### Tangent Continuity

When extending, maintain tangent continuity with existing surface:

```python
class TangentContinuityEnforcer:
    """
    Ensure new patches are tangent-continuous with existing geometry.
    """
    
    def enforce(self, new_field: SpectralField, 
                existing_patch: ConfirmedPatch,
                shared_boundary: Curve) -> SpectralField:
        """
        Modify new field to match tangent at shared boundary.
        """
        # Sample normals from existing patch at boundary
        existing_normals = existing_patch.sample_normals_at_boundary(shared_boundary)
        
        # Constrain new field's normals to match
        # This is a boundary condition for the FNO
        constrained_field = self.apply_normal_constraint(
            new_field,
            shared_boundary,
            existing_normals
        )
        
        return constrained_field
```

---

## 8. Implementation Architecture

### Module Structure

```
neural_sketch_field/
├── core/
│   ├── curve.py              # Curve parameterization, curvature sampling
│   ├── classifier.py         # Curve classification and primitive fitting
│   ├── spectral_field.py     # Resolution-independent field representation
│   └── mesh_extraction.py    # Marching cubes, dual contouring
│
├── neural/
│   ├── fno.py                # Fourier Neural Operator
│   ├── latent_encoder.py     # Mesh/text/symmetry → latent
│   ├── cross_attention.py    # Spatial-semantic alignment
│   └── pretrained/           # Pre-trained model weights
│
├── sketch/
│   ├── stroke_handler.py     # Raw input processing
│   ├── bilateral_mode.py     # Symmetric sketching
│   ├── reef_mode.py          # Surface-constrained sketching
│   └── ghost_scaffold.py     # Real-time preview
│
├── anticipation/
│   ├── anticipator.py        # Main anticipation logic
│   ├── confidence.py         # Confidence estimation
│   ├── alternatives.py       # Multiple interpretation handling
│   └── priming.py            # Latent prior management
│
├── extension/
│   ├── patch_manager.py      # Confirmed patch tracking
│   ├── boundary_extrusion.py # Patch extension
│   └── continuity.py         # Tangent/curvature continuity
│
└── ui/
    ├── maya_plugin.py        # Maya integration
    ├── blender_addon.py      # Blender integration
    └── standalone.py         # Standalone test app
```

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA FLOW                                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STROKE INPUT                                                              │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────┐                                                      │
│   │ StrokeHandler   │ → Raw points                                         │
│   └────────┬────────┘                                                      │
│            ▼                                                                │
│   ┌─────────────────┐                                                      │
│   │ CurveParameterizer │ → Arc-length parameterized curve                  │
│   └────────┬────────┘                                                      │
│            ▼                                                                │
│   ┌─────────────────┐                                                      │
│   │ CurvatureSampler │ → Curvature signature                               │
│   └────────┬────────┘                                                      │
│            ▼                                                                │
│   ┌─────────────────┐                                                      │
│   │ CurveClassifier │ → Classification + fitted primitive                  │
│   └────────┬────────┘                                                      │
│            │                                                                │
│            │  ┌─────────────────┐                                          │
│            │  │ LatentPrimer    │ → Latent prior (from mesh/text/symmetry) │
│            │  └────────┬────────┘                                          │
│            │           │                                                    │
│            ▼           ▼                                                    │
│   ┌─────────────────────────────┐                                          │
│   │ NeuralFieldAnticipator      │ → SpectralField                          │
│   │ (FNO + Cross-Attention)     │                                          │
│   └────────────┬────────────────┘                                          │
│                │                                                            │
│                ▼                                                            │
│   ┌─────────────────┐                                                      │
│   │ ConfidenceEstimator │ → Confidence score                               │
│   └────────┬────────┘                                                      │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                      │
│   │ GhostScaffold   │ → Visual preview                                     │
│   └────────┬────────┘                                                      │
│            │                                                                │
│            ▼                                                                │
│   USER CONFIRMS / REJECTS / REFINES                                        │
│            │                                                                │
│            ▼                                                                │
│   ┌─────────────────┐                                                      │
│   │ PatchManager    │ → Confirmed geometry + extrudable boundaries         │
│   └─────────────────┘                                                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Test Scenarios

### Test 1: Circle → Sphere

1. User draws closed curve
2. Classifier detects: circle
3. FNO predicts: hemisphere (minimal surface) or full sphere (inflation mode)
4. Ghost shows predicted surface
5. User confirms → bake to mesh

### Test 2: Rounded Rectangle → Box with Fillets

1. User draws closed curve with four rounded corners
2. Classifier detects: rounded rectangle (corner radius estimated from curvature)
3. FNO predicts: extruded box with matching fillet radius
4. Ghost shows predicted surface
5. User confirms → bake to mesh

### Test 3: Bilateral Character Profile

1. Enable bilateral mode
2. User draws profile of head (nose, forehead, chin) on one side
3. System mirrors to other side
4. FNO predicts: surface spanning the bilateral boundary
5. Ghost shows head shape
6. User refines with additional strokes
7. Each stroke → immediate re-anticipation

### Test 4: Reef Topology Detail

1. Start with base sphere
2. Enable reef mode
3. User draws closed curve on sphere surface
4. FNO predicts: extruded bump following base curvature
5. Ghost shows anticipated detail
6. User confirms → bump added to sphere
7. New boundaries available for further extension

### Test 5: Primed Anticipation

1. Load base mesh (e.g., car body)
2. System primes latent from base mesh shape
3. User draws curve on side panel region
4. Anticipation biased toward automotive-style surfaces
5. Ghost shows door panel shape (smooth, flowing)
6. User confirms → panel added

---

## 10. Training Requirements

### FNO Training Data

Generate synthetic training pairs:

```python
def generate_training_pair():
    """
    Generate (boundary, surface) pair for FNO training.
    """
    # Random primitive type
    primitive_type = random.choice(['sphere', 'cylinder', 'box', 'torus', 'blend'])
    
    # Generate surface
    surface = generate_primitive(primitive_type)
    
    # Extract random boundary on surface
    boundary = extract_random_boundary(surface)
    
    # Compute boundary features (curvature, position)
    features = compute_boundary_features(boundary)
    
    # Target: SDF or height field of surface
    target = compute_sdf(surface, resolution=(64, 64, 64))
    
    return features, target
```

### Cross-Attention Training

Align text/geometry via contrastive learning:

```python
def contrastive_loss(text_embedding, geometry_embedding, temperature=0.07):
    """
    CLIP-style contrastive loss for text-geometry alignment.
    """
    # Normalize
    text_embedding = F.normalize(text_embedding, dim=-1)
    geometry_embedding = F.normalize(geometry_embedding, dim=-1)
    
    # Similarity matrix
    logits = text_embedding @ geometry_embedding.T / temperature
    
    # Labels: diagonal is positive
    labels = torch.arange(len(logits), device=logits.device)
    
    # Cross-entropy both directions
    loss_t2g = F.cross_entropy(logits, labels)
    loss_g2t = F.cross_entropy(logits.T, labels)
    
    return (loss_t2g + loss_g2t) / 2
```

---

## 11. Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    NEURAL SKETCH-FIELD FRAMEWORK                            │
│                    Minimalist Test Bed for Generative Anticipation          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CORE LOOP                                                                 │
│   ─────────                                                                 │
│   Prime → Sketch → Anticipate → Confirm → Extend                           │
│                                                                             │
│   TWO MODES                                                                 │
│   ─────────                                                                 │
│   Bilateral: Free-form symmetric sketching                                 │
│   Reef: Drawing on existing surfaces                                       │
│                                                                             │
│   CURVE PIPELINE                                                            │
│   ──────────────                                                            │
│   Stroke → Parameterize → Curvature → Classify → Fit Primitive             │
│                                                                             │
│   NEURAL ANTICIPATION                                                       │
│   ───────────────────                                                       │
│   Boundary + Curvature + Context + Priming → FNO → SpectralField           │
│                                                                             │
│   PRIMING SOURCES                                                           │
│   ───────────────                                                           │
│   Base mesh, text description, symmetry specification                      │
│                                                                             │
│   GHOST SCAFFOLDING                                                         │
│   ─────────────────                                                         │
│   Real-time preview of anticipated geometry                                │
│   Confidence-based opacity and alternatives                                │
│                                                                             │
│   EXTENSION                                                                 │
│   ─────────                                                                 │
│   Confirmed patches → extrudable boundaries → tangent-continuous growth    │
│                                                                             │
│   THE GOAL                                                                  │
│   ────────                                                                  │
│   Validate: Neural operators can anticipate surfaces from boundaries       │
│   Validate: Latent priming guides anticipation meaningfully                │
│   Validate: The system suggests, the artist confirms                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Draw a boundary. The system anticipates the surface. Confirm or refine. Extend. Repeat.*

*This is sketch-based modeling where the tool knows geometry.*
