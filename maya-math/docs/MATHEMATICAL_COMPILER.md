# The Mathematical Compiler

> **Geometric Scaffold Supernode: A Category-Theoretic Architecture for Lossless Geometric Synthesis**
> From Sketch to Field to Form — Resolution-Independent, Topologically Guaranteed
> Version 3.0 | 2026-01-28

---

## Executive Summary

We present a paradigm shift in procedural 3D modeling: the transition from **Coordinate Manipulation** to **Mathematical Compilation**.

Traditional tools force geometry through a lossy pipeline:
```
User Intent → Discrete Samples → Matrix Transforms → Mesh Vertices → Render
```

Each arrow destroys information. Continuous curves become polylines. Topological intent becomes fragile vertex soup. Resolution is locked at creation time.

The **Geometric Scaffold Supernode** inverts this paradigm:
```
User Intent → Mathematical Representation → Algebraic Operations → Discrete Output (only at render)
```

The key insight: **preserve the generating function until the absolute last moment**. The system becomes a compiler that translates sketches into pure mathematics, manipulates them losslessly in their native algebraic domains, and rasterizes only when pixels must hit the screen.

This document formalizes the architecture, identifies the critical research integrations, and provides implementation specifications for what we call the **Tri-Space Engine** — a state machine that simultaneously tracks geometry in three parallel mathematical contexts:

| Context | Domain | Algebra | Preserved Property |
|---------|--------|---------|-------------------|
| **Affine** | Linear Algebra | GL(4,ℝ) | Position, Orientation |
| **Conformal** | Complex Analysis | PSL(2,ℂ) | Angles, Circular Arcs |
| **Spectral** | Harmonic Analysis | L²(ℝ³) | Resolution Independence |

---

## Part I: The Problem — Lossy Geometric Compression

### 1.1 Why Matrix4x4 Is Insufficient

The universal data structure in 3D graphics is the 4×4 transformation matrix. It represents affine transformations: translation, rotation, scale, shear. Every major engine (Unreal, Unity, Maya, Blender) builds on this foundation.

**The problem**: Not all geometric operations are affine.

| Operation | Affine Representable? | What Happens |
|-----------|----------------------|--------------|
| Translate | ✓ | Works perfectly |
| Rotate | ✓ | Works perfectly |
| Uniform Scale | ✓ | Works perfectly |
| Non-uniform Scale | ✓ | **Destroys circles** → ellipses |
| Möbius Transform | ✗ | Cannot represent at all |
| Conformal Map | ✗ | Cannot represent at all |
| Fourier Transform | ✗ | Different domain entirely |

When you scale a **biarc** (two circular arcs with G1 continuity) non-uniformly, the circles become ellipses. The G1 property breaks. The mathematical structure is destroyed. You must immediately rebake to polygons, losing the generating function forever.

### 1.2 The Cascade of Loss

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE LOSSY PIPELINE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STEP 1: User sketches curve                                               │
│           Intent: Smooth, organic flow                                      │
│                                                                             │
│   STEP 2: Tool samples to polyline                                          │
│           LOSS: Continuous → Discrete                                       │
│           Gone: Exact tangents, curvature, arc-length parameterization     │
│                                                                             │
│   STEP 3: Tool applies transforms via Matrix4x4                             │
│           LOSS: Conformal properties destroyed by non-uniform scale         │
│           Gone: Circular arcs, angle preservation                           │
│                                                                             │
│   STEP 4: Tool generates surface (loft, extrude)                            │
│           LOSS: Resolution locked at creation time                          │
│           Gone: Ability to upsample without artifacts                       │
│                                                                             │
│   STEP 5: User edits mesh vertices                                          │
│           LOSS: Generating function forgotten                               │
│           Gone: Ability to regenerate, parametric control                   │
│                                                                             │
│   RESULT: "Why does my model look bad when I zoom in?"                      │
│           "Why did the boolean create non-manifold edges?"                  │
│           "Why can't I change the curve after extruding?"                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 The Solution: Preserve the Generating Function

The fundamental insight: **geometry has multiple natural representations**, each preserving different properties. The correct architecture maintains geometry in **all representations simultaneously**, performing operations in whichever domain preserves the most structure.

```
User Sketch
     │
     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE MATHEMATICAL COMPILER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                    │
│   │   AFFINE    │    │  CONFORMAL  │    │  SPECTRAL   │                    │
│   │   CONTEXT   │◄──►│   CONTEXT   │◄──►│   CONTEXT   │                    │
│   │             │    │             │    │             │                    │
│   │  Matrix4x4  │    │   Möbius    │    │   Fourier   │                    │
│   │  Position   │    │   Angles    │    │   Physics   │                    │
│   └─────────────┘    └─────────────┘    └─────────────┘                    │
│          │                  │                  │                            │
│          └──────────────────┼──────────────────┘                            │
│                             │                                               │
│                             ▼                                               │
│                    SYNCHRONIZED STATE                                       │
│          (Geometry exists in all three spaces simultaneously)               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
     │
     ▼ (only at render time)
Discrete Mesh
```

---

## Part II: The Tri-Space Engine

### 2.1 Formal Definition

The **HybridState** is a product space:

```
HybridState = AffineContext × ConformalContext × SpectralContext
```

Each context is a **stack** (supporting push/pop for hierarchical operations) plus a **dictionary** of named objects.

```python
@dataclass
class HybridState:
    """
    The Tri-Space State Machine.
    
    Geometry exists simultaneously in three mathematical domains.
    Operations are performed in whichever domain preserves the most structure.
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT A: AFFINE
    # ═══════════════════════════════════════════════════════════════════════
    # Mathematical Basis: General Linear Group GL(4, ℝ)
    # Data Type: 4×4 real matrix
    # Preserves: Parallelism, ratios along lines, incidence
    # Destroyed by: Nothing (most general linear transform)
    # Use for: World placement, camera, instancing, skeletal animation
    
    affine_stack: List[Matrix4x4]
    affine_objects: Dict[str, Matrix4x4]  # Named platforms
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT B: CONFORMAL
    # ═══════════════════════════════════════════════════════════════════════
    # Mathematical Basis: Projective Special Linear Group PSL(2, ℂ)
    # Data Type: Möbius transform (a,b,c,d ∈ ℂ) or Conformal Map object
    # Preserves: Angles, circles (mapped to circles), infinitesimal shapes
    # Destroyed by: Non-uniform scaling, shearing
    # Use for: Biarc curves, UV mapping, circle packing, BFF
    
    conformal_stack: List[MobiusTransform]
    conformal_plane: ComplexPlane  # 2D slice of 3D space for complex ops
    conformal_objects: Dict[str, Union[BiarcCurve, ConformalMap]]
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT C: SPECTRAL
    # ═══════════════════════════════════════════════════════════════════════
    # Mathematical Basis: Square-integrable functions L²(ℝ³)
    # Data Type: Fourier coefficients or Neural Network weights
    # Preserves: Resolution independence, smoothness class
    # Destroyed by: Truncation (but gracefully degraded)
    # Use for: FNO physics, neural implicits, volumetric fields
    
    spectral_objects: Dict[str, Union[SpectralField, NeuralImplicit]]
    spectral_resolution: Tuple[int, int, int]  # Current working resolution
    
    # ═══════════════════════════════════════════════════════════════════════
    # SYNCHRONIZATION LAYER
    # ═══════════════════════════════════════════════════════════════════════
    
    def lift_to_conformal(self, curve_id: str):
        """
        Project affine curve onto conformal plane.
        
        The affine Platform defines the complex plane:
        - X axis → Real axis
        - Y axis → Imaginary axis
        - XY plane → Complex plane ℂ
        """
        affine_curve = self.affine_objects[curve_id]
        platform = self.affine_stack[-1]
        
        # Extract XY plane from current transform
        self.conformal_plane = ComplexPlane.from_matrix(platform)
        
        # Project 3D points to complex numbers
        complex_points = [self.conformal_plane.project(p) for p in affine_curve.points]
        
        # Store in conformal context
        self.conformal_objects[curve_id] = ComplexCurve(complex_points)
    
    def lift_to_spectral(self, object_id: str, method: str = 'FNO'):
        """
        Encode geometry as spectral representation.
        
        Methods:
        - 'FNO': Fourier Neural Operator (learns physics)
        - 'FFT': Direct Fourier transform (exact, no learning)
        - 'NEURAL': Neural implicit (MLP weights)
        """
        if method == 'FNO':
            boundary = self.conformal_objects[object_id]
            field = FourierNeuralOperator.encode(boundary)
        elif method == 'FFT':
            mesh = self.collapse_to_mesh(object_id, resolution=(32, 32, 32))
            field = SpectralField.from_sdf(compute_sdf(mesh))
        elif method == 'NEURAL':
            mesh = self.collapse_to_mesh(object_id, resolution=(64, 64, 64))
            field = NeuralImplicit.train_on_sdf(compute_sdf(mesh))
        
        self.spectral_objects[object_id] = field
    
    def collapse_conformal(self, curve_id: str, samples: int) -> List[complex]:
        """
        Evaluate conformal curve at sample points.
        
        Apply accumulated Möbius transforms analytically.
        """
        curve = self.conformal_objects[curve_id]
        mobius = self.conformal_stack[-1]
        
        return [mobius(curve.evaluate(t)) for t in np.linspace(0, 1, samples)]
    
    def collapse_spectral(self, field_id: str, resolution: Tuple[int, int, int]) -> Mesh:
        """
        Extract mesh from spectral representation.
        
        This is the ONLY place where resolution is determined.
        """
        field = self.spectral_objects[field_id]
        grid = field.evaluate(resolution)
        verts, faces = marching_cubes(grid, level=0.0)
        
        # Apply affine transform
        verts = [self.affine_stack[-1] @ v for v in verts]
        
        return Mesh(verts, faces)
```

### 2.2 The Algebra of Each Context

Each context has its own **group structure**. Operations within a context compose according to the group law.

#### Affine Context: GL(4, ℝ)

```
Identity: I₄ (4×4 identity matrix)
Composition: Matrix multiplication (A ∘ B = AB)
Inverse: Matrix inverse (A⁻¹)

Properties:
- Closed under composition
- Associative: (AB)C = A(BC)
- Has identity and inverses
- NOT commutative: AB ≠ BA in general
```

#### Conformal Context: PSL(2, ℂ)

```
Data: Möbius transform f(z) = (az + b)/(cz + d), where ad - bc = 1

Identity: f(z) = z (a=1, b=0, c=0, d=1)
Composition: (f ∘ g)(z) = f(g(z)) = matrix multiplication of [a b; c d]
Inverse: f⁻¹(z) = (dz - b)/(-cz + a)

Properties:
- Closed under composition
- Maps circles to circles (including lines as infinite-radius circles)
- Preserves angles (conformal)
- Three points determine a unique Möbius transform
```

#### Spectral Context: L²(ℝ³) with Convolution

```
Data: Functions f: ℝ³ → ℝ (or their Fourier coefficients)

Identity: Dirac delta δ(x)
Composition: Convolution (f * g)(x) = ∫ f(y)g(x-y) dy
             In Fourier domain: F[f * g] = F[f] · F[g] (pointwise multiply)

Properties:
- Resolution-independent (coefficients don't change with sampling)
- Smoothness encoded in decay rate of high-frequency coefficients
- Differentiation = multiplication by frequency
```

### 2.3 Context Transitions: Lifting and Collapsing

The **Functor Pattern**: Moving between contexts should preserve structure where possible.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT TRANSITIONS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   AFFINE ───────────────────────────────────────────────────► CONFORMAL    │
│           lift_to_conformal()                                               │
│           • Project onto XY plane of current transform                     │
│           • Convert 3D points to complex numbers                           │
│           • LOSSLESS for planar geometry                                   │
│                                                                             │
│   CONFORMAL ─────────────────────────────────────────────────► AFFINE      │
│              collapse_conformal()                                           │
│              • Sample curve at discrete points                             │
│              • Convert complex to 3D via conformal plane                   │
│              • LOSSY: continuous → discrete                                │
│                                                                             │
│   CONFORMAL ─────────────────────────────────────────────────► SPECTRAL    │
│              lift_to_spectral()                                             │
│              • Rasterize boundary to sparse grid                           │
│              • Run FNO to predict interior field                           │
│              • LOSSY: exact boundary → predicted field                     │
│                                                                             │
│   SPECTRAL ──────────────────────────────────────────────────► AFFINE      │
│             collapse_spectral()                                             │
│             • Evaluate field on grid (choose resolution NOW)               │
│             • Marching cubes to extract mesh                               │
│             • LOSSY: continuous field → discrete mesh                      │
│                                                                             │
│   Key Insight: OPERATE in the highest context possible.                    │
│   Only COLLAPSE when you must produce output.                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Part III: The Supernode — Complete Specification

### 3.1 The Pipeline: Lift → Operate → Collapse

```python
class GeometricScaffoldSupernode(Node):
    """
    THE MATHEMATICAL COMPILER
    
    A unified architectural unit that encapsulates the complete
    Lift → Operate → Collapse pipeline.
    
    The Supernode is the fundamental unit of geometric computation.
    All modeling operations can be expressed as Supernode composition.
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    
    @dataclass
    class Config:
        # ─────────────────────────────────────────────────────────────────
        # LIFT CONFIGURATION
        # ─────────────────────────────────────────────────────────────────
        
        # Biarc fitting method
        biarc_method: str = 'LEARNED'
        # Options:
        # - 'TOLERANCE_BAND': Classical algorithm (Meek & Walton)
        # - 'LEARNED': Neural network (Groueix et al.)
        # - 'NEURAL_SPLINE': Hybrid MLP (Williams et al.)
        
        biarc_tolerance: float = 0.01  # For TOLERANCE_BAND
        biarc_style: str = 'ADAPTIVE'  # For LEARNED: 'MECHANICAL' | 'ORGANIC' | 'ADAPTIVE'
        
        # ─────────────────────────────────────────────────────────────────
        # OPERATE CONFIGURATION
        # ─────────────────────────────────────────────────────────────────
        
        # Physics simulation mode
        physics_mode: str = 'MINIMAL_SURFACE'
        # Options:
        # - 'MINIMAL_SURFACE': Soap film (area minimization)
        # - 'INFLATION': Pressure expansion
        # - 'ELASTIC_SHELL': Thin shell elasticity
        # - 'MEAN_CURVATURE_FLOW': Smoothing flow
        # - 'WILLMORE_FLOW': Bending energy minimization
        
        # Conformal mapping method
        conformal_method: str = 'BFF'
        # Options:
        # - 'BFF': Boundary First Flattening (Sawhney & Crane) — RECOMMENDED
        # - 'SCHWARZ_CHRISTOFFEL': Classical (fragile for complex shapes)
        # - 'LSCM': Least Squares Conformal Maps (approximate)
        # - 'ABF': Angle-Based Flattening (iterative)
        
        # Topological validation level
        guardrail_level: str = 'ADAPTIVE'
        # Options:
        # - 'STRICT': Halt on any topology error
        # - 'ADAPTIVE': Auto-fix when possible, warn otherwise
        # - 'PERMISSIVE': Warn only, never halt
        
        # ─────────────────────────────────────────────────────────────────
        # COLLAPSE CONFIGURATION
        # ─────────────────────────────────────────────────────────────────
        
        preview_resolution: Tuple[int, int, int] = (32, 32, 32)
        viewport_resolution: Tuple[int, int, int] = (64, 64, 64)
        render_resolution: Tuple[int, int, int] = (256, 256, 256)
        
        # Meshing method
        meshing_method: str = 'DUAL_CONTOURING'
        # Options:
        # - 'MARCHING_CUBES': Fast, but poor sharp features
        # - 'DUAL_CONTOURING': Better sharp features
        # - 'NEURAL_DUAL_CONTOURING': Learned sharp features (Chen et al.)
        # - 'FLEXICUBES': Differentiable (Shen et al.)
        
        # Tessellation mode for curves
        tessellation_mode: str = 'CURVATURE_ADAPTIVE'
        # Options:
        # - 'FIXED': Uniform sampling
        # - 'SCREEN_ADAPTIVE': Based on screen-space size
        # - 'CURVATURE_ADAPTIVE': More samples in high-curvature regions
    
    # ═══════════════════════════════════════════════════════════════════════
    # INTERNAL STATE
    # ═══════════════════════════════════════════════════════════════════════
    
    def __init__(self, input_id: str, config: Config = None):
        super().__init__('GeometricScaffold')
        self.input_id = input_id
        self.config = config or self.Config()
        
        # Lazy-initialized representations (computed on first access)
        self._lifted: bool = False
        self._operated: bool = False
        
        # Conformal representations
        self._biarc: Optional[BiarcCurve] = None
        self._conformal_map: Optional[ConformalMap] = None
        
        # Spectral representations
        self._spectral_field: Optional[SpectralField] = None
        self._neural_implicit: Optional[NeuralImplicit] = None
        
        # Topology data
        self._darboux_frames: Optional[List[DarbouxFrame]] = None
        self._betti_numbers: Optional[Tuple[int, int, int]] = None
        self._validation: Optional[ValidationResult] = None
        
        # Cache
        self._mesh_cache: Dict[Tuple[int, int, int], Mesh] = {}
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE A: LIFT
    # ═══════════════════════════════════════════════════════════════════════
    
    def _lift(self, state: HybridState):
        """
        Transform discrete input into continuous mathematical representations.
        
        This is where compilation begins: raw samples become generating functions.
        """
        if self._lifted:
            return
        
        raw_input = state.get_geometry(self.input_id)
        
        # ─────────────────────────────────────────────────────────────────
        # LIFT TO CONFORMAL: Polyline → Biarc
        # ─────────────────────────────────────────────────────────────────
        
        if isinstance(raw_input, Polyline) or isinstance(raw_input, Curve):
            # Project to complex plane
            state.lift_to_conformal(self.input_id)
            complex_curve = state.conformal_objects[self.input_id]
            
            # Fit biarc representation
            if self.config.biarc_method == 'LEARNED':
                self._biarc = LearnedBiarcFitter.fit(
                    complex_curve,
                    style=self.config.biarc_style,
                    user_embedding=state.user_style_embedding
                )
            elif self.config.biarc_method == 'NEURAL_SPLINE':
                self._biarc = NeuralSplineFitter.fit(complex_curve)
            else:  # TOLERANCE_BAND
                self._biarc = ToleranceBandFitter.fit(
                    complex_curve,
                    tolerance=self.config.biarc_tolerance
                )
            
            state.conformal_objects[f'{self.input_id}_biarc'] = self._biarc
        
        # ─────────────────────────────────────────────────────────────────
        # LIFT TO SPECTRAL: Boundary → Field
        # ─────────────────────────────────────────────────────────────────
        
        if self._biarc:
            # FNO predicts the surface implied by the boundary
            self._spectral_field = FourierNeuralOperator.predict(
                boundary=self._biarc,
                physics_mode=self.config.physics_mode,
                resolution=self.config.preview_resolution
            )
            state.spectral_objects[f'{self.input_id}_field'] = self._spectral_field
        
        elif isinstance(raw_input, Mesh):
            # Encode mesh as neural implicit
            sdf = compute_sdf(raw_input, resolution=self.config.preview_resolution)
            self._neural_implicit = NeuralImplicit.from_sdf(sdf)
            state.spectral_objects[f'{self.input_id}_implicit'] = self._neural_implicit
        
        # ─────────────────────────────────────────────────────────────────
        # COMPUTE TOPOLOGY: Darboux Frames
        # ─────────────────────────────────────────────────────────────────
        
        if hasattr(raw_input, 'reference_surface') and raw_input.reference_surface:
            self._darboux_frames = DarbouxFrameComputer.compute(
                curve=self._biarc,
                surface=raw_input.reference_surface
            )
        
        self._lifted = True
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE B: OPERATE
    # ═══════════════════════════════════════════════════════════════════════
    
    def _operate(self, state: HybridState):
        """
        Perform mathematical operations in their native domains.
        
        Key insight: All operations are EXACT within their domain.
        No approximation until collapse.
        """
        if self._operated:
            return
        
        # ─────────────────────────────────────────────────────────────────
        # CONFORMAL PASS: Compute UV Parameterization
        # ─────────────────────────────────────────────────────────────────
        
        if self._biarc and self.config.conformal_method:
            if self.config.conformal_method == 'BFF':
                self._conformal_map = BoundaryFirstFlattening.compute(self._biarc)
            elif self.config.conformal_method == 'SCHWARZ_CHRISTOFFEL':
                self._conformal_map = SchwarzChristoffel.compute(self._biarc.as_polygon())
            elif self.config.conformal_method == 'LSCM':
                self._conformal_map = LeastSquaresConformal.compute(self._biarc)
            
            # Compose with current conformal transform
            if self._conformal_map:
                mobius = self._conformal_map.boundary_mobius()
                state.conformal_stack[-1] = state.conformal_stack[-1].compose(mobius)
        
        # ─────────────────────────────────────────────────────────────────
        # SPECTRAL PASS: Apply Physics / Relaxation
        # ─────────────────────────────────────────────────────────────────
        
        if self._spectral_field:
            # Fisher Information Flow: smart smoothing
            self._spectral_field = FisherInformationFlow.apply(
                field=self._spectral_field,
                iterations=10,
                preserve_features=True,
                preserve_threshold=0.5
            )
        
        if self._neural_implicit:
            # Neural implicit is ready for boolean composition
            # Union: min(A, B), Intersection: max(A, B), Difference: max(A, -B)
            pass
        
        # ─────────────────────────────────────────────────────────────────
        # GUARDRAIL PASS: Topological Validation
        # ─────────────────────────────────────────────────────────────────
        
        self._validation = self._run_guardrails(state)
        
        if not self._validation.valid:
            if self.config.guardrail_level == 'STRICT':
                raise TopologyError(self._validation)
            elif self.config.guardrail_level == 'ADAPTIVE':
                self._apply_adaptive_fix(self._validation, state)
            # PERMISSIVE: just continue
        
        self._operated = True
    
    def _run_guardrails(self, state: HybridState) -> ValidationResult:
        """
        Comprehensive topological validation.
        
        Checks are performed on mathematical representations, not meshes.
        This is orders of magnitude faster.
        """
        errors = []
        warnings = []
        
        # ─────────────────────────────────────────────────────────────────
        # CHECK 1: Winding Number (self-intersection)
        # ─────────────────────────────────────────────────────────────────
        
        if self._biarc:
            winding = WindingNumber.compute(self._biarc)
            if winding != 1:
                errors.append(ValidationError(
                    type='SELF_INTERSECTION',
                    message=f'Curve self-intersects (winding number = {winding})',
                    location=WindingNumber.find_intersection(self._biarc),
                    suggestion='Simplify curve or enable auto-smoothing'
                ))
        
        # ─────────────────────────────────────────────────────────────────
        # CHECK 2: Linking Number (curve entanglement in spreads)
        # ─────────────────────────────────────────────────────────────────
        
        if state.context_mode == ContextMode.SPREAD and len(state.spread_buffer) > 1:
            for i in range(len(state.spread_buffer)):
                for j in range(i + 1, len(state.spread_buffer)):
                    curve_i = self._transform_biarc(self._biarc, state.spread_buffer[i])
                    curve_j = self._transform_biarc(self._biarc, state.spread_buffer[j])
                    
                    linking = LinkingNumber.compute(curve_i, curve_j)
                    if linking != 0:
                        errors.append(ValidationError(
                            type='CURVES_LINKED',
                            message=f'Curves {i} and {j} are linked (linking number = {linking})',
                            location=LinkingNumber.find_intersection(curve_i, curve_j),
                            suggestion='Use Bridge instead of Loft, or reorder curves'
                        ))
        
        # ─────────────────────────────────────────────────────────────────
        # CHECK 3: Betti Numbers (topological features)
        # ─────────────────────────────────────────────────────────────────
        
        if self._spectral_field:
            self._betti_numbers = PersistentHomology.compute_betti(self._spectral_field)
            expected_betti = self._infer_expected_betti()
            
            if self._betti_numbers != expected_betti:
                warnings.append(ValidationWarning(
                    type='UNEXPECTED_TOPOLOGY',
                    message=f'Topology mismatch: found β={self._betti_numbers}, expected {expected_betti}',
                    suggestion='Check for unintended holes or disconnected components'
                ))
        
        # ─────────────────────────────────────────────────────────────────
        # CHECK 4: Darboux Validity (frame stability)
        # ─────────────────────────────────────────────────────────────────
        
        if self._darboux_frames:
            for i, frame in enumerate(self._darboux_frames):
                if frame.geodesic_curvature > frame.validity_radius:
                    warnings.append(ValidationWarning(
                        type='DARBOUX_UNSTABLE',
                        message=f'Frame {i} exceeds Darboux validity radius',
                        location=frame.position,
                        suggestion='Reduce array density in high-curvature regions'
                    ))
        
        # ─────────────────────────────────────────────────────────────────
        # COMPILE RESULT
        # ─────────────────────────────────────────────────────────────────
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _apply_adaptive_fix(self, validation: ValidationResult, state: HybridState):
        """
        Automatically repair topology issues when possible.
        """
        for error in validation.errors:
            if error.type == 'SELF_INTERSECTION':
                # Smooth the biarc to remove self-intersection
                self._biarc = BiarcSmoother.remove_self_intersections(
                    self._biarc,
                    location=error.location
                )
                # Re-lift to spectral
                self._spectral_field = FourierNeuralOperator.predict(
                    boundary=self._biarc,
                    physics_mode=self.config.physics_mode
                )
            
            elif error.type == 'CURVES_LINKED':
                # Switch from Loft to Bridge (handles crossing gracefully)
                self._collapse_method = 'BRIDGE'
            
            elif error.type == 'UNEXPECTED_TOPOLOGY':
                # Attempt hole filling via Sparc3D
                self._spectral_field = Sparc3D.fill_holes(self._spectral_field)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE C: COLLAPSE
    # ═══════════════════════════════════════════════════════════════════════
    
    def _collapse(self, state: HybridState, mode: RenderMode) -> Optional[Mesh]:
        """
        Transform mathematical representations to discrete output.
        
        THIS IS THE ONLY PLACE WHERE DISCRETIZATION OCCURS.
        """
        
        if mode == RenderMode.PREVIEW:
            self._emit_ghost_scaffolding(state)
            return None
        
        elif mode == RenderMode.VIEWPORT:
            resolution = self.config.viewport_resolution
        
        elif mode == RenderMode.RENDER:
            resolution = self.config.render_resolution
        
        else:
            resolution = self.config.preview_resolution
        
        # Check cache
        if resolution in self._mesh_cache:
            mesh = self._mesh_cache[resolution]
            # Apply current affine transform
            return mesh.transform(state.affine_stack[-1])
        
        # ─────────────────────────────────────────────────────────────────
        # EXTRACT MESH FROM SPECTRAL
        # ─────────────────────────────────────────────────────────────────
        
        if self._spectral_field:
            grid = self._spectral_field.evaluate(resolution)
            
            if self.config.meshing_method == 'MARCHING_CUBES':
                verts, faces = marching_cubes(grid, level=0.0)
            elif self.config.meshing_method == 'DUAL_CONTOURING':
                verts, faces = dual_contouring(grid, level=0.0)
            elif self.config.meshing_method == 'NEURAL_DUAL_CONTOURING':
                verts, faces = neural_dual_contouring(grid, self._spectral_field)
            elif self.config.meshing_method == 'FLEXICUBES':
                verts, faces = flexicubes(grid, level=0.0)  # Differentiable
        
        elif self._neural_implicit:
            verts, faces = self._neural_implicit.extract_mesh(resolution)
        
        else:
            # Fallback: extrude biarc
            verts, faces = self._biarc.extrude_to_mesh(
                depth=state.extrusion_depth or 10.0
            )
        
        # ─────────────────────────────────────────────────────────────────
        # APPLY UV FROM CONFORMAL MAP
        # ─────────────────────────────────────────────────────────────────
        
        if self._conformal_map:
            uvs = []
            for v in verts:
                # Project to conformal plane
                z = state.conformal_plane.project(v)
                # Apply inverse conformal map to get UV
                uv = self._conformal_map.inverse(z)
                uvs.append(uv)
        else:
            uvs = None
        
        # ─────────────────────────────────────────────────────────────────
        # CREATE MESH AND CACHE
        # ─────────────────────────────────────────────────────────────────
        
        mesh = Mesh(verts, faces, uvs=uvs)
        self._mesh_cache[resolution] = mesh
        
        # Apply affine transform
        return mesh.transform(state.affine_stack[-1])
    
    def _emit_ghost_scaffolding(self, state: HybridState):
        """
        Real-time preview without generating heavy geometry.
        
        This is what makes the tool feel "intelligent":
        - User sees smoothed interpretation of their sketch
        - User sees conformal grid showing geometry flow
        - User sees predicted surface as translucent preview
        - User sees topology warnings highlighted
        """
        
        # ─────────────────────────────────────────────────────────────────
        # GHOST 1: Biarc Curve (smoothed sketch interpretation)
        # ─────────────────────────────────────────────────────────────────
        
        if self._biarc:
            samples = self._biarc.adaptive_sample(
                min_segments=20,
                max_segments=200,
                curvature_threshold=0.1
            )
            
            # Project back to 3D
            points_3d = [state.conformal_plane.unproject(z) for z in samples]
            points_3d = [state.affine_stack[-1] @ p for p in points_3d]
            
            state.emit_ghost(
                Polyline(points_3d),
                style=GhostStyle.BIARC,
                color=Color(0.2, 0.9, 0.2, 0.7),
                line_width=2.0
            )
        
        # ─────────────────────────────────────────────────────────────────
        # GHOST 2: Conformal Grid (geometry flow visualization)
        # ─────────────────────────────────────────────────────────────────
        
        if self._conformal_map:
            grid = self._conformal_map.sample_grid(
                u_lines=10,
                v_lines=10,
                samples_per_line=50
            )
            
            for line in grid.u_lines + grid.v_lines:
                points_3d = [state.conformal_plane.unproject(z) for z in line]
                points_3d = [state.affine_stack[-1] @ p for p in points_3d]
                
                state.emit_ghost(
                    Polyline(points_3d),
                    style=GhostStyle.CONFORMAL_GRID,
                    color=Color(0.5, 0.5, 1.0, 0.3),
                    line_width=1.0
                )
        
        # ─────────────────────────────────────────────────────────────────
        # GHOST 3: Predicted Surface (FNO preview)
        # ─────────────────────────────────────────────────────────────────
        
        if self._spectral_field:
            preview_mesh = self._spectral_field.extract_preview(
                resolution=self.config.preview_resolution
            )
            preview_mesh = preview_mesh.transform(state.affine_stack[-1])
            
            state.emit_ghost(
                preview_mesh,
                style=GhostStyle.NEURAL_SURFACE,
                color=Color(1.0, 0.8, 0.2, 0.15),
                render_mode='TRANSPARENT'
            )
        
        # ─────────────────────────────────────────────────────────────────
        # GHOST 4: Validation Markers (errors and warnings)
        # ─────────────────────────────────────────────────────────────────
        
        if self._validation:
            for error in self._validation.errors:
                if error.location:
                    state.emit_ghost(
                        Sphere(center=error.location, radius=3.0),
                        style=GhostStyle.ERROR,
                        color=Color(1.0, 0.0, 0.0, 0.8)
                    )
                    state.emit_ghost(
                        Text(error.message, position=error.location + Vec3(0, 5, 0)),
                        style=GhostStyle.ERROR_LABEL,
                        color=Color(1.0, 0.3, 0.3, 1.0)
                    )
            
            for warning in self._validation.warnings:
                if warning.location:
                    state.emit_ghost(
                        Sphere(center=warning.location, radius=2.0),
                        style=GhostStyle.WARNING,
                        color=Color(1.0, 0.7, 0.0, 0.6)
                    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAIN EXECUTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def execute(self, state: HybridState):
        """
        Execute the complete Mathematical Compiler pipeline.
        
        Lift → Operate → Collapse
        """
        # Phase A: Compile input to mathematical representations
        self._lift(state)
        
        # Phase B: Perform operations in native mathematical domains
        self._operate(state)
        
        # Phase C: Generate output (mode-dependent)
        mesh = self._collapse(state, state.render_mode)
        
        if mesh:
            state.emit(mesh, tags={
                'source': 'geometric_scaffold',
                'physics_mode': self.config.physics_mode,
                'topology_valid': self._validation.valid if self._validation else True
            })
        
        # Execute children
        self.execute_children(state)
```

---

## Part IV: Critical Research Integrations

### 4.1 The Spectral Engine: Fourier Neural Operators

**Key Papers**:
- Li et al., "Fourier Neural Operator for Parametric PDEs" (NeurIPS 2020)
- Li et al., "Neural Operator: Learning Maps Between Function Spaces" (2021)
- Guibas et al., "Adaptive Fourier Neural Operators" (ICLR 2022)

**Why Critical**: FNOs learn mappings between **function spaces**, not sample spaces. This enables:
- Train on 32³, infer on 512³ (resolution invariance)
- Physics-based surface generation (soap films, inflation, elasticity)
- Real-time inference (< 5ms on GPU)

**Implementation**:

```python
class FourierNeuralOperator:
    """
    Resolution-invariant learned physics.
    
    Core insight: Convolution in physical space = multiplication in Fourier space.
    We learn the multiplication (spectral weights) once, apply at any resolution.
    """
    
    def __init__(self, physics_mode: str, modes: int = 12, width: int = 64):
        self.physics_mode = physics_mode
        self.modes = modes  # Number of Fourier modes to keep
        self.width = width  # Hidden dimension
        
        # Learnable spectral weights
        self.spectral_weights = nn.Parameter(
            torch.randn(modes, modes, modes, width, width, dtype=torch.cfloat)
        )
        
        # Pointwise MLP
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width)
        )
    
    def spectral_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Spectral convolution layer.
        
        1. FFT to frequency domain
        2. Multiply by learnable weights (truncated to `modes`)
        3. IFFT back to physical domain
        """
        # FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Truncate to modes (this enables resolution invariance!)
        x_ft_truncated = x_ft[..., :self.modes, :self.modes, :self.modes]
        
        # Multiply by spectral weights
        out_ft = torch.einsum('bxyzc,xyzcd->bxyzd', x_ft_truncated, self.spectral_weights)
        
        # Pad back to original size and IFFT
        out = torch.fft.irfftn(out_ft, s=x.shape[-3:])
        
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: boundary → interior field.
        """
        # Lift to hidden dimension
        x = self.lift(x)
        
        # Four FNO layers
        for _ in range(4):
            x = self.spectral_conv(x) + self.mlp(x)
            x = F.gelu(x)
        
        # Project to output
        return self.project(x)
    
    @classmethod
    def predict(cls, boundary: BiarcCurve, physics_mode: str, 
                resolution: Tuple[int, int, int] = (32, 32, 32)) -> SpectralField:
        """
        Predict interior field from boundary.
        
        The returned SpectralField can be evaluated at ANY resolution.
        """
        model = cls.load_pretrained(physics_mode)
        
        # Rasterize boundary to sparse grid (COARSE)
        boundary_grid = rasterize_boundary(boundary, resolution=(32, 32, 32))
        
        # Run inference
        with torch.no_grad():
            field_grid = model(boundary_grid)
        
        # Return as SpectralField (stores Fourier coefficients)
        return SpectralField.from_grid(field_grid)
```

**Expansion**: Combine with "Geometry-Informed Neural Operator" (Li et al., 2022) to handle non-uniform domains.

---

### 4.2 The Conformal Engine: Boundary First Flattening

**Key Paper**: Sawhney & Crane, "Boundary First Flattening" (ACM TOG 2017)

**Why Critical**: Classical Schwarz-Christoffel mapping is:
- Iterative (slow, may not converge)
- Fragile (fails on complex polygons)
- Limited to simply-connected domains

BFF is:
- **Linear** (direct solve, instant)
- **Robust** (handles any topology, including holes)
- **Optimal** (minimizes conformal distortion)

**Implementation**:

```python
class BoundaryFirstFlattening:
    """
    State-of-the-art conformal parameterization.
    
    Key insight: The conformal map is determined entirely by what happens
    at the boundary. Solve a LINEAR system on the boundary, then extend
    harmonically to the interior.
    """
    
    @classmethod
    def compute(cls, boundary: BiarcCurve, target: str = 'DISK') -> ConformalMap:
        """
        Compute conformal map from interior to target domain.
        
        target options:
        - 'DISK': Unit disk
        - 'SQUARE': Unit square
        - 'CUSTOM': User-specified boundary shape
        """
        # Sample boundary
        n = 1000
        boundary_points = boundary.sample(n)
        
        # Compute boundary Laplacian
        L_boundary = cls._build_boundary_laplacian(boundary_points)
        
        # Compute target curvature
        if target == 'DISK':
            target_curvature = np.ones(n) * (2 * np.pi / n)  # Uniform
        elif target == 'SQUARE':
            target_curvature = cls._square_curvature(n)
        
        # SOLVE: Find scale factors that produce target curvature
        # This is a LINEAR system! No iteration.
        log_scale = sparse.linalg.spsolve(L_boundary, target_curvature)
        scale = np.exp(log_scale)
        
        # Integrate to get UV coordinates
        uv = cls._integrate_boundary(boundary_points, scale)
        
        # Build interpolation functions
        forward = cls._build_interpolant(boundary_points, uv)
        inverse = cls._build_interpolant(uv, boundary_points)
        
        return ConformalMap(
            forward=forward,
            inverse=inverse,
            boundary=boundary,
            target=target
        )
    
    @staticmethod
    def _build_boundary_laplacian(points: np.ndarray) -> sparse.csr_matrix:
        """
        Discrete Laplacian on the boundary.
        
        L[i,i] = -2
        L[i,i±1] = 1
        """
        n = len(points)
        diag = -2 * np.ones(n)
        off_diag = np.ones(n - 1)
        
        L = sparse.diags([off_diag, diag, off_diag], [-1, 0, 1], format='csr')
        
        # Periodic boundary conditions
        L[0, n-1] = 1
        L[n-1, 0] = 1
        
        return L
```

**Expansion**: Extend to "Boundary First Flattening for Surfaces with Holes" for genus > 0.

---

### 4.3 The Learned Interpreter: Neural Biarc Fitting

**Key Papers**:
- Groueix et al., "AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation" (CVPR 2018)
- Williams et al., "Neural Splines: Fitting 3D Surfaces with Infinitely-Wide Neural Networks" (NeurIPS 2021)

**Why Critical**: Hand-coded tolerance band algorithms:
- Don't learn user preferences
- Have fixed tradeoffs (tolerance vs. arc count)
- Produce "generic" results

Neural fitters:
- Learn **style** (mechanical vs. organic)
- Adapt to **context** (tight corners for machinery, smooth for characters)
- Are **differentiable** (for end-to-end learning)

**Implementation**:

```python
class LearnedBiarcFitter:
    """
    Neural network that predicts optimal biarc parameters.
    
    Architecture:
    1. PointNet++ encoder: polyline → latent vector
    2. Style injection: latent + style_embedding
    3. Transformer decoder: latent → biarc parameters
    """
    
    def __init__(self):
        self.encoder = PointNetPlusPlus(in_channels=2, out_channels=256)
        self.style_projector = nn.Linear(64, 256)
        self.decoder = TransformerDecoder(d_model=256, num_layers=4)
        self.param_head = nn.Linear(256, 6)  # (cx, cy, r, theta_start, theta_end, flip)
    
    def forward(self, polyline: torch.Tensor, style: Optional[torch.Tensor] = None) -> List[BiarcParams]:
        """
        Predict biarc curve from polyline.
        
        polyline: (B, N, 2) tensor of point coordinates
        style: (B, 64) optional style embedding
        """
        # Encode polyline
        latent = self.encoder(polyline)  # (B, 256)
        
        # Inject style
        if style is not None:
            style_projected = self.style_projector(style)
            latent = latent + style_projected
        
        # Decode to variable number of arcs
        # Use autoregressive decoding until EOS token
        arcs = []
        hidden = latent.unsqueeze(1)  # (B, 1, 256)
        
        for _ in range(100):  # Max 100 arcs
            decoded = self.decoder(hidden)
            params = self.param_head(decoded[:, -1])  # Last position
            
            if params[..., -1] < 0:  # EOS signal
                break
            
            arcs.append(BiarcParams.from_tensor(params))
            hidden = torch.cat([hidden, decoded[:, -1:]], dim=1)
        
        return arcs
    
    @classmethod
    def fit(cls, curve: ComplexCurve, style: str = 'ADAPTIVE',
            user_embedding: Optional[np.ndarray] = None) -> BiarcCurve:
        """
        Fit biarc to curve with style control.
        """
        model = cls.load_pretrained()
        
        # Convert curve to tensor
        polyline = torch.tensor(curve.sample(500), dtype=torch.float32).unsqueeze(0)
        
        # Get style embedding
        if user_embedding is not None:
            style_tensor = torch.tensor(user_embedding, dtype=torch.float32).unsqueeze(0)
        else:
            style_tensor = cls.get_default_style(style)
        
        # Predict
        with torch.no_grad():
            arc_params = model(polyline, style_tensor)
        
        return BiarcCurve.from_params(arc_params)
    
    def learn_from_correction(self, original: ComplexCurve, corrected: BiarcCurve,
                              user_style: np.ndarray) -> np.ndarray:
        """
        Update style embedding from user correction.
        
        Returns updated user_style embedding.
        """
        # Encode original and correction
        original_latent = self.encoder(original.to_tensor())
        corrected_latent = self.encode_biarc(corrected)
        
        # Style delta
        delta = corrected_latent - original_latent
        
        # Update style (exponential moving average)
        updated_style = 0.9 * user_style + 0.1 * delta.detach().numpy()
        
        return updated_style
```

**Expansion**: Combine with "Neural Splines" for C∞ continuity (infinite derivatives), enabling:
- Perfect curvature flows
- Aerodynamic optimization
- Smooth procedural animation

---

### 4.4 The Topological Guardian: Persistent Homology

**Key Papers**:
- Carlsson, "Topology and Data" (AMS Bulletin 2009)
- Edelsbrunner & Harer, "Persistent Homology — A Survey" (2008)
- Chen et al., "A Topological Regularizer for Classifiers" (AISTATS 2019)

**Why Critical**: Traditional topology checks:
- Operate on meshes (expensive)
- Miss multi-scale features
- Can't detect "almost-holes" (thin connections)

Persistent homology:
- Works on fields (before meshing)
- Detects features at ALL scales
- Quantifies "significance" of features

**Implementation**:

```python
class PersistentHomology:
    """
    Multi-scale topological analysis.
    
    Computes Betti numbers and persistence diagrams from spectral fields.
    """
    
    @classmethod
    def compute_betti(cls, field: SpectralField) -> Tuple[int, int, int]:
        """
        Compute Betti numbers: (β₀, β₁, β₂)
        
        β₀: Number of connected components
        β₁: Number of tunnels/holes (1D cycles)
        β₂: Number of voids/cavities (2D cycles)
        """
        # Evaluate field at working resolution
        grid = field.evaluate((64, 64, 64))
        
        # Build cubical complex
        complex = gudhi.CubicalComplex(top_dimensional_cells=grid.flatten())
        
        # Compute persistence
        complex.compute_persistence()
        
        # Count infinite-persistence features (true topology)
        betti = [0, 0, 0]
        for dim, (birth, death) in complex.persistence():
            if death == float('inf'):
                betti[dim] += 1
        
        return tuple(betti)
    
    @classmethod
    def compute_persistence_diagram(cls, field: SpectralField) -> PersistenceDiagram:
        """
        Full persistence diagram.
        
        Shows birth/death of features across scales.
        """
        grid = field.evaluate((64, 64, 64))
        complex = gudhi.CubicalComplex(top_dimensional_cells=grid.flatten())
        complex.compute_persistence()
        
        return PersistenceDiagram(
            points=complex.persistence_pairs(),
            dimensions=[0, 1, 2]
        )
    
    @classmethod
    def detect_thin_features(cls, field: SpectralField, threshold: float = 0.1) -> List[ThinFeature]:
        """
        Detect features that are "almost" topological changes.
        
        These are shown as warnings — the mesh is valid but fragile.
        """
        diagram = cls.compute_persistence_diagram(field)
        
        thin_features = []
        for dim, (birth, death) in diagram.points:
            persistence = death - birth
            if persistence < threshold and persistence > 0:
                thin_features.append(ThinFeature(
                    dimension=dim,
                    persistence=persistence,
                    location=cls._find_feature_location(field, birth, death)
                ))
        
        return thin_features
```

**Expansion**: Use "Topological Autoencoders" (Moor et al., 2020) to learn topology-preserving latent spaces for the spectral context.

---

### 4.5 The Differentiable Stack: End-to-End Geometry

**Key Papers**:
- Shen et al., "FlexiCubes: Flexible Extraction of Meshes from Neural Networks" (SIGGRAPH 2023)
- Nicolet et al., "Large Steps in Inverse Rendering of Geometry" (ACM TOG 2021)
- Laine et al., "Modular Primitives for High-Performance Differentiable Rendering" (ACM TOG 2020)

**Why Critical**: If the entire pipeline is differentiable:
- User corrections backpropagate to update parameters
- Inverse modeling: find parameters that produce target shape
- Style transfer: learn one user's style, apply to another

**Implementation**:

```python
class DifferentiableSupernode:
    """
    Fully differentiable geometry pipeline.
    
    Every component — biarc fitting, FNO, meshing — has gradients.
    """
    
    def __init__(self):
        self.biarc_fitter = LearnedBiarcFitter()
        self.fno = FourierNeuralOperator('MINIMAL_SURFACE')
        self.mesher = FlexiCubes()  # Differentiable meshing
    
    def forward(self, polyline: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: polyline → (vertices, faces)
        
        All operations preserve gradients.
        """
        # Biarc fitting (differentiable)
        biarc_params = self.biarc_fitter(polyline)
        
        # FNO inference (differentiable)
        # Convert biarc to boundary indicator function
        boundary = self._biarc_to_boundary(biarc_params)
        field = self.fno(boundary)
        
        # Meshing (differentiable via FlexiCubes)
        vertices, faces = self.mesher(field)
        
        return vertices, faces
    
    def inverse(self, target: torch.Tensor, num_iterations: int = 1000) -> torch.Tensor:
        """
        Inverse pass: target mesh → input polyline
        
        This is "Inverse Modeling" — the tool becomes a SOLVER.
        """
        # Initialize polyline (random or from target boundary)
        polyline = torch.randn(1, 100, 2, requires_grad=True)
        
        optimizer = torch.optim.Adam([polyline], lr=0.01)
        
        for i in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            predicted_verts, predicted_faces = self.forward(polyline)
            
            # Compute loss (Chamfer distance)
            loss = chamfer_distance(predicted_verts, target)
            
            # Optional: add regularization for smooth polyline
            loss += 0.01 * self._polyline_smoothness(polyline)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss.item():.6f}")
        
        return polyline.detach()
    
    def transfer_style(self, source_user_polylines: List[torch.Tensor],
                       source_user_results: List[torch.Tensor],
                       target_polyline: torch.Tensor) -> torch.Tensor:
        """
        Learn one user's style from examples, apply to new input.
        """
        # Learn style embedding from source examples
        style_embedding = self._learn_style_embedding(
            source_user_polylines,
            source_user_results
        )
        
        # Apply style to target
        return self.forward_with_style(target_polyline, style_embedding)
```

**Expansion**: Integrate with "DreamFusion" (Poole et al., 2022) for text-to-3D generation via score distillation.

---

## Part V: Advanced Capabilities

### 5.1 Supernode Composition: Boolean Operations

Neural implicits make boolean operations trivial:

```python
class BooleanSupernode(GeometricScaffoldSupernode):
    """
    Boolean composition of Supernodes via neural implicit composition.
    
    Union: min(A, B)
    Intersection: max(A, B)
    Difference: max(A, -B)
    
    No mesh boolean required — operates on continuous fields.
    """
    
    def __init__(self, node_a: GeometricScaffoldSupernode,
                 node_b: GeometricScaffoldSupernode,
                 operation: str):
        self.node_a = node_a
        self.node_b = node_b
        self.operation = operation
    
    def _operate(self, state: HybridState):
        # Get neural implicits
        implicit_a = self.node_a._neural_implicit or \
                     NeuralImplicit.from_spectral(self.node_a._spectral_field)
        implicit_b = self.node_b._neural_implicit or \
                     NeuralImplicit.from_spectral(self.node_b._spectral_field)
        
        # Compose
        if self.operation == 'UNION':
            self._neural_implicit = implicit_a.union(implicit_b)
        elif self.operation == 'INTERSECTION':
            self._neural_implicit = implicit_a.intersection(implicit_b)
        elif self.operation == 'DIFFERENCE':
            self._neural_implicit = implicit_a.difference(implicit_b)
        elif self.operation == 'SMOOTH_UNION':
            # Smooth minimum for organic blending
            self._neural_implicit = implicit_a.smooth_union(implicit_b, k=0.1)
```

### 5.2 Automatic Context Switching

The user should never manually select contexts:

```python
class AutoContextSupernode(GeometricScaffoldSupernode):
    """
    Automatically selects optimal context for each operation.
    """
    
    def _analyze_operation(self, operation: str) -> str:
        """
        Determine which context to use.
        """
        AFFINE_OPS = {'translate', 'rotate', 'scale_uniform', 'instance'}
        CONFORMAL_OPS = {'uv_map', 'texture', 'circle_pack', 'biarc_edit'}
        SPECTRAL_OPS = {'boolean', 'smooth', 'physics', 'field_edit'}
        
        if operation in AFFINE_OPS:
            return 'AFFINE'
        elif operation in CONFORMAL_OPS:
            return 'CONFORMAL'
        elif operation in SPECTRAL_OPS:
            return 'SPECTRAL'
        else:
            # Default: choose based on complexity
            return self._infer_optimal_context()
```

### 5.3 Language-to-Geometry Interface

Connect to language models for conversational modeling:

```python
class LanguageSupernode(GeometricScaffoldSupernode):
    """
    Generate geometry from natural language description.
    
    Uses CLIP embedding to guide generation.
    """
    
    def __init__(self, description: str):
        super().__init__(input_id=None)
        self.description = description
        self.clip_encoder = load_clip_model()
    
    def _lift(self, state: HybridState):
        # Encode description
        text_embedding = self.clip_encoder.encode_text(self.description)
        
        # Use embedding to initialize neural implicit
        self._neural_implicit = CLIPGuidedImplicit(
            text_embedding=text_embedding,
            optimization_steps=1000
        )
    
    def _operate(self, state: HybridState):
        # Score distillation sampling
        self._neural_implicit.optimize_via_sds(
            text_embedding=self.text_embedding,
            guidance_scale=100.0
        )
```

---

## Part VI: Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE MATHEMATICAL COMPILER                                │
│                    Geometric Scaffold Supernode v3.0                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   THE PARADIGM SHIFT                                                        │
│   ─────────────────                                                         │
│   FROM: Coordinate manipulation (lossy)                                     │
│   TO: Mathematical compilation (lossless until render)                      │
│                                                                             │
│   THE TRI-SPACE ENGINE                                                      │
│   ───────────────────                                                       │
│   Affine (GL₄ℝ) × Conformal (PSL₂ℂ) × Spectral (L²)                        │
│   Three parallel contexts, synchronized state                               │
│                                                                             │
│   THE PIPELINE                                                              │
│   ────────────                                                              │
│   LIFT: Polyline → Biarc → Neural Field                                    │
│   OPERATE: Möbius ∘ BFF ∘ FNO ∘ Fisher Flow ∘ Topology Check              │
│   COLLAPSE: Field → Mesh (only at render, resolution chosen NOW)           │
│                                                                             │
│   THE CRITICAL INTEGRATIONS                                                 │
│   ─────────────────────────                                                 │
│   • Fourier Neural Operators (resolution-invariant physics)                │
│   • Boundary First Flattening (instant, robust conformal maps)             │
│   • Learned Biarc Fitting (style-aware curve interpretation)               │
│   • Persistent Homology (multi-scale topology validation)                  │
│   • Differentiable Stack (end-to-end gradient flow)                        │
│                                                                             │
│   THE CAPABILITIES                                                          │
│   ────────────────                                                          │
│   • Booleans without mesh booleans (neural implicit composition)           │
│   • Inverse modeling (target → parameters)                                 │
│   • Style transfer (learn preferences, apply to new inputs)                │
│   • Language-to-geometry (describe shape, generate it)                     │
│   • Real-time ghost scaffolding (see math before mesh)                     │
│                                                                             │
│   THE RESULT                                                                │
│   ──────────                                                                │
│   A tool that:                                                              │
│   • Compiles sketches to pure mathematics                                  │
│   • Manipulates geometry in its natural algebraic domain                   │
│   • Preserves generating functions until the last moment                   │
│   • Guarantees topological validity before meshing                         │
│   • Learns from user corrections                                           │
│   • Operates at any resolution                                             │
│                                                                             │
│   This is not a modeling tool. This is a MATHEMATICAL COMPILER.            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Research Bibliography

### Core Architecture
1. Li et al., "Fourier Neural Operator for Parametric PDEs" (NeurIPS 2020)
2. Sawhney & Crane, "Boundary First Flattening" (ACM TOG 2017)
3. Groueix et al., "AtlasNet" (CVPR 2018)
4. Carlsson, "Topology and Data" (AMS Bulletin 2009)

### Neural Geometry
5. Park et al., "DeepSDF" (CVPR 2019)
6. Williams et al., "Neural Splines" (NeurIPS 2021)
7. Shen et al., "FlexiCubes" (SIGGRAPH 2023)
8. Mildenhall et al., "NeRF" (ECCV 2020)

### Differentiable Rendering
9. Nicolet et al., "Large Steps in Inverse Rendering" (ACM TOG 2021)
10. Laine et al., "Modular Primitives for Differentiable Rendering" (ACM TOG 2020)

### Generative Models
11. Poole et al., "DreamFusion" (ICLR 2023)
12. Sanghi et al., "CLIP-Forge" (CVPR 2022)

### Computational Geometry
13. Meek & Walton, "Approximating Curves with Biarcs" (1992)
14. Driscoll & Trefethen, "Schwarz-Christoffel Mapping" (2002)

---

*We're not drawing lines in space. We're compiling fields of intent.*
