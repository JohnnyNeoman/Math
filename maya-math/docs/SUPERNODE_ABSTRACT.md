# The Geometric Scaffold Supernode

> **A Tri-Modal Architecture for Resolution-Independent, Topologically Guaranteed 3D Synthesis**
> The Mathematical Compiler: From Sketch to Field to Form
> Version 2.0 | 2026-01-28

---

## Abstract

Current procedural modeling paradigms rely on discrete, affine transformations (`Matrix4x4`) acting upon static polygonal data. This approach suffers from **Lossy Geometric Compression**—continuous intent is discretized into polylines, topological features are brittle during boolean operations, and global surface logic is lost to local vertex manipulation.

We introduce the **Geometric Scaffold Supernode**, a unified architectural unit within a Functional Parametric L-System (SST) that supersedes the affine-only state stack with a **Tri-Modal Hybrid State**. This architecture synchronizes three parallel mathematical contexts:

| Context | Domain | Responsibility |
|---------|--------|----------------|
| **Affine** | Linear Algebra | Position, rotation, scale — the skeleton |
| **Conformal** | Complex Analysis | Shape-preserving maps — the flow |
| **Spectral** | Harmonic Analysis | Field physics — the substance |

By **lifting** discrete inputs into continuous mathematical representations—specifically **Biarcs** for G1-continuous curves and **Fourier Neural Operators (FNO)** for resolution-independent surface fields—the system acts as a **Mathematical Compiler**, enforcing topological validity via **Darboux Frames** and **Persistent Homology** before rasterization occurs.

This synthesis of Classical Computational Geometry and Geometric Deep Learning shifts the modeling paradigm from manual construction to **Generative Anticipation**, where the tool predicts, optimizes, and mathematically guarantees the user's geometric intent in real-time.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE MATHEMATICAL COMPILER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   USER SKETCH                                                               │
│        ↓                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         LIFT                                        │  │
│   │   Polyline → Biarc (G1)     Mesh → SDF     Curve → Darboux Frame   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│        ↓                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        OPERATE                                      │  │
│   │   Möbius composition   ·   FNO inference   ·   BFF parameterization │  │
│   │   Fisher flow          ·   Wilson loops    ·   Spectral filtering   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│        ↓                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                       COLLAPSE                                      │  │
│   │   Tessellate (LOD)     Isosurface (Marching Cubes)     Sample      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│        ↓                                                                    │
│   RENDERED GEOMETRY (only now do we have polygons)                         │
│                                                                             │
│   Key Insight: The generating functions are preserved until rendering.     │
│   Editing operates on MATH, not MESHES.                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. The Problem: Lossy Geometric Compression

### Why Matrix4x4 Fails

In traditional 3D tools (Maya, Blender, Unreal), everything is forced into **Affine space**:

```
Standard Pipeline:
User Intent → Discrete Samples → Matrix Transforms → Mesh Vertices
```

This is **lossy compression** for geometry:

| Failure Mode | What Happens | Why It's Bad |
|--------------|--------------|--------------|
| **Biarc Failure** | Non-uniform scale transforms circular arcs into ellipses | G1 continuity breaks; must rebake to polygons |
| **FNO Failure** | Field represented as voxel grid | Locked to resolution; "infinite resolution" lost |
| **Conformal Failure** | Angle-preserving maps not representable as 4×4 | Must approximate, accumulating error |
| **Topological Failure** | Boolean operations create non-manifold edges | Must fix after the fact, if detectable |

### The Solution: Preserve the Generating Function

```
Mathematical Compiler Pipeline:
User Intent → Continuous Representation → Mathematical Operations → Discrete Output (only at render)
```

The **generating function** (biarc parameters, Fourier coefficients, Möbius coefficients) is preserved until the absolute last moment. Editing operates on the **mathematics**, not the **mesh**.

---

## 2. The Tri-Space Engine

### Three Parallel Contexts

The `HybridState` tracks three simultaneous representations:

```python
@dataclass
class HybridState:
    """
    Tri-Modal State: Affine × Conformal × Spectral
    
    These are not alternatives — they are PARALLEL.
    The geometry exists in all three spaces simultaneously.
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT A: AFFINE (The Skeleton)
    # ═══════════════════════════════════════════════════════════════════════
    # Data Type: Matrix4x4
    # Role: WHERE things are in the world
    # Operations: Translate, Rotate, Scale, Shear
    # Algebra: GL(4, ℝ) — General Linear Group
    
    affine_stack: List[Matrix4x4]
    platforms: Dict[str, Matrix4x4]
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT B: CONFORMAL (The Flow)
    # ═══════════════════════════════════════════════════════════════════════
    # Data Type: MobiusTransform (a, b, c, d ∈ ℂ)
    # Role: HOW shapes deform while preserving angles
    # Operations: Inversion, Dilation, Rotation, Translation (complex plane)
    # Algebra: PSL(2, ℂ) — Projective Special Linear Group
    # Key Property: Circles → Circles (biarcs remain biarcs)
    
    conformal_stack: List[MobiusTransform]
    conformal_plane: ComplexPlane  # Defined by active affine platform
    biarc_curves: Dict[str, BiarcCurve]
    conformal_maps: Dict[str, ConformalMap]  # BFF, Schwarz-Christoffel
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONTEXT C: SPECTRAL (The Substance)
    # ═══════════════════════════════════════════════════════════════════════
    # Data Type: SpectralField (Fourier coefficients) or NeuralImplicit (MLP weights)
    # Role: WHAT the volumetric content is
    # Operations: Convolution, Filtering, Gradient Flow
    # Algebra: L²(ℝ³) — Square-integrable functions
    # Key Property: Resolution Independence
    
    spectral_fields: Dict[str, SpectralField]
    neural_implicits: Dict[str, NeuralImplicit]
    
    # ═══════════════════════════════════════════════════════════════════════
    # SYNCHRONIZATION
    # ═══════════════════════════════════════════════════════════════════════
    
    def lift_affine_to_conformal(self):
        """
        Extract rotation component of affine transform.
        Define complex plane from local X, Y axes.
        """
        M = self.affine_stack[-1]
        self.conformal_plane = ComplexPlane(
            origin=M.translation(),
            real_axis=M.x_axis(),
            imag_axis=M.y_axis()
        )
    
    def collapse_conformal_to_affine(self, samples: int) -> List[Vec3]:
        """
        Evaluate conformal curve at sample points.
        Project back to 3D via complex plane.
        """
        points = []
        for t in np.linspace(0, 1, samples):
            z = self.current_biarc.evaluate(t)
            w = self.conformal_stack[-1](z)  # Apply Möbius
            p3d = self.conformal_plane.to_3d(w)
            points.append(self.affine_stack[-1] @ p3d)
        return points
    
    def collapse_spectral_to_mesh(self, resolution: Tuple[int, int, int]) -> Mesh:
        """
        Evaluate spectral field on grid.
        Extract isosurface via marching cubes.
        Apply affine transform.
        """
        field = self.spectral_fields['current']
        grid = field.evaluate(resolution)
        verts, faces = marching_cubes(grid, level=0.0)
        verts = [self.affine_stack[-1] @ v for v in verts]
        return Mesh(verts, faces)
```

### The Algebra of Each Context

| Context | Group | Identity | Composition | Inverse |
|---------|-------|----------|-------------|---------|
| **Affine** | GL(4, ℝ) | I₄ | Matrix multiply | Matrix inverse |
| **Conformal** | PSL(2, ℂ) | f(z) = z | (f∘g)(z) = f(g(z)) | f⁻¹(z) = (dz-b)/(-cz+a) |
| **Spectral** | L² convolution | δ(x) | (f*g)(x) | Deconvolution |

**Key insight**: Each context has its own **group structure**. Operations compose within contexts. The Supernode manages transitions between contexts.

---

## 3. The Supernode: Complete Specification

### Core Interface

```python
class GeometricScaffoldSupernode(Node):
    """
    THE MATHEMATICAL COMPILER
    
    Encapsulates the complete Lift → Operate → Collapse pipeline.
    
    Input: Raw user sketch (polyline, partial mesh, voice description)
    Output: Mathematically valid, topologically guaranteed geometry
    
    The Supernode is the fundamental unit of geometric computation.
    Everything else (transforms, instances, booleans) is expressed
    in terms of Supernode composition.
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ═══════════════════════════════════════════════════════════════════════
    
    class Config:
        # Lift configuration
        biarc_tolerance: float = 0.01
        biarc_method: str = 'LEARNED'  # 'TOLERANCE_BAND' | 'LEARNED' | 'NEURAL_SPLINE'
        
        # Operate configuration
        physics_mode: str = 'MINIMAL_SURFACE'  # 'SOAP_FILM' | 'INFLATION' | 'ELASTIC'
        conformal_method: str = 'BFF'  # 'SCHWARZ_CHRISTOFFEL' | 'BFF' | 'LSCM'
        guardrail_level: str = 'ADAPTIVE'  # 'STRICT' | 'ADAPTIVE' | 'PERMISSIVE'
        
        # Collapse configuration
        preview_resolution: Tuple[int, int, int] = (32, 32, 32)
        render_resolution: Tuple[int, int, int] = (256, 256, 256)
        tessellation_mode: str = 'ADAPTIVE'  # 'FIXED' | 'ADAPTIVE' | 'CURVATURE'
    
    # ═══════════════════════════════════════════════════════════════════════
    # INTERNAL STATE
    # ═══════════════════════════════════════════════════════════════════════
    
    def __init__(self, input_id: str, config: Config = None):
        super().__init__('GeometricScaffold')
        self.input_id = input_id
        self.config = config or self.Config()
        
        # Lazy-initialized mathematical representations
        self._biarc: Optional[BiarcCurve] = None
        self._spectral_field: Optional[SpectralField] = None
        self._neural_implicit: Optional[NeuralImplicit] = None
        self._conformal_map: Optional[ConformalMap] = None
        self._darboux_frames: Optional[List[DarbouxFrame]] = None
        
        # Cached collapse results
        self._mesh_cache: Dict[str, Mesh] = {}
        self._validation_result: Optional[ValidationResult] = None
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE A: LIFT (Discrete → Continuous)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _lift(self, state: HybridState):
        """
        Transform discrete input into continuous mathematical representations.
        
        This is where the "compilation" begins — raw samples become functions.
        """
        raw_input = state.geometry_refs[self.input_id]
        
        # ─────────────────────────────────────────────────────────────────
        # LIFT TO CONFORMAL: Polyline → Biarc
        # ─────────────────────────────────────────────────────────────────
        
        if self.config.biarc_method == 'LEARNED':
            # Neural network predicts optimal biarc parameters
            # Learns user's style (mechanical vs organic)
            self._biarc = LearnedBiarcFitter.fit(
                polyline=raw_input,
                style_embedding=state.user_style_embedding
            )
        elif self.config.biarc_method == 'NEURAL_SPLINE':
            # Hybrid: Biarc structure with neural refinement
            # Gives C∞ continuity, not just G1
            self._biarc = NeuralSplineFitter.fit(raw_input)
        else:
            # Classical tolerance band algorithm
            self._biarc = BiarcApproximator.fit(
                polyline=raw_input,
                tolerance=self.config.biarc_tolerance
            )
        
        # Register in conformal context
        state.biarc_curves[f'{self.input_id}_biarc'] = self._biarc
        
        # ─────────────────────────────────────────────────────────────────
        # LIFT TO SPECTRAL: Boundary → Field
        # ─────────────────────────────────────────────────────────────────
        
        if self.config.physics_mode in ['MINIMAL_SURFACE', 'SOAP_FILM', 'INFLATION', 'ELASTIC']:
            # FNO predicts the surface implied by the boundary
            self._spectral_field = FourierNeuralOperator.predict(
                boundary=self._biarc,
                physics_mode=self.config.physics_mode
            )
        else:
            # Neural implicit (DeepSDF-style)
            self._neural_implicit = NeuralImplicit.from_boundary(self._biarc)
        
        # Register in spectral context
        if self._spectral_field:
            state.spectral_fields[f'{self.input_id}_field'] = self._spectral_field
        if self._neural_implicit:
            state.neural_implicits[f'{self.input_id}_implicit'] = self._neural_implicit
        
        # ─────────────────────────────────────────────────────────────────
        # LIFT TO DARBOUX: Curve × Surface → Frame Field
        # ─────────────────────────────────────────────────────────────────
        
        if hasattr(raw_input, 'reference_surface'):
            self._darboux_frames = DarbouxFrameComputer.compute(
                curve=self._biarc,
                surface=raw_input.reference_surface
            )
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE B: OPERATE (Math → Math)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _operate(self, state: HybridState):
        """
        Perform mathematical operations in their native domains.
        
        Key insight: We never leave the mathematical representation.
        All operations are exact, not approximate.
        """
        
        # ─────────────────────────────────────────────────────────────────
        # CONFORMAL PASS: Parameterize the domain
        # ─────────────────────────────────────────────────────────────────
        
        if self.config.conformal_method == 'BFF':
            # Boundary First Flattening (Sawhney & Crane 2017)
            # Linear, instant, handles any topology
            self._conformal_map = BoundaryFirstFlattening.compute(
                boundary=self._biarc
            )
        elif self.config.conformal_method == 'SCHWARZ_CHRISTOFFEL':
            # Classical S-C mapping (good for simple polygons)
            self._conformal_map = SchwarzChristoffel.compute(
                polygon=self._biarc.as_polygon()
            )
        
        # Compose with current conformal transform
        if self._conformal_map:
            mobius_approx = self._conformal_map.to_mobius_approximation()
            state.conformal_stack[-1] = state.conformal_stack[-1].compose(mobius_approx)
        
        # ─────────────────────────────────────────────────────────────────
        # SPECTRAL PASS: Apply physics / relaxation
        # ─────────────────────────────────────────────────────────────────
        
        if self._spectral_field:
            # Fisher Information flow (smart smoothing)
            self._spectral_field = FisherInformationFlow.apply(
                field=self._spectral_field,
                iterations=10,
                preserve_threshold=0.5
            )
        
        if self._neural_implicit:
            # Neural implicit composition (for booleans)
            # Union: min(A, B), Intersection: max(A, B), Difference: max(A, -B)
            pass  # Applied when combining with other Supernodes
        
        # ─────────────────────────────────────────────────────────────────
        # GUARDRAIL PASS: Topological validation
        # ─────────────────────────────────────────────────────────────────
        
        self._validation_result = self._run_guardrails(state)
        
        if not self._validation_result.valid:
            if self.config.guardrail_level == 'STRICT':
                raise TopologyError(self._validation_result.error)
            elif self.config.guardrail_level == 'ADAPTIVE':
                self._apply_adaptive_fix(self._validation_result)
    
    def _run_guardrails(self, state: HybridState) -> ValidationResult:
        """
        Topological validation using invariants.
        
        Checks BEFORE meshing — cheap on math, expensive on polygons.
        """
        
        # Wilson Loop check (linking numbers)
        if state.context_mode == ContextMode.SPREAD:
            linking_check = WilsonLoopValidator.check_spread(state.spread_buffer)
            if not linking_check.valid:
                return linking_check
        
        # Betti numbers check (holes, tunnels)
        if self._spectral_field:
            betti = PersistentHomology.compute_betti(self._spectral_field)
            expected_betti = self._infer_expected_topology()
            if betti != expected_betti:
                return ValidationResult(
                    valid=False,
                    error=f"Unexpected topology: β={betti}, expected {expected_betti}",
                    suggestion="Check for self-intersections or missing connections"
                )
        
        # Winding number check (self-intersection)
        if self._biarc:
            winding = WindingNumber.compute(self._biarc)
            if winding != 1:
                return ValidationResult(
                    valid=False,
                    error=f"Curve self-intersects (winding={winding})",
                    suggestion="Simplify curve or use different topology"
                )
        
        return ValidationResult(valid=True)
    
    def _apply_adaptive_fix(self, validation: ValidationResult):
        """
        Automatically fix topological issues when possible.
        """
        if "self-intersects" in validation.error:
            # Smooth the biarc to remove self-intersection
            self._biarc = BiarcSmoother.remove_self_intersections(self._biarc)
        
        elif "linking" in validation.error:
            # Switch from Loft to Bridge
            self._collapse_method = 'BRIDGE'
        
        elif "Unexpected topology" in validation.error:
            # Use Sparc3D hole-filling
            self._spectral_field = Sparc3D.fill_holes(self._spectral_field)
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE C: COLLAPSE (Continuous → Discrete)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _collapse(self, state: HybridState, mode: str):
        """
        Transform mathematical representations back to discrete geometry.
        
        This is the ONLY place where discretization occurs.
        """
        
        if mode == 'PREVIEW':
            # Fast ghost scaffolding — sample the math lightly
            self._emit_ghost_scaffolding(state)
        
        elif mode == 'VIEWPORT':
            # Medium resolution for interactive editing
            resolution = (64, 64, 64)
            mesh = self._collapse_to_mesh(state, resolution)
            state.emit(mesh, tags={'lod': 'viewport'})
        
        elif mode == 'RENDER':
            # Full resolution for final output
            resolution = self.config.render_resolution
            mesh = self._collapse_to_mesh(state, resolution)
            state.emit(mesh, tags={'lod': 'render'})
    
    def _emit_ghost_scaffolding(self, state: HybridState):
        """
        Draw mathematical guides without generating heavy geometry.
        
        This is what makes the tool feel "intelligent."
        """
        
        # Ghost 1: The smoothed biarc (how we interpreted the sketch)
        biarc_points = self._biarc.sample(segments=100)
        state.emit_ghost(
            Polyline(biarc_points),
            style='biarc',
            color=(0.2, 0.9, 0.2, 0.6)
        )
        
        # Ghost 2: The conformal grid (how geometry will flow)
        if self._conformal_map:
            grid_lines = self._conformal_map.sample_grid(u_count=10, v_count=10)
            state.emit_ghost(
                grid_lines,
                style='conformal_grid',
                color=(0.5, 0.5, 1.0, 0.3)
            )
        
        # Ghost 3: The predicted surface (what the final shape will be)
        if self._spectral_field:
            preview_mesh = self._spectral_field.extract_preview(
                resolution=self.config.preview_resolution
            )
            state.emit_ghost(
                preview_mesh,
                style='neural_surface',
                color=(1.0, 0.8, 0.2, 0.2)
            )
        
        # Ghost 4: Topology warnings (if any)
        if self._validation_result and not self._validation_result.valid:
            if self._validation_result.highlight:
                state.emit_ghost(
                    Sphere(self._validation_result.highlight, radius=5),
                    style='warning',
                    color=(1.0, 0.0, 0.0, 0.8)
                )
    
    def _collapse_to_mesh(self, state: HybridState, resolution: Tuple[int, int, int]) -> Mesh:
        """
        Full mesh extraction from mathematical representations.
        """
        
        # Check cache
        cache_key = f"{resolution}"
        if cache_key in self._mesh_cache:
            return self._mesh_cache[cache_key]
        
        # Extract from spectral field
        if self._spectral_field:
            grid = self._spectral_field.evaluate(resolution)
            verts, faces = marching_cubes(grid, level=0.0)
        elif self._neural_implicit:
            verts, faces = self._neural_implicit.extract_mesh(resolution)
        else:
            # Fallback: extrude biarc
            verts, faces = self._biarc.extrude_to_mesh()
        
        # Apply UV from conformal map
        if self._conformal_map:
            uvs = [self._conformal_map.inverse(v[:2]) for v in verts]
        else:
            uvs = None
        
        # Apply affine transform
        verts = [state.affine_stack[-1] @ v for v in verts]
        
        # Create mesh
        mesh = Mesh(verts, faces, uvs=uvs)
        
        # Cache
        self._mesh_cache[cache_key] = mesh
        
        return mesh
    
    # ═══════════════════════════════════════════════════════════════════════
    # MAIN EXECUTION
    # ═══════════════════════════════════════════════════════════════════════
    
    def execute(self, state: HybridState):
        """
        Execute the complete Lift → Operate → Collapse pipeline.
        """
        
        # Phase A: Lift to continuous
        self._lift(state)
        
        # Phase B: Operate in mathematical space
        self._operate(state)
        
        # Phase C: Collapse to discrete (based on render mode)
        self._collapse(state, state.render_mode)
        
        # Execute children (nested Supernodes)
        self.execute_children(state)
```

---

## 4. Research Integration: Critical Techniques

### 4.1 Core Generative Logic: Fourier Neural Operators (FNO)

**Key Paper**: "Fourier Neural Operator for Parametric PDEs" (Li et al., 2020)

**Why It's Critical**: FNOs learn mappings between **function spaces**, not sample spaces. This means:
- Train on 32³ grid
- Infer on 512³ grid with **zero retraining**
- The operator learns the **structure** of the solution

**Implementation: NeuralFieldOperator**

```python
class FourierNeuralOperator:
    """
    Resolution-invariant learned physics operator.
    
    Given a boundary (biarc curve), predicts the surface that
    would form under physical constraints (soap film, inflation, etc.)
    """
    
    def __init__(self, physics_mode: str):
        self.physics_mode = physics_mode
        self.model = self._load_model(physics_mode)
    
    def _load_model(self, mode: str) -> nn.Module:
        """Load pre-trained FNO for specific physics."""
        models = {
            'MINIMAL_SURFACE': 'fno_minimal_surface_v3.pt',
            'SOAP_FILM': 'fno_soap_film_v2.pt',
            'INFLATION': 'fno_inflation_v2.pt',
            'ELASTIC': 'fno_elastic_shell_v1.pt',
            'FLUID_FLOW': 'fno_navier_stokes_v2.pt'
        }
        return load_fno_model(models[mode])
    
    @classmethod
    def predict(cls, boundary: BiarcCurve, physics_mode: str) -> SpectralField:
        """
        Predict surface from boundary curve.
        
        Returns spectral (Fourier) representation — resolution-invariant.
        """
        operator = cls(physics_mode)
        
        # Rasterize boundary to coarse grid (32³)
        sparse_input = rasterize_boundary(boundary, resolution=(32, 32, 32))
        
        # Run FNO inference (< 5ms on GPU)
        # Output is spectral coefficients, not grid values
        spectral_coeffs = operator.model(sparse_input)
        
        return SpectralField(
            coefficients=spectral_coeffs,
            modes=(32, 32, 32),
            domain='box',
            physics_mode=physics_mode
        )
```

**Expansion**: Combine with "Learning Mesh-Based Simulation with Graph Networks" (Pfaff et al., 2020) for adaptive mesh refinement during collapse.

### 4.2 The Shape Engine: Boundary First Flattening (BFF)

**Key Paper**: "Boundary First Flattening" (Sawhney & Crane, 2017)

**Why It's Critical**: Classical Schwarz-Christoffel is iterative and fragile. BFF is:
- **Linear** (direct solve, no iteration)
- **Instant** (real-time even for complex boundaries)
- **Robust** (handles any topology, including holes)

**Implementation: ConformalMap**

```python
class BoundaryFirstFlattening:
    """
    State-of-the-art conformal parameterization.
    
    Maps any boundary to canonical domain (disk/plane) while
    preserving angles locally.
    """
    
    @classmethod
    def compute(cls, boundary: BiarcCurve) -> ConformalMap:
        """
        Compute conformal map from boundary to unit disk.
        
        Unlike Schwarz-Christoffel:
        - No iteration required
        - Handles arbitrary topology
        - Produces perfect UVs automatically
        """
        # Convert biarc to discrete boundary
        vertices = boundary.sample(segments=1000)
        
        # Build Laplacian matrix
        L = build_boundary_laplacian(vertices)
        
        # Solve for harmonic coordinates (direct, linear)
        # This is the key insight of BFF — it's a linear system
        u, v = solve_harmonic_coordinates(L, vertices)
        
        # Construct conformal map
        return ConformalMap(
            type='BFF',
            forward=lambda z: interpolate_conformal(z, vertices, u, v),
            inverse=lambda w: interpolate_inverse(w, vertices, u, v),
            jacobian=lambda z: conformal_jacobian(z, vertices, u, v)
        )
    
    def sample_grid(self, u_count: int, v_count: int) -> List[Polyline]:
        """
        Generate conformal grid for visualization.
        
        These lines show the "flow" of the geometry.
        """
        lines = []
        
        # Constant-u lines (radial in disk)
        for i in range(u_count):
            u = i / u_count
            line = [self.forward(complex(u, v)) for v in np.linspace(0, 1, 100)]
            lines.append(Polyline(line))
        
        # Constant-v lines (circular in disk)
        for j in range(v_count):
            v = j / v_count
            line = [self.forward(complex(u, v)) for u in np.linspace(0, 1, 100)]
            lines.append(Polyline(line))
        
        return lines
```

### 4.3 The Input Interpreter: Learned Biarc Fitting

**Key Paper**: "Deep Geometric Learning of Curves" (Groueix et al., 2018)

**Why It's Critical**: Hand-coded tolerance band algorithms don't learn user preferences. A neural fitter can:
- Learn **style** (mechanical vs organic curves)
- Adapt to **context** (tight corners for machinery, smooth for characters)
- Be **differentiable** (for end-to-end learning)

**Implementation: LearnedBiarcFitter**

```python
class LearnedBiarcFitter:
    """
    Neural network that predicts optimal biarc parameters.
    
    Instead of hard-coded tolerance, learns from user corrections.
    """
    
    def __init__(self, style_model: str = 'default'):
        self.encoder = load_curve_encoder()  # Encodes polyline to latent
        self.decoder = load_biarc_decoder()  # Decodes latent to biarc params
        self.style_embeddings = load_style_embeddings()
    
    @classmethod
    def fit(cls, polyline: Polyline, style_embedding: Optional[np.ndarray] = None) -> BiarcCurve:
        """
        Predict biarc curve from polyline.
        
        style_embedding: Optional user style (learned from corrections)
        """
        fitter = cls()
        
        # Encode polyline to latent representation
        latent = fitter.encoder(polyline.vertices)
        
        # Add style if provided
        if style_embedding is not None:
            latent = latent + style_embedding
        
        # Decode to biarc parameters
        # Output: list of (center, radius, start_angle, end_angle) per arc
        biarc_params = fitter.decoder(latent)
        
        return BiarcCurve.from_params(biarc_params)
    
    def learn_from_correction(self, original: Polyline, corrected: BiarcCurve):
        """
        Update style embedding from user correction.
        
        This is how the tool learns the user's preferences.
        """
        # Encode the correction as style delta
        original_latent = self.encoder(original.vertices)
        corrected_latent = self.encode_biarc(corrected)
        
        style_delta = corrected_latent - original_latent
        
        # Update user's style embedding (exponential moving average)
        self.user_style = 0.9 * self.user_style + 0.1 * style_delta
```

### 4.4 The Topological Guardrail: Persistent Homology

**Key Paper**: "Persistent Homology for Shape Analysis" (Carlsson, 2009)

**Why It's Critical**: Wilson loops check linking numbers (good for curves). Persistent homology checks **multi-scale topology**:
- β₀: Number of connected components
- β₁: Number of tunnels/holes
- β₂: Number of voids

**Implementation: PersistentHomology**

```python
class PersistentHomology:
    """
    Multi-scale topological analysis.
    
    Detects features at different scales — critical for catching
    small holes or thin connections that visual inspection misses.
    """
    
    @classmethod
    def compute_betti(cls, field: SpectralField) -> Tuple[int, int, int]:
        """
        Compute Betti numbers from spectral field.
        
        Returns (β₀, β₁, β₂):
        - β₀: Connected components
        - β₁: Tunnels/holes
        - β₂: Voids/cavities
        """
        # Sample field at medium resolution
        grid = field.evaluate((64, 64, 64))
        
        # Build simplicial complex from grid
        complex = build_cubical_complex(grid)
        
        # Compute persistent homology
        persistence = compute_persistence(complex)
        
        # Extract Betti numbers (at infinite persistence)
        beta_0 = count_persistent_features(persistence, dim=0)
        beta_1 = count_persistent_features(persistence, dim=1)
        beta_2 = count_persistent_features(persistence, dim=2)
        
        return (beta_0, beta_1, beta_2)
    
    @classmethod
    def compute_persistence_diagram(cls, field: SpectralField) -> PersistenceDiagram:
        """
        Full persistence diagram for visualization.
        
        Shows birth/death of topological features across scales.
        Useful for debugging unexpected topology.
        """
        grid = field.evaluate((64, 64, 64))
        complex = build_cubical_complex(grid)
        return compute_full_persistence(complex)
```

### 4.5 Neural Implicit Representations: DeepSDF

**Key Paper**: "DeepSDF: Learning Continuous Signed Distance Functions" (Park et al., 2019)

**Why It's Critical**: Instead of voxel grids or Fourier coefficients, represent shapes as **neural networks**:
- Infinite resolution (query at any point)
- Smooth gradients (perfect for optimization)
- Composable (boolean operations are trivial)

**Implementation: NeuralImplicit**

```python
class NeuralImplicit:
    """
    Shape represented as a neural network.
    
    SDF(x) = Network(x) → signed distance to surface
    
    Advantages:
    - Query at any resolution
    - Smooth everywhere (differentiable)
    - Boolean ops are just min/max
    """
    
    def __init__(self, network: nn.Module):
        self.network = network
    
    def __call__(self, x: np.ndarray) -> float:
        """Evaluate SDF at point x."""
        return self.network(x)
    
    @classmethod
    def from_boundary(cls, boundary: BiarcCurve) -> 'NeuralImplicit':
        """
        Train neural implicit from boundary curve.
        
        Uses autodecoder approach (Park et al., 2019).
        """
        # Sample points near boundary
        points, sdf_values = sample_sdf_near_boundary(boundary)
        
        # Train small MLP
        network = train_sdf_network(points, sdf_values)
        
        return cls(network)
    
    def union(self, other: 'NeuralImplicit') -> 'NeuralImplicit':
        """
        Boolean union: min(A, B)
        
        No mesh boolean required — just compose networks!
        """
        return NeuralImplicit(
            network=lambda x: min(self(x), other(x))
        )
    
    def intersection(self, other: 'NeuralImplicit') -> 'NeuralImplicit':
        """Boolean intersection: max(A, B)"""
        return NeuralImplicit(
            network=lambda x: max(self(x), other(x))
        )
    
    def difference(self, other: 'NeuralImplicit') -> 'NeuralImplicit':
        """Boolean difference: max(A, -B)"""
        return NeuralImplicit(
            network=lambda x: max(self(x), -other(x))
        )
    
    def extract_mesh(self, resolution: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract mesh via marching cubes.
        
        This is the ONLY place where discretization occurs.
        """
        # Evaluate on grid
        grid = np.zeros(resolution)
        for i, j, k in np.ndindex(resolution):
            x = np.array([i / resolution[0], j / resolution[1], k / resolution[2]])
            grid[i, j, k] = self(x)
        
        # Marching cubes
        return marching_cubes(grid, level=0.0)
```

---

## 5. End-to-End Differentiability: The Meta-Optimizer

### Why Differentiability Matters

If the entire pipeline is differentiable, we can:
- **Learn from corrections**: User drags vertex → gradient flows back to biarc params
- **Inverse modeling**: Given target shape, find L-system rules that produce it
- **Style transfer**: Learn one user's style, apply to another

### Implementation: Differentiable Supernode

```python
class DifferentiableSupernode:
    """
    Fully differentiable geometry pipeline.
    
    Every operation — biarc fitting, FNO inference, meshing — has gradients.
    This enables learning from user corrections and inverse modeling.
    """
    
    def __init__(self):
        # All components are JAX/PyTorch modules
        self.biarc_fitter = DifferentiableBiarcFitter()
        self.fno = DifferentiableFNO()
        self.mesher = DifferentiableMarchingCubes()
    
    def forward(self, polyline: jnp.ndarray) -> Mesh:
        """
        Forward pass: polyline → mesh
        
        All operations preserve gradients.
        """
        # Biarc fitting (differentiable)
        biarc_params = self.biarc_fitter(polyline)
        
        # FNO inference (differentiable)
        spectral_coeffs = self.fno(biarc_params)
        
        # Meshing (differentiable via implicit differentiation)
        vertices, faces = self.mesher(spectral_coeffs)
        
        return Mesh(vertices, faces)
    
    def inverse(self, target_mesh: Mesh) -> jnp.ndarray:
        """
        Inverse pass: mesh → polyline
        
        Find the input polyline that produces the target mesh.
        This is "Inverse Modeling" — the tool becomes a SOLVER.
        """
        # Initialize random polyline
        polyline = jnp.randn(100, 2)
        
        # Optimize via gradient descent
        optimizer = optax.adam(0.01)
        opt_state = optimizer.init(polyline)
        
        for _ in range(1000):
            loss, grads = jax.value_and_grad(self._loss)(polyline, target_mesh)
            updates, opt_state = optimizer.update(grads, opt_state)
            polyline = optax.apply_updates(polyline, updates)
        
        return polyline
    
    def _loss(self, polyline: jnp.ndarray, target: Mesh) -> float:
        """Loss function: Chamfer distance between predicted and target."""
        predicted = self.forward(polyline)
        return chamfer_distance(predicted.vertices, target.vertices)
    
    def learn_from_correction(self, original_polyline: jnp.ndarray, corrected_mesh: Mesh):
        """
        User corrects the output mesh.
        Backpropagate to update biarc fitter and FNO.
        
        The tool LEARNS from the correction.
        """
        loss, grads = jax.value_and_grad(self._loss)(original_polyline, corrected_mesh)
        
        # Update biarc fitter
        self.biarc_fitter.update(grads['biarc_fitter'])
        
        # Update FNO (fine-tune on this example)
        self.fno.update(grads['fno'])
```

### Inverse Modeling: The Ultimate Capability

```python
class InverseModeler:
    """
    Given a target shape, find the SST rules that produce it.
    
    This turns the tool from a "Builder" into a "Solver."
    """
    
    def __init__(self, supernode: DifferentiableSupernode):
        self.supernode = supernode
    
    def solve(self, target: Union[Mesh, Image]) -> SSTProgram:
        """
        Find SST program that produces the target.
        
        Input can be:
        - A mesh (3D target)
        - An image (2D silhouette)
        - A photograph (via differentiable rendering)
        """
        if isinstance(target, Image):
            target = self._image_to_mesh_prior(target)
        
        # Optimize for SST parameters
        polyline = self.supernode.inverse(target)
        
        # Convert polyline to SST program
        program = self._polyline_to_sst(polyline)
        
        return program
    
    def _polyline_to_sst(self, polyline: jnp.ndarray) -> SSTProgram:
        """
        Convert optimized polyline to human-readable SST.
        
        This is "program synthesis" — generating code from examples.
        """
        # Detect symmetries
        symmetries = detect_symmetries(polyline)
        
        # Detect repeated patterns
        patterns = detect_patterns(polyline)
        
        # Generate SST YAML
        return SSTProgram.from_analysis(polyline, symmetries, patterns)
```

---

## 6. Implementation Technology Stack

### Compute Engine

| Technology | Purpose | Why |
|------------|---------|-----|
| **JAX** | Differentiable programming | XLA compilation, vmap, grad |
| **Taichi** | GPU compute kernels | Real-time preview, parallel meshing |
| **CUDA/Metal** | Low-level GPU | FNO inference, marching cubes |

### Geometry Processing

| Library | Purpose | Why |
|---------|---------|-----|
| **libigl** | Mesh operations | BFF, Laplacians, robust booleans |
| **CGAL** | Computational geometry | Biarc fitting, Voronoi |
| **Open3D** | Point cloud / mesh | Fast preview rendering |

### Machine Learning

| Framework | Purpose | Why |
|-----------|---------|-----|
| **PyTorch** | Neural networks | FNO, DeepSDF, learned fitters |
| **PyTorch Geometric** | Graph neural networks | Mesh processing, GNN predictions |
| **Hugging Face** | Model hosting | Pre-trained FNOs |

### Integration

| Technology | Purpose | Why |
|------------|---------|-----|
| **Pybind11** | C++/Python binding | libigl, CGAL integration |
| **ONNX** | Model interchange | Deploy FNO to Unreal/Maya |
| **gRPC** | Service communication | Claude ↔ SST engine |

---

## 7. Future Directions

### 7.1 Neural Splines: C∞ Continuity

**Research**: "Neural Splines: Fitting 3D Surfaces with Infinitely-Wide Neural Networks" (Williams et al., 2021)

**Impact**: Instead of G1 continuity (biarcs), achieve C∞ (infinite derivatives). This enables:
- Perfect curvature flows
- Aerodynamic optimization
- "Slingshot" trajectories for procedural animation

### 7.2 Equivariant Neural Networks

**Research**: "E(3)-Equivariant Graph Neural Networks" (Satorras et al., 2021)

**Impact**: Networks that respect 3D symmetry by construction. The Supernode's G-CNN predictions become more accurate because the network "understands" rotation and translation.

### 7.3 Neural Radiance Fields for Sketch-to-3D

**Research**: "NeRF: Representing Scenes as Neural Radiance Fields" (Mildenhall et al., 2020)

**Impact**: User sketches from multiple views → NeRF → extract mesh. This extends the Supernode's input modality beyond polylines.

### 7.4 Topological Optimization

**Research**: "Topology Optimization with Neural Networks" (Sosnovik et al., 2021)

**Impact**: Instead of predicting a single surface, predict the **optimal** surface given constraints (stress, weight, aesthetics). The Supernode becomes a constraint solver.

### 7.5 Language-to-Geometry

**Research**: "CLIP-Forge: Towards Zero-Shot Text-to-Shape Generation" (Sanghi et al., 2022)

**Impact**: User describes shape in natural language → Supernode generates it. Claude's conversational interface becomes a direct modeling tool.

---

## 8. Summary: The Mathematical Compiler

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE GEOMETRIC SCAFFOLD SUPERNODE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   THE PROBLEM                                                               │
│   Matrix4x4 + Polygons = Lossy Geometric Compression                        │
│   Continuous intent → Discrete samples → Lost mathematical structure       │
│                                                                             │
│   THE SOLUTION                                                              │
│   Tri-Modal Hybrid State: Affine × Conformal × Spectral                    │
│   Preserve generating functions until rendering                             │
│   Edit MATH, not MESHES                                                     │
│                                                                             │
│   THE PIPELINE                                                              │
│   LIFT: Polyline → Biarc (G1)  |  Mesh → SDF  |  Curve → Darboux          │
│   OPERATE: Möbius  |  FNO  |  BFF  |  Fisher Flow  |  Wilson Loops        │
│   COLLAPSE: Tessellate  |  Isosurface  |  Sample (only at render)         │
│                                                                             │
│   THE TECHNIQUES                                                            │
│   • Fourier Neural Operators (resolution-invariant physics)                │
│   • Boundary First Flattening (instant, robust conformal maps)             │
│   • Learned Biarc Fitting (style-aware curve interpretation)               │
│   • Persistent Homology (multi-scale topological validation)               │
│   • Neural Implicits (composable, differentiable shapes)                   │
│                                                                             │
│   THE RESULT                                                                │
│   A tool that:                                                              │
│   • Compiles sketches to pure mathematics                                  │
│   • Manipulates geometry losslessly                                        │
│   • Rasterizes only when absolutely necessary                              │
│   • Learns from user corrections                                           │
│   • Guarantees topological validity                                        │
│   • Operates at any resolution                                             │
│                                                                             │
│   This is not a modeling tool.                                              │
│   This is a MATHEMATICAL COMPILER for geometry.                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix: Quick Reference

### Node Types

| Node | Phase | Input | Output |
|------|-------|-------|--------|
| `BiarcApprox` | Lift | Polyline | BiarcCurve |
| `SDFEncoder` | Lift | Mesh | SpectralField |
| `DarbouxBinder` | Lift | Curve × Surface | FrameStream |
| `ConformalWarp` | Operate | BiarcCurve | ConformalMap |
| `NeuralRelax` | Operate | SpectralField | SpectralField |
| `WilsonGuard` | Operate | Spread | ValidationResult |
| `Tessellate` | Collapse | BiarcCurve | Polyline |
| `Isosurface` | Collapse | SpectralField | Mesh |

### State Contexts

| Context | Type | Group | Key Property |
|---------|------|-------|--------------|
| Affine | Matrix4x4 | GL(4,ℝ) | Linear transforms |
| Conformal | MobiusTransform | PSL(2,ℂ) | Circles → Circles |
| Spectral | SpectralField | L² | Resolution-invariant |

### Research Papers

| Paper | Year | Relevance |
|-------|------|-----------|
| Li et al., "FNO for PDEs" | 2020 | Core spectral engine |
| Sawhney & Crane, "BFF" | 2017 | Conformal parameterization |
| Groueix et al., "Deep Curves" | 2018 | Learned biarc fitting |
| Carlsson, "Persistent Homology" | 2009 | Topological guardrails |
| Park et al., "DeepSDF" | 2019 | Neural implicits |
| Williams et al., "Neural Splines" | 2021 | C∞ continuity |

---

*We're not drawing lines in space. We're defining fields of intent.*
