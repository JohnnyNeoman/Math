# Conformal Topology Modeler

> **BFF-Powered ZModeler: Where Conformal Charts Drive Topology Operations**
> Boundary-First Flattening as the Intelligence Behind Interactive Retopology
> Version 1.0 | 2026-02-04

---

## Vision

A **ZModeler-style polygon modeling tool** where:

> **The conformal chart knows where edge loops should go.**
> **The Yamabe distortion shows where cuts are needed.**
> **The Poincaré-Steklov operators solve for optimal topology.**
> **You confirm what the math suggests.**

Instead of manually inserting edge loops by clicking, the tool **anticipates topology** from the conformal structure and presents ghost scaffolding of ideal quad flow.

---

## 1. The Core Insight: Conformal Isolines ARE Edge Loops

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE FUNDAMENTAL CONNECTION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CONFORMAL CHART                         POLYGON TOPOLOGY                  │
│   ───────────────                         ────────────────                  │
│                                                                             │
│   ┌─────────────────┐                     ┌─────────────────┐              │
│   │  │  │  │  │  │  │                     │  ═══════════════│              │
│   │──┼──┼──┼──┼──┼──│                     │  ║  ║  ║  ║  ║  ║              │
│   │  │  │  │  │  │  │        ═══►         │  ═══════════════│              │
│   │──┼──┼──┼──┼──┼──│                     │  ║  ║  ║  ║  ║  ║              │
│   │  │  │  │  │  │  │                     │  ═══════════════│              │
│   └─────────────────┘                     └─────────────────┘              │
│                                                                             │
│   Conformal isolines (u, v)               Quad edge loops                  │
│   are ORTHOGONAL by definition            with LOW SHEAR                   │
│                                                                             │
│   The chart ALREADY ENCODES               We just need to                  │
│   the ideal edge flow!                    EXTRACT and COMMIT it            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why This Matters for ZModeler-Style Tools

| Traditional ZModeler | Conformal Topology Modeler |
|----------------------|----------------------------|
| User clicks to insert edge loop | Chart shows where loops belong |
| User decides loop spacing | Yamabe distortion optimizes spacing |
| User manually bridges edges | PS operators solve bridging constraints |
| Topology is trial-and-error | Topology is mathematically guided |

---

## 2. The Yamabe Equation as Distortion Map

From the research (and the cow visualization):

```
Yamabe Equation:  Δu = -K   on Ω
                   u = 0    on ∂Ω (cut γ)

where:
  u = conformal distortion (log scale factor)
  K = Gaussian curvature
  γ = cut (seam) where distortion is zero
```

### What This Tells Us

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DISTORTION AS TOPOLOGY GUIDANCE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   HIGH DISTORTION (red/blue extremes)     LOW DISTORTION (white)           │
│   ─────────────────────────────────────   ──────────────────────           │
│                                                                             │
│   • Geometry is being stretched/compressed • Chart fits naturally          │
│   • NEED more edge loops here              • Topology is adequate          │
│   • Consider adding a CUT (seam)           • Leave as-is                   │
│   • High curvature regions                 • Flat/gentle regions           │
│                                                                             │
│   ACTION: Insert loops, add cuts           ACTION: Maintain current flow   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### UI Visualization

```python
class DistortionOverlay:
    """
    Real-time distortion heatmap on the mesh surface.
    """
    
    def compute_distortion(self, mesh: Mesh, chart: BFFChart) -> np.ndarray:
        """
        Compute per-vertex distortion from conformal map.
        
        u = log(scale_factor) where scale_factor = |dz/dw|
        """
        # Get conformal coordinates
        uv = chart.uv_coordinates
        
        # Compute Jacobian per face
        for face in mesh.faces:
            # 3D edge vectors
            e1_3d = mesh.vertices[face[1]] - mesh.vertices[face[0]]
            e2_3d = mesh.vertices[face[2]] - mesh.vertices[face[0]]
            
            # 2D edge vectors (in chart)
            e1_2d = uv[face[1]] - uv[face[0]]
            e2_2d = uv[face[2]] - uv[face[0]]
            
            # Scale factor
            area_3d = np.linalg.norm(np.cross(e1_3d, e2_3d)) / 2
            area_2d = np.abs(e1_2d[0] * e2_2d[1] - e1_2d[1] * e2_2d[0]) / 2
            
            scale = np.sqrt(area_2d / area_3d)
            distortion = np.log(scale)  # u in Yamabe equation
        
        return per_vertex_distortion
    
    def suggest_cuts(self, distortion: np.ndarray, threshold: float = 0.5) -> List[Edge]:
        """
        Suggest cut locations where distortion exceeds threshold.
        
        These become seam candidates (γ in Yamabe: u=0 along cut).
        """
        high_distortion_vertices = np.where(np.abs(distortion) > threshold)[0]
        
        # Find shortest path connecting high-distortion regions
        # This path becomes a suggested cut
        cut_path = self.find_geodesic_through(high_distortion_vertices)
        
        return cut_path
```

---

## 3. Poincaré-Steklov Operators as Topology Solvers

The PS operators (Dirichlet↔Neumann maps) tell us:

> **Given boundary conditions, what happens in the interior?**

For topology modeling, this means:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PS OPERATORS FOR TOPOLOGY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   DIRICHLET → NEUMANN (Λ)                                                  │
│   ─────────────────────────                                                │
│   "Given boundary POSITIONS, what FLUX flows through?"                     │
│                                                                             │
│   Application: You specify edge loop POSITIONS                             │
│                System tells you how much STRETCH occurs                    │
│                                                                             │
│   NEUMANN → DIRICHLET (Λ†)                                                 │
│   ─────────────────────────                                                │
│   "Given boundary FLUX, what POSITIONS result?"                            │
│                                                                             │
│   Application: You specify UNIFORM edge density                            │
│                System tells you where LOOPS should go                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Topology Solver

```python
class TopologySolver:
    """
    Use PS operators to solve for optimal edge loop placement.
    """
    
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.laplacian = build_cotan_laplacian(mesh)
        self.cholesky = cholesky_factor(self.laplacian)  # Factor once, reuse
    
    def dirichlet_to_neumann(self, boundary_values: np.ndarray) -> np.ndarray:
        """
        Λ operator: Given boundary positions, compute boundary flux.
        
        This tells us: if we place edge loops HERE, how much
        distortion flows through the boundary?
        """
        # Solve Laplace equation with Dirichlet BC
        interior_values = self.solve_dirichlet(boundary_values)
        
        # Compute normal derivative at boundary
        boundary_flux = self.compute_normal_derivative(interior_values)
        
        return boundary_flux
    
    def neumann_to_dirichlet(self, boundary_flux: np.ndarray) -> np.ndarray:
        """
        Λ† operator: Given boundary flux, compute boundary positions.
        
        This tells us: if we want UNIFORM edge density (constant flux),
        where should the loops be placed?
        """
        # Solve Laplace equation with Neumann BC
        interior_values = self.solve_neumann(boundary_flux)
        
        # Extract boundary values
        boundary_positions = interior_values[self.boundary_indices]
        
        return boundary_positions
    
    def solve_for_uniform_loops(self, n_loops: int, region: ChartRegion) -> List[EdgeLoop]:
        """
        Given desired number of loops, solve for their optimal positions.
        
        Key insight: Uniform flux (constant ∂u/∂n) → evenly-spaced isolines
        """
        # Specify uniform flux on boundary
        uniform_flux = np.ones(len(region.boundary)) / len(region.boundary)
        
        # Solve for positions
        isoline_positions = self.neumann_to_dirichlet(uniform_flux)
        
        # Extract isolines at regular intervals
        loops = []
        for i in range(n_loops):
            t = (i + 1) / (n_loops + 1)
            loop = region.extract_isoline(t)
            loops.append(loop)
        
        return loops
```

---

## 4. ZModeler Operations via Conformal Charts

### Operation 1: Insert Edge Loop (Conformal-Guided)

**Traditional**: Click on an edge, loop inserted perpendicular to selection.

**Conformal**: System shows ghost isolines, user clicks to confirm which ones to commit.

```python
class ConformalEdgeLoopInserter:
    """
    Insert edge loops guided by conformal chart structure.
    """
    
    def __init__(self, mesh: Mesh, chart: BFFChart):
        self.mesh = mesh
        self.chart = chart
        self.ghost_loops = []
    
    def show_ghost_loops(self, n_suggestions: int = 5):
        """
        Display ghost edge loops at optimal conformal isoline positions.
        """
        # Get u-isolines (one direction)
        u_isolines = self.chart.extract_u_isolines(n_suggestions)
        
        # Get v-isolines (orthogonal direction)
        v_isolines = self.chart.extract_v_isolines(n_suggestions)
        
        # Weight by distortion - suggest more loops where distortion is high
        distortion = self.compute_distortion()
        
        for loop in u_isolines + v_isolines:
            avg_distortion = np.mean([distortion[v] for v in loop.vertices])
            loop.priority = avg_distortion  # Higher distortion = more needed
        
        # Sort by priority
        all_loops = sorted(u_isolines + v_isolines, key=lambda l: -l.priority)
        
        # Display as ghosts
        self.ghost_loops = all_loops[:n_suggestions * 2]
        self.display_ghosts()
    
    def on_click(self, position: Vec3):
        """
        User clicks near a ghost loop → commit it.
        """
        # Find nearest ghost loop
        nearest = self.find_nearest_ghost(position)
        
        if nearest is not None and nearest.distance < self.snap_threshold:
            # Commit the loop
            self.commit_loop(nearest.loop)
            
            # Update chart (BFF recomputes with new boundary)
            self.chart.update_with_cut(nearest.loop)
            
            # Refresh ghosts
            self.show_ghost_loops()
    
    def commit_loop(self, loop: EdgeLoop):
        """
        Insert the edge loop into the mesh.
        """
        # Standard edge loop insertion
        new_vertices = []
        new_edges = []
        
        for edge in loop.edges:
            # Split edge at isoline crossing point
            t = loop.parameter_at(edge)
            new_vert = self.mesh.split_edge(edge, t)
            new_vertices.append(new_vert)
        
        # Connect new vertices to form the loop
        for i in range(len(new_vertices)):
            new_edge = self.mesh.add_edge(new_vertices[i], new_vertices[(i+1) % len(new_vertices)])
            new_edges.append(new_edge)
        
        return EdgeLoop(new_vertices, new_edges)
```

### Operation 2: Bridge Edges (PS-Solved)

**Traditional**: Select two edge loops, bridge creates connecting faces.

**Conformal**: PS operators solve for the bridging surface that minimizes distortion.

```python
class ConformalBridge:
    """
    Bridge two edge loops using PS-guided interpolation.
    """
    
    def bridge(self, loop_a: EdgeLoop, loop_b: EdgeLoop) -> List[Face]:
        """
        Create bridging faces between two loops.
        
        The conformal chart ensures the bridge has minimal shear.
        """
        # Create temporary boundary (loop_a + connection + loop_b + connection)
        boundary = self.construct_annular_boundary(loop_a, loop_b)
        
        # Solve BFF for the annular region
        # This gives us the optimal parameterization for the bridge
        chart = BFFChart.from_boundary(boundary)
        
        # The chart's u-direction spans from loop_a to loop_b
        # The chart's v-direction wraps around
        
        # Extract intermediate loops from chart isolines
        n_intermediate = self.estimate_loop_count(loop_a, loop_b)
        
        intermediate_loops = []
        for i in range(n_intermediate):
            t = (i + 1) / (n_intermediate + 1)
            loop = chart.extract_u_isoline(t)
            intermediate_loops.append(loop)
        
        # Create quad faces connecting successive loops
        all_loops = [loop_a] + intermediate_loops + [loop_b]
        
        faces = []
        for i in range(len(all_loops) - 1):
            faces.extend(self.connect_loops(all_loops[i], all_loops[i+1]))
        
        return faces
    
    def estimate_loop_count(self, loop_a: EdgeLoop, loop_b: EdgeLoop) -> int:
        """
        Estimate how many intermediate loops are needed.
        
        Based on:
        - Distance between loops
        - Average edge length in existing mesh
        - Distortion tolerance
        """
        distance = self.geodesic_distance(loop_a, loop_b)
        avg_edge_length = self.mesh.average_edge_length()
        
        return int(distance / avg_edge_length)
```

### Operation 3: Extrude (Conformal-Preserved)

**Traditional**: Select faces, extrude along normal.

**Conformal**: Extrusion preserves chart structure, new faces get inherited UVs.

```python
class ConformalExtrude:
    """
    Extrude faces while preserving conformal chart structure.
    """
    
    def extrude(self, faces: List[Face], distance: float, chart: BFFChart) -> ExtrusionResult:
        """
        Extrude faces, extending the conformal chart to new geometry.
        """
        # Get boundary of selection
        boundary_edges = self.get_boundary_edges(faces)
        
        # Extrude geometry
        new_vertices = {}
        for face in faces:
            for vert in face.vertices:
                if vert not in new_vertices:
                    normal = self.mesh.vertex_normal(vert)
                    new_pos = self.mesh.vertices[vert] + normal * distance
                    new_vertices[vert] = self.mesh.add_vertex(new_pos)
        
        # Create side faces (the extrusion walls)
        side_faces = []
        for edge in boundary_edges:
            v0, v1 = edge.vertices
            v2, v3 = new_vertices[v1], new_vertices[v0]
            side_face = self.mesh.add_face([v0, v1, v2, v3])
            side_faces.append(side_face)
        
        # Create top faces
        top_faces = []
        for face in faces:
            new_face_verts = [new_vertices[v] for v in face.vertices]
            top_face = self.mesh.add_face(new_face_verts)
            top_faces.append(top_face)
        
        # EXTEND CHART to new geometry
        # Side faces get chart from boundary condition (BFF extension)
        side_chart = self.extend_chart_to_sides(chart, boundary_edges, side_faces)
        
        # Top faces inherit chart from original faces (translated in w-direction)
        top_chart = self.translate_chart(chart, faces, top_faces, distance)
        
        return ExtrusionResult(
            side_faces=side_faces,
            top_faces=top_faces,
            side_chart=side_chart,
            top_chart=top_chart
        )
    
    def extend_chart_to_sides(self, chart: BFFChart, 
                              boundary: List[Edge], 
                              side_faces: List[Face]) -> BFFChart:
        """
        Extend conformal chart to extrusion side walls.
        
        Uses BFF's ExtendCurve algorithm:
        - Boundary UVs are known (from original chart)
        - New UVs are harmonically extended
        """
        # Boundary condition: existing UVs at the base edge loop
        boundary_uvs = {edge: chart.uv_at_edge(edge) for edge in boundary}
        
        # Solve for interior UVs via harmonic extension
        # (This is BFF's Algorithm 6: ExtendCurve)
        extended_chart = BFFChart.extend(
            boundary_uvs=boundary_uvs,
            target_faces=side_faces,
            method='harmonic'  # or 'conformal' for smooth corners
        )
        
        return extended_chart
```

### Operation 4: QMesh (Quick Mesh) — Conformal Anticipation

ZBrush's QMesh lets you click-drag to extrude/inset dynamically. With conformal charts:

```python
class ConformalQMesh:
    """
    Quick mesh operations with conformal anticipation.
    """
    
    def __init__(self, mesh: Mesh, chart: BFFChart):
        self.mesh = mesh
        self.chart = chart
        self.active_mode = None
    
    def on_hover(self, face: Face):
        """
        Show ghost preview of likely operation.
        """
        # Analyze face context
        context = self.analyze_context(face)
        
        if context.is_boundary:
            # Near edge → likely extrude outward
            self.show_extrude_ghost(face, direction='outward')
        
        elif context.high_distortion:
            # High distortion → likely needs subdivision
            self.show_subdivision_ghost(face)
        
        elif context.is_flat_region:
            # Flat region → likely inset
            self.show_inset_ghost(face)
        
        else:
            # Default: show both options faintly
            self.show_multiple_ghosts(face, ['extrude', 'inset', 'bridge'])
    
    def on_drag(self, face: Face, delta: Vec3):
        """
        Dynamic operation based on drag direction.
        """
        normal = self.mesh.face_normal(face)
        
        # Decompose drag into normal and tangent components
        normal_component = np.dot(delta, normal)
        tangent_component = delta - normal_component * normal
        
        if abs(normal_component) > np.linalg.norm(tangent_component):
            # Dragging along normal → extrude/inset
            if normal_component > 0:
                self.dynamic_extrude(face, normal_component)
            else:
                self.dynamic_inset(face, -normal_component)
        else:
            # Dragging tangentially → slide/move
            self.dynamic_slide(face, tangent_component)
    
    def dynamic_extrude(self, face: Face, distance: float):
        """
        Real-time extrusion preview with chart update.
        """
        # Preview geometry
        preview = self.compute_extrude_preview(face, distance)
        
        # Preview chart extension
        preview_chart = self.chart.preview_extension(face, distance)
        
        # Show ghost with chart isolines
        self.display_preview(preview, preview_chart)
```

---

## 5. The Sharp & Crane Conformal Cut Flow

From the research paper (Fig. 8 in your images), the algorithm:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONFORMAL CUT FLOW ALGORITHM                             │
│                    (Sharp & Crane, 2018)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   (1) Extract cut γ and submeshes M⁺, M⁻ from level set                   │
│                                                                             │
│   (2) Compute scale factor u⁺, u⁻ and adjoints v± := ∂u±/∂n              │
│                                                                             │
│   (3) Compute flow velocity along curve on each side                       │
│       σD := σD⁺ - σD⁻  (distortion jump)                                  │
│                                                                             │
│   (4) Harmonically extend flow velocity                                    │
│       φ̃ ← φ + τσD                                                         │
│                                                                             │
│   (5) Integrate distortion term of flow                                    │
│                                                                             │
│   (6) Integrate length term of flow                                        │
│       φ ← (Wᵥ + ταL L)⁻¹ Wᵥφ̃                                             │
│                                                                             │
│   (7) Redistance implicit function                                         │
│       |∇φ| = 1                                                             │
│                                                                             │
│   Iterate until convergence → optimal cut γ*                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Application to Topology Modeling

This algorithm **automatically finds optimal seam locations**. For ZModeler:

```python
class ConformalCutSuggester:
    """
    Use Sharp & Crane's conformal cut flow to suggest seams.
    """
    
    def suggest_cuts(self, mesh: Mesh, initial_cut: Optional[Curve] = None) -> List[Curve]:
        """
        Find optimal cut locations that minimize conformal distortion.
        """
        if initial_cut is None:
            # Initialize from medial axis or skeleton
            initial_cut = self.compute_initial_cut(mesh)
        
        # Level set representation of cut
        phi = self.curve_to_levelset(initial_cut, mesh)
        
        # Iterate conformal cut flow
        for iteration in range(self.max_iterations):
            # Step 1: Extract cut and submeshes
            gamma, M_plus, M_minus = self.extract_cut_submeshes(phi, mesh)
            
            # Step 2: Compute scale factors
            u_plus = self.solve_yamabe(M_plus, gamma)
            u_minus = self.solve_yamabe(M_minus, gamma)
            
            # Step 3: Compute flow velocity (distortion jump)
            sigma_D = self.compute_distortion_jump(u_plus, u_minus, gamma)
            
            # Step 4-5: Advect level set
            phi_tilde = phi + self.tau * sigma_D
            
            # Step 6: Integrate length term (regularization)
            phi = self.integrate_length_term(phi_tilde)
            
            # Step 7: Redistance
            phi = self.redistance(phi)
            
            # Check convergence
            if self.converged(phi, phi_prev):
                break
            
            phi_prev = phi
        
        # Extract final cut
        optimal_cut = self.levelset_to_curve(phi, mesh)
        
        return [optimal_cut]
    
    def compute_initial_cut(self, mesh: Mesh) -> Curve:
        """
        Initialize cut from mesh skeleton/medial axis.
        
        Good initial guesses:
        - Medial axis
        - Ridge lines of Gaussian curvature
        - User-specified strokes
        """
        # Compute mesh skeleton
        skeleton = self.compute_skeleton(mesh)
        
        # Convert to curve
        return skeleton.to_curve()
```

---

## 6. Chart-Aware Retopology Workflow

### The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CHART-AWARE RETOPOLOGY                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT: High-poly sculpt or scan                                          │
│                                                                             │
│   STEP 1: AUTOMATIC CHART SEGMENTATION                                     │
│   ─────────────────────────────────────────                                │
│   • Compute Gaussian curvature K                                           │
│   • Run conformal cut flow to find optimal seams                           │
│   • Segment mesh into charts                                               │
│                                                                             │
│   STEP 2: BFF PARAMETERIZATION PER CHART                                   │
│   ─────────────────────────────────────────                                │
│   • Compute conformal map via BFF                                          │
│   • Cherrier boundary conditions preserve corner angles                    │
│   • Each chart gets low-distortion UVs                                     │
│                                                                             │
│   STEP 3: EDGE FLOW FROM ISOLINES                                          │
│   ─────────────────────────────────────────                                │
│   • Extract u-isolines as horizontal loops                                 │
│   • Extract v-isolines as vertical loops                                   │
│   • Spacing determined by distortion (denser where |u| high)               │
│                                                                             │
│   STEP 4: QUAD MESH GENERATION                                             │
│   ─────────────────────────────────────────                                │
│   • Intersect u and v isolines → quad vertices                            │
│   • Connect to form quad faces                                             │
│   • Handle chart boundaries (seam stitching)                               │
│                                                                             │
│   STEP 5: INTERACTIVE REFINEMENT                                           │
│   ─────────────────────────────────────────                                │
│   • User adjusts loop density (re-solve BFF)                               │
│   • User adds/removes seams (re-segment)                                   │
│   • Real-time preview via ghost scaffolding                                │
│                                                                             │
│   OUTPUT: Clean quad mesh with optimal edge flow                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation

```python
class ChartAwareRetopology:
    """
    Full retopology pipeline using BFF charts.
    """
    
    def __init__(self, source_mesh: Mesh):
        self.source = source_mesh
        self.charts = []
        self.retopo_mesh = None
    
    def auto_segment(self):
        """
        Step 1: Automatically segment into charts.
        """
        # Compute curvature
        K = compute_gaussian_curvature(self.source)
        
        # Find optimal cuts via conformal cut flow
        cut_suggester = ConformalCutSuggester()
        cuts = cut_suggester.suggest_cuts(self.source)
        
        # Segment mesh along cuts
        self.charts = segment_mesh_by_cuts(self.source, cuts)
        
        return self.charts
    
    def parameterize_charts(self):
        """
        Step 2: BFF parameterization for each chart.
        """
        for chart in self.charts:
            # Build cotangent Laplacian
            L = build_cotan_laplacian(chart.mesh)
            
            # Cholesky factorization (reused for all operations)
            chart.cholesky = cholesky_factor(L)
            
            # Compute BFF conformal map
            # Using Cherrier boundary conditions for corner preservation
            chart.uv = BFF.compute(
                mesh=chart.mesh,
                boundary_mode='angles',  # Preserve corner angles
                cholesky=chart.cholesky
            )
            
            # Store distortion for visualization
            chart.distortion = compute_distortion(chart.mesh, chart.uv)
    
    def extract_edge_flow(self, target_density: float = 1.0):
        """
        Step 3: Extract edge loops from chart isolines.
        """
        all_loops = []
        
        for chart in self.charts:
            # Compute number of loops based on chart size and density
            u_range = chart.uv[:, 0].max() - chart.uv[:, 0].min()
            v_range = chart.uv[:, 1].max() - chart.uv[:, 1].min()
            
            n_u_loops = int(u_range / target_density)
            n_v_loops = int(v_range / target_density)
            
            # Adjust density by distortion (more loops where distortion high)
            n_u_loops = self.adjust_by_distortion(n_u_loops, chart, 'u')
            n_v_loops = self.adjust_by_distortion(n_v_loops, chart, 'v')
            
            # Extract isolines
            u_loops = chart.extract_u_isolines(n_u_loops)
            v_loops = chart.extract_v_isolines(n_v_loops)
            
            all_loops.extend(u_loops)
            all_loops.extend(v_loops)
        
        return all_loops
    
    def generate_quad_mesh(self, u_loops: List[EdgeLoop], v_loops: List[EdgeLoop]):
        """
        Step 4: Generate quad mesh from loop intersections.
        """
        # Find all intersections
        intersections = []
        for u_loop in u_loops:
            for v_loop in v_loops:
                intersection = find_intersection(u_loop, v_loop)
                if intersection is not None:
                    intersections.append(intersection)
        
        # Create vertices at intersections
        vertices = [self.source.sample_point(isec.position) for isec in intersections]
        
        # Create quad faces
        faces = []
        # ... (grid connectivity based on loop topology)
        
        self.retopo_mesh = Mesh(vertices, faces)
        return self.retopo_mesh
```

---

## 7. Interactive Features (ZModeler Style)

### Density Brush

Paint to adjust edge loop density locally:

```python
class DensityBrush:
    """
    Paint to increase/decrease edge loop density.
    """
    
    def __init__(self, retopo: ChartAwareRetopology):
        self.retopo = retopo
        self.density_map = np.ones(len(retopo.source.vertices))
    
    def paint(self, position: Vec3, radius: float, delta: float):
        """
        Paint density adjustment.
        
        delta > 0: Increase density (more loops)
        delta < 0: Decrease density (fewer loops)
        """
        # Find affected vertices
        affected = self.retopo.source.vertices_in_radius(position, radius)
        
        # Apply gaussian falloff
        for vert in affected:
            dist = np.linalg.norm(self.retopo.source.vertices[vert] - position)
            weight = np.exp(-(dist / radius) ** 2)
            self.density_map[vert] += delta * weight
        
        # Clamp
        self.density_map = np.clip(self.density_map, 0.1, 5.0)
        
        # Recompute edge flow with new density
        self.retopo.update_edge_flow(self.density_map)
```

### Seam Editor

Click to add/remove chart seams:

```python
class SeamEditor:
    """
    Interactive seam editing.
    """
    
    def add_seam(self, start: Vec3, end: Vec3):
        """
        Add a seam between two points.
        """
        # Find geodesic path on surface
        path = self.find_geodesic(start, end)
        
        # Add as new cut
        self.retopo.add_cut(path)
        
        # Re-segment affected charts
        self.retopo.re_segment()
        
        # Re-parameterize (fast: reuse Cholesky factors for unchanged charts)
        self.retopo.re_parameterize()
    
    def remove_seam(self, seam: Curve):
        """
        Remove a seam, merging adjacent charts.
        """
        # Find charts on either side
        chart_a, chart_b = self.find_adjacent_charts(seam)
        
        # Merge charts
        merged = self.merge_charts(chart_a, chart_b)
        
        # Re-parameterize merged chart
        merged.parameterize()
        
        # Update display
        self.retopo.update_display()
```

### Flow Direction Control

Click to rotate edge flow orientation:

```python
class FlowDirectionControl:
    """
    Control the orientation of edge flow in each chart.
    """
    
    def rotate_flow(self, chart: Chart, angle: float):
        """
        Rotate the u-v directions in a chart.
        
        This rotates which direction is "horizontal" vs "vertical" loops.
        """
        # Rotation matrix
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        
        # Rotate UVs
        chart.uv = chart.uv @ R.T
        
        # Recompute isolines
        chart.recompute_isolines()
    
    def set_flow_from_stroke(self, chart: Chart, stroke: Stroke):
        """
        Set flow direction from user stroke.
        
        The stroke direction becomes the "u" direction.
        """
        # Project stroke to chart UV space
        stroke_uv = chart.project_to_uv(stroke)
        
        # Compute dominant direction
        direction = compute_dominant_direction(stroke_uv)
        
        # Compute rotation needed to align u with stroke
        current_u = np.array([1, 0])
        angle = np.arctan2(direction[1], direction[0]) - np.arctan2(current_u[1], current_u[0])
        
        self.rotate_flow(chart, angle)
```

---

## 8. Ghost Scaffolding Visualization

### What the Artist Sees

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GHOST SCAFFOLD LAYERS                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   LAYER 1: DISTORTION HEATMAP (toggle)                                     │
│   ─────────────────────────────────────                                    │
│   Red/blue overlay showing conformal distortion u                          │
│   Highlights where topology needs work                                     │
│                                                                             │
│   LAYER 2: SEAM SUGGESTIONS (toggle)                                       │
│   ─────────────────────────────────────                                    │
│   Dashed lines showing optimal cut locations                               │
│   From conformal cut flow algorithm                                        │
│                                                                             │
│   LAYER 3: EDGE LOOP GHOSTS (always on)                                    │
│   ─────────────────────────────────────                                    │
│   Semi-transparent lines showing conformal isolines                        │
│   These ARE the edge loops (just not committed yet)                        │
│                                                                             │
│   LAYER 4: QUAD PREVIEW (toggle)                                           │
│   ─────────────────────────────────────                                    │
│   Wireframe showing resulting quad mesh                                    │
│   Updates in real-time as loops change                                     │
│                                                                             │
│   LAYER 5: OPERATION PREVIEW (context-sensitive)                           │
│   ─────────────────────────────────────                                    │
│   Shows result of pending operation before commit                          │
│   Extrude preview, bridge preview, etc.                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Integration with Neural Sketch Field

### The Complete Stack

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED FRAMEWORK                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   USER INPUT                                                                │
│   ──────────                                                               │
│   Sketches, clicks, drags                                                  │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────┐                              │
│   │     NEURAL SKETCH FIELD (geometry)      │                              │
│   │     ────────────────────────────────    │                              │
│   │     Stroke → Biarc → FNO → Surface      │                              │
│   │     Anticipates WHAT geometry            │                              │
│   └─────────────────┬───────────────────────┘                              │
│                     │                                                       │
│                     ▼                                                       │
│   ┌─────────────────────────────────────────┐                              │
│   │     CONFORMAL TOPOLOGY MODELER          │                              │
│   │     ────────────────────────────────    │                              │
│   │     BFF → Charts → Isolines → Quads     │                              │
│   │     Determines HOW to mesh it            │                              │
│   └─────────────────┬───────────────────────┘                              │
│                     │                                                       │
│                     ▼                                                       │
│   ┌─────────────────────────────────────────┐                              │
│   │     GSS BACKEND (Mathematical Compiler)  │                              │
│   │     ────────────────────────────────    │                              │
│   │     Tri-Space → Guardrails → Output     │                              │
│   │     Guarantees topological validity      │                              │
│   └─────────────────────────────────────────┘                              │
│                     │                                                       │
│                     ▼                                                       │
│   OUTPUT: Valid mesh with optimal topology                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### FNO in Chart Space

The key fusion: **FNO operates on BFF charts**, not raw 3D space:

```python
class ChartFNO:
    """
    Fourier Neural Operator on conformal chart domain.
    
    Because BFF charts are nearly isometric, FNO predictions
    are resolution-invariant and geometrically stable.
    """
    
    def __init__(self, chart: BFFChart):
        self.chart = chart
        self.fno = FourierNeuralOperator(
            modes=12,
            width=32,
            input_channels=3,   # (u, v, curvature)
            output_channels=1   # predicted field
        )
    
    def predict_density_field(self) -> np.ndarray:
        """
        Predict optimal edge loop density across the chart.
        
        Input: (u, v, gaussian_curvature)
        Output: density multiplier (1.0 = default, >1 = more loops)
        """
        # Build input tensor on chart grid
        grid = self.chart.sample_grid(resolution=64)
        
        u = grid[:, :, 0]
        v = grid[:, :, 1]
        K = self.sample_curvature_on_grid(grid)
        
        input_tensor = np.stack([u, v, K], axis=-1)
        
        # FNO inference
        density = self.fno(input_tensor)
        
        return density
    
    def predict_crease_likelihood(self) -> np.ndarray:
        """
        Predict where creases (hard edges) should be.
        
        High values → suggest hard edge (for mechanical parts).
        """
        # Similar to density, but trained on crease detection
        # ...
        pass
```

---

## 10. Summary: How BFF Becomes ZModeler

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    THE CONFORMAL TOPOLOGY MODELER                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BFF PROVIDES                           ZMODELER-STYLE OPERATIONS         │
│   ────────────                           ─────────────────────────         │
│                                                                             │
│   Conformal isolines        ─────────►   Edge loop insertion              │
│   (u, v curves)                          (ghosts show where loops go)      │
│                                                                             │
│   Yamabe distortion         ─────────►   Topology guidance                 │
│   (scalar field u)                       (heatmap shows problem areas)     │
│                                                                             │
│   Conformal cut flow        ─────────►   Seam suggestions                  │
│   (Sharp & Crane)                        (optimal chart boundaries)        │
│                                                                             │
│   Poincaré-Steklov ops      ─────────►   Topology solving                  │
│   (Λ, Λ†)                               (given constraints, solve layout)  │
│                                                                             │
│   Cherrier boundary         ─────────►   Corner preservation               │
│   conditions                             (hard edges stay hard)            │
│                                                                             │
│   Cholesky factorization    ─────────►   Real-time interactivity          │
│   (one-time, reused)                     (instant preview updates)         │
│                                                                             │
│   Chart-based FNO           ─────────►   Learned predictions               │
│   (resolution-invariant)                 (density, creases, flow)          │
│                                                                             │
│   ═══════════════════════════════════════════════════════════════════════  │
│                                                                             │
│   THE RESULT:                                                               │
│   A topology modeler where the math tells you what to do,                  │
│   and you confirm or adjust.                                               │
│                                                                             │
│   Instead of:  "Click here to insert loop"                                 │
│   You get:     "Here's where loops should go. Click to confirm."           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*The conformal chart knows topology. You just confirm what it suggests.*
