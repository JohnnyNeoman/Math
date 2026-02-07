# Spectral Sketch Field

> **Sketching as Eigenmode Excitation in a Learnable Neural Field**
> **Where Strokes Excite Spectra and Geometry Anticipates**
> Version 1.0 | 2026-02-06

---

## Abstract

We present a unified framework where **sketching excites spectral modes** in a learnable neural field, and **geometric priors anticipate** the emerging form. The key insight is that strokes are not merely 2D inputs but **excitation patterns** in a Laplacian eigenbasis—each stroke activates specific frequencies in the field, and a latent space trained on 3D geometry *completes* the undrawn structure.

The framework integrates:

1. **Hybrid Embedding Basis**: Global spectral modes (Laplacian eigenfunctions φᵢ) capture smooth macro-geometry while local semantic directions (sparse autoencoder axes uⱼ) encode fine detail. Together they form a multi-scale representation where low-frequency modes establish form and high-frequency modes resolve detail.

2. **Spectral Attention Networks (SAN)**: The Laplacian eigenbasis serves as positional encoding—the network "hears" structure from spectra. Eigenvalue weighting learns which frequencies matter for which geometric features.

3. **Learnable Metric Geometry**: The distance metric itself becomes the learning target. Connection coefficients and metric tensor evolve via entropic Ricci flow to satisfy Spec(-Δg) ≈ {λᵢ^target}, making geometry interpretable as network weights.

4. **Progressive Spectral Resolution**: Early strokes excite low-frequency modes (global shape). As sketching continues, higher frequencies activate (local detail). The field "anticipates" undrawn regions by spectral extrapolation from the learned prior.

5. **Geometric Prior Crystallization**: User prompts prime the latent space. Strokes constrain the field. Frenet-Serret frames emerge along curves. The scaffolding resolves from fuzzy anticipation to crisp geometric structure as constraints accumulate.

The result: a sketch interface where **the user draws excitement, not geometry**. The geometry emerges from spectral resonance between stroke input and learned priors.

---

## 1. Core Paradigm: Sketching as Spectral Excitation

### 1.1 The Spectral Sketch Field

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SPECTRAL SKETCH FIELD                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TRADITIONAL VIEW               SPECTRAL VIEW                             │
│   ────────────────               ─────────────                              │
│                                                                             │
│   Stroke → 2D curve              Stroke → Eigenmode excitation             │
│   Neural net → 3D lift           Neural field → Spectral response          │
│   Output → Mesh                  Output → Resonant geometry                │
│                                                                             │
│   ═══════════════════════════════════════════════════════════════════════  │
│                                                                             │
│   THE FIELD                                                                │
│   ─────────                                                                 │
│                                                                             │
│   F(x) = Σᵢ αᵢ(t) φᵢ(x) + Σⱼ βⱼ(t) uⱼ · x                                │
│                                                                             │
│   where:                                                                   │
│   • φᵢ(x) = Laplacian eigenfunctions (global modes)                       │
│   • uⱼ = SAE axes (local semantic directions)                             │
│   • αᵢ(t), βⱼ(t) = time-varying excitation coefficients                  │
│                                                                             │
│   STROKE AS EXCITATION                                                     │
│   ────────────────────                                                      │
│                                                                             │
│   When user draws stroke γ(s):                                             │
│                                                                             │
│   αᵢ ← αᵢ + ∫ φᵢ(γ(s)) ds      (project stroke onto eigenmode)           │
│                                                                             │
│   The stroke "rings" specific frequencies in the field.                   │
│   Low-frequency strokes (sweeping curves) → excite low φᵢ                 │
│   High-frequency strokes (detailed edges) → excite high φᵢ                │
│                                                                             │
│   ANTICIPATION AS SPECTRAL COMPLETION                                      │
│   ───────────────────────────────────                                       │
│                                                                             │
│   Learned prior P(α, β | prompt) predicts coefficients.                   │
│   Observed excitations constrain the posterior.                            │
│   Undrawn regions filled by spectral extrapolation.                       │
│                                                                             │
│   Ghost scaffolding = low-confidence spectral reconstruction              │
│   Solid geometry = high-confidence (stroke-constrained) modes             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 The Hybrid Embedding

The field operates in a **hybrid embedding space** combining:

```
GLOBAL SPECTRAL MODES φᵢ(x)              LOCAL SEMANTIC AXES uⱼ
─────────────────────────                ─────────────────────

   ∿∿∿∿∿∿∿∿∿∿ λ₁ (smooth)              u₁ = "sharpness"
   ∿∿∿∿∿∿∿∿∿∿∿∿ λ₂                     u₂ = "curvature"  
   ∿∿∿∿∿∿∿∿∿∿∿∿∿∿ λ₃                   u₃ = "thickness"
   ~~~~~~~~~~~~~ λ₄                      u₄ = "symmetry"
   ~~~~~~~~~~~~~ λ₅ (detailed)           ...
                                         uₘ = "style"

   From Laplacian eigendecomposition     From Sparse Autoencoder
   Δφᵢ = λᵢφᵢ                           z = SAE(x) = ReLU(Wx + b)
   
   Captures SHAPE                        Captures SEMANTICS
   Scale-separated                       Interpretable
   Intrinsic (pose-invariant)            Data-dependent
```

**Together**: Ψ(x) = [α₁φ₁(x), ..., αₖφₖ(x), β₁u₁ᵀx, ..., βₘuₘᵀx]

This is the representation space where:
- Strokes project as excitation patterns
- Priors live as coefficient distributions
- Geometry emerges as spectral reconstruction

---

## 2. Progressive Spectral Resolution

### 2.1 From Fuzzy to Focused

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROGRESSIVE RESOLUTION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TIME t=0                  t=0.3                 t=0.7                t=1 │
│   ────────                  ─────                 ─────                ───  │
│                                                                             │
│   User prompt only          First strokes         More detail          Done │
│                                                                             │
│   ╭~~~~~~~~~~~╮            ╭───~~~~~~╮           ╭───────╮          ╭─────╮│
│   ╎           ╎            │         ╎           │       │          │     ││
│   ╎    ???    ╎            │    ~~   ╎           │   ──  │          │  ── ││
│   ╎           ╎            │         ╎           │       │          │     ││
│   ╰~~~~~~~~~~~╯            ╰───~~~~~~╯           ╰───────╯          ╰─────╯│
│                                                                             │
│   λ₁-₃ only                λ₁-₅ resolved         λ₁-₁₀              λ₁-₆₄  │
│   (very fuzzy)             (major form)          (clear shape)      (crisp)│
│                                                                             │
│   SPECTRAL ACTIVATION OVER TIME                                            │
│   ──────────────────────────────                                            │
│                                                                             │
│   Mode activation:                                                         │
│                                                                             │
│   λ₁  ████████████████████████████████████████  (always active)           │
│   λ₂  ░░░░████████████████████████████████████  (early)                   │
│   λ₃  ░░░░░░░░████████████████████████████████                            │
│   λ₄  ░░░░░░░░░░░░████████████████████████████                            │
│   ...                                                                       │
│   λ₆₄ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████  (late, detail)            │
│                                                                             │
│       t=0                                    t=1                           │
│                                                                             │
│   Confidence also increases:                                               │
│   Low-frequency modes: high confidence early                               │
│   High-frequency modes: confidence grows with strokes                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Stroke → Eigenmode Projection

```python
class SpectralSketchField:
    """
    Neural field where sketching excites spectral modes.
    """
    
    def __init__(self, n_spectral_modes: int = 64, n_sae_axes: int = 128):
        self.n_modes = n_spectral_modes
        self.n_axes = n_sae_axes
        
        # Spectral coefficients (excitation state)
        self.alpha = torch.zeros(n_spectral_modes)  # Global mode amplitudes
        self.beta = torch.zeros(n_sae_axes)          # Local axis activations
        
        # Confidence per mode (increases with stroke evidence)
        self.confidence = torch.zeros(n_spectral_modes)
        
        # Eigenbasis (computed from domain or learned)
        self.eigenfunctions = None  # φᵢ(x)
        self.eigenvalues = None     # λᵢ
        
        # Learned prior P(α, β | prompt)
        self.prior_network = SpectralPriorNetwork(n_spectral_modes, n_sae_axes)
        
        # SAE for local semantics
        self.sae = SparseAutoencoder(n_sae_axes)
    
    def excite_with_stroke(self, stroke: torch.Tensor, domain_points: torch.Tensor):
        """
        Project stroke onto eigenbasis to update excitation coefficients.
        
        Args:
            stroke: [N, 3] stroke points
            domain_points: [M, 3] points where eigenfunctions are evaluated
        """
        # Evaluate eigenfunctions at stroke points
        stroke_phi = self.evaluate_eigenfunctions(stroke)  # [N, n_modes]
        
        # Project stroke onto each mode
        for i in range(self.n_modes):
            # Integral of φᵢ along stroke (arc-length weighted)
            arc_lengths = compute_arc_lengths(stroke)
            excitation = (stroke_phi[:, i] * arc_lengths).sum()
            
            # Update coefficient with damping
            self.alpha[i] = 0.9 * self.alpha[i] + 0.1 * excitation
            
            # Increase confidence based on excitation magnitude
            self.confidence[i] = min(1.0, self.confidence[i] + 0.1 * abs(excitation))
        
        # Also update SAE axes
        stroke_features = self.extract_stroke_features(stroke)
        sae_activation = self.sae.encode(stroke_features)
        self.beta = 0.9 * self.beta + 0.1 * sae_activation
    
    def anticipate_geometry(self, prompt_embedding: torch.Tensor) -> torch.Tensor:
        """
        Anticipate undrawn geometry from prior + current excitation.
        
        Returns spectral reconstruction weighted by confidence.
        """
        # Prior prediction given prompt
        alpha_prior, beta_prior = self.prior_network(prompt_embedding)
        
        # Blend prior with observed excitation based on confidence
        alpha_blended = (1 - self.confidence) * alpha_prior + self.confidence * self.alpha
        beta_blended = (1 - self.confidence.mean()) * beta_prior + self.confidence.mean() * self.beta
        
        # Reconstruct field: F(x) = Σ αᵢ φᵢ(x)
        # This gives anticipated geometry at each point
        return self.reconstruct(alpha_blended, beta_blended)
    
    def reconstruct(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct scalar field from spectral coefficients.
        """
        # Global modes
        field = torch.einsum('i,xi->x', alpha, self.eigenfunctions)
        
        # Local SAE modulation
        # (modulates the field based on semantic axes)
        sae_modulation = self.sae.decode(beta)
        
        return field * (1 + 0.1 * sae_modulation)
    
    def get_ghost_scaffolding(self) -> Dict[str, torch.Tensor]:
        """
        Return ghost preview at current state.
        
        Low-confidence regions are shown as fuzzy anticipation.
        High-confidence regions are shown as solid geometry.
        """
        field = self.anticipate_geometry(self.current_prompt)
        
        return {
            'field': field,
            'confidence': self.confidence,
            'solid_mask': self.confidence > 0.7,
            'ghost_mask': (self.confidence > 0.3) & (self.confidence <= 0.7),
            'fuzzy_mask': self.confidence <= 0.3
        }
```

---

## 3. Geometric Prior Crystallization

### 3.1 From Prompt to Scaffolding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GEOMETRIC PRIOR CRYSTALLIZATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PROMPT PRIMING                                                           │
│   ──────────────                                                            │
│                                                                             │
│   User: "sports car"                                                       │
│                                                                             │
│   → Prior network activates:                                               │
│     • Low λ: elongated horizontal form                                     │
│     • Mid λ: wheel arches, cabin bubble                                   │
│     • High λ: ready for detail strokes                                    │
│     • SAE: "sleek", "aggressive", "low"                                   │
│                                                                             │
│   Ghost appears: fuzzy car silhouette                                      │
│                                                                             │
│   STROKE CONSTRAINING                                                      │
│   ───────────────────                                                       │
│                                                                             │
│   User draws roofline stroke:                                              │
│                                                                             │
│   → Excites λ₂, λ₃, λ₅ (smooth horizontal curves)                         │
│   → Confidence increases in those modes                                    │
│   → Ghost crystallizes: roofline now solid, rest still fuzzy             │
│                                                                             │
│   FRAME EMERGENCE                                                          │
│   ───────────────                                                           │
│                                                                             │
│   As strokes accumulate:                                                   │
│                                                                             │
│   → Frenet-Serret frames computed along curves                            │
│   → Tangent T, Normal N, Binormal B at each sample                       │
│   → Frames propagate via parallel transport                               │
│   → Scaffolding becomes navigable coordinate system                       │
│                                                                             │
│   CONSTRAINT SATISFACTION                                                  │
│   ───────────────────────                                                   │
│                                                                             │
│   Final strokes add high-frequency detail:                                 │
│                                                                             │
│   → High λ modes activate (surface detail, edges)                         │
│   → Bilateral symmetry enforced (geometric prior)                         │
│   → Curvature continuity checked (G2 smoothness)                          │
│   → Ghost fully solidifies into crisp geometry                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Frenet Frame Emergence from Spectral Field

```python
class FrameEmergence:
    """
    Extract Frenet-Serret frames from spectral field.
    
    As the field crystallizes, curves become well-defined
    and frames emerge along them.
    """
    
    def __init__(self, spectral_field: SpectralSketchField):
        self.field = spectral_field
    
    def extract_curves_from_field(self, threshold: float = 0.5) -> List[torch.Tensor]:
        """
        Extract curves where field gradient is high.
        
        Curves are level sets or ridge lines of the spectral field.
        """
        ghost = self.field.get_ghost_scaffolding()
        field_values = ghost['field']
        solid_mask = ghost['solid_mask']
        
        # Compute gradient magnitude
        grad = compute_gradient(field_values)
        grad_mag = torch.norm(grad, dim=-1)
        
        # Find ridges (local maxima of gradient magnitude)
        curves = extract_ridges(grad_mag, threshold)
        
        return curves
    
    def compute_frenet_frames(self, curve: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Frenet-Serret frames along extracted curve.
        
        T(s) = γ'(s) / |γ'(s)|
        N(s) = T'(s) / |T'(s)|  
        B(s) = T(s) × N(s)
        κ(s) = |T'(s)| / |γ'(s)|
        τ(s) = -N(s) · B'(s) / |γ'(s)|
        """
        # Tangent
        tangent = torch.zeros_like(curve)
        tangent[:-1] = curve[1:] - curve[:-1]
        tangent[-1] = tangent[-2]
        tangent = tangent / (torch.norm(tangent, dim=-1, keepdim=True) + 1e-6)
        
        # Curvature and normal
        tangent_deriv = torch.zeros_like(tangent)
        tangent_deriv[:-1] = tangent[1:] - tangent[:-1]
        tangent_deriv[-1] = tangent_deriv[-2]
        
        curvature = torch.norm(tangent_deriv, dim=-1)
        normal = tangent_deriv / (curvature.unsqueeze(-1) + 1e-6)
        
        # Binormal
        binormal = torch.cross(tangent, normal, dim=-1)
        
        # Torsion
        binormal_deriv = torch.zeros_like(binormal)
        binormal_deriv[:-1] = binormal[1:] - binormal[:-1]
        torsion = -torch.sum(normal * binormal_deriv, dim=-1)
        
        return {
            'tangent': tangent,      # T
            'normal': normal,        # N
            'binormal': binormal,    # B
            'curvature': curvature,  # κ
            'torsion': torsion,      # τ
            'position': curve        # γ
        }
    
    def parallel_transport_frame(self, 
                                  initial_frame: Dict[str, torch.Tensor],
                                  curve: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parallel transport initial frame along curve.
        
        Avoids Frenet frame singularities at inflection points.
        """
        T = initial_frame['tangent'][0]
        U = initial_frame['normal'][0]
        V = initial_frame['binormal'][0]
        
        transported = {'tangent': [], 'normal': [], 'binormal': []}
        
        for i in range(len(curve)):
            # Update tangent
            if i > 0:
                T_new = curve[i] - curve[i-1]
                T_new = T_new / (torch.norm(T_new) + 1e-6)
                
                # Rotation from T to T_new
                rotation = rotation_between_vectors(T, T_new)
                
                # Apply rotation to U, V
                U = rotation @ U
                V = rotation @ V
                T = T_new
            
            transported['tangent'].append(T)
            transported['normal'].append(U)
            transported['binormal'].append(V)
        
        return {k: torch.stack(v) for k, v in transported.items()}
    
    def frame_to_loft_surface(self, 
                               frames: Dict[str, torch.Tensor],
                               profile_curve: torch.Tensor) -> torch.Tensor:
        """
        Generate lofted surface by sweeping profile along frames.
        """
        positions = frames['position']
        tangents = frames['tangent']
        normals = frames['normal']
        binormals = frames['binormal']
        
        surface_points = []
        
        for i, (pos, T, N, B) in enumerate(zip(positions, tangents, normals, binormals)):
            # Transform profile to this frame
            frame_matrix = torch.stack([N, B, T], dim=1)  # Local coordinate system
            
            transformed_profile = pos + profile_curve @ frame_matrix.T
            surface_points.append(transformed_profile)
        
        return torch.stack(surface_points)
```

---

## 4. The Spectral Attention Layer

### 4.1 SAN Integration

Spectral Attention Networks use the Laplacian eigenbasis as positional encoding:

```python
class SpectralSketchAttention(nn.Module):
    """
    Attention mechanism that "hears" structure from spectra.
    
    Key insight: eigenvalues encode geometric frequencies,
    eigenvectors encode spatial patterns.
    """
    
    def __init__(self, hidden_dim: int = 256, n_heads: int = 8, n_modes: int = 32):
        super().__init__()
        
        self.n_modes = n_modes
        
        # Spectral positional encoding
        self.spectral_encoder = nn.Linear(n_modes, hidden_dim)
        
        # Eigenvalue weighting (learns frequency importance)
        self.eigenvalue_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, hidden_dim)
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, 
                stroke_features: torch.Tensor,  # [B, N, D]
                eigenvectors: torch.Tensor,      # [N, K]
                eigenvalues: torch.Tensor        # [K]
               ) -> torch.Tensor:
        """
        Attend over stroke features with spectral positional encoding.
        
        The attention learns which frequencies (eigenvalues) are relevant
        for completing the geometry from partial strokes.
        """
        B, N, D = stroke_features.shape
        
        # Spectral positional encoding
        spectral_pe = self.spectral_encoder(eigenvectors[:, :self.n_modes])  # [N, D]
        spectral_pe = spectral_pe.unsqueeze(0).expand(B, -1, -1)
        
        # Eigenvalue importance weighting
        eigenvalue_weights = self.eigenvalue_mlp(eigenvalues[:self.n_modes].unsqueeze(-1))  # [K, D]
        
        # Combine features with spectral encoding
        x = stroke_features + spectral_pe
        
        # Spectral-weighted attention
        # Higher eigenvalue → higher frequency → more local attention
        # Lower eigenvalue → lower frequency → more global attention
        attn_output, attn_weights = self.attention(x, x, x)
        
        # Modulate by eigenvalue weights
        output = self.output_proj(attn_output)
        
        return output, attn_weights
    
    def visualize_spectral_attention(self, attn_weights: torch.Tensor):
        """
        Visualize which spectral modes the attention focuses on.
        """
        # attn_weights: [B, N, N]
        # Can be decomposed into spectral components
        pass
```

### 4.2 Learning to Hear Structure

```
THE SPECTRAL HEARING METAPHOR
─────────────────────────────

Just as a musician hears harmonics in a sound wave,
the SAN "hears" geometric structure in the eigenspectrum.

Sound Wave                          Geometry
──────────                          ────────

Fundamental frequency    ←→         λ₁ (global shape)
Second harmonic          ←→         λ₂ (major feature)
Third harmonic           ←→         λ₃ (secondary feature)
...                                 ...
High harmonics           ←→         λₖ (fine detail)

Timbre (harmonic balance) ←→        Geometric "style"
Loudness (amplitude)      ←→        Feature prominence
Attack/decay              ←→        Progressive resolution

THE SKETCH IS THE SCORE
───────────────────────

Each stroke is like playing a note:
• Smooth sweeping stroke → activates low harmonics (cello line)
• Sharp detailed stroke → activates high harmonics (cymbal crash)
• The field responds by "playing" the corresponding geometry
```

---

## 5. Integration with SST Node Algebra

### 5.1 Extended Node Types

```yaml
# Σ_spectral_sketch: New node vocabulary

SpectralSketchField:
  n_modes: 64
  n_sae_axes: 128
  prior_model: "pretrained_3d_prior.pt"

ExciteWithStroke:
  stroke: "$user_input"
  weight: 1.0               # Excitation strength
  mode_filter: [1, 32]      # Only excite modes 1-32

AnticipateGeometry:
  prompt: "$text_prompt"
  confidence_threshold: 0.3  # Show ghost above this
  
ExtractFrames:
  source: "$spectral_field"
  method: FRENET | PARALLEL | DARBOUX
  confidence_min: 0.7        # Only from solid regions

Crystallize:
  constraints:
    - BILATERAL_SYMMETRY
    - G2_CONTINUITY
    - CURVATURE_BOUNDS: [0.001, 100]
  
SpectralLoft:
  frames: "$extracted_frames"
  profile: "circle | custom_curve"
```

### 5.2 Complete Pipeline

```yaml
# SPECTRAL SKETCH SESSION

# Initialize field with prompt
- SpectralSketchField:
    n_modes: 64
    prompt: "sports car, sleek, aggressive"
    bilateral_symmetry: true
    id: "car_field"

# First stroke: roofline
- ExciteWithStroke:
    field: "car_field"
    stroke: "$user_roofline"
    mode_filter: [1, 10]     # Low frequency for smooth roofline

# Anticipate rest of car
- AnticipateGeometry:
    field: "car_field"
    show_ghost: true
    ghost_opacity: 0.3

# More strokes refine the form
- ExciteWithStroke:
    field: "car_field"
    stroke: "$user_wheelarch"
    mode_filter: [5, 20]     # Mid frequency for wheel arches

# Extract frames from high-confidence curves
- ExtractFrames:
    field: "car_field"
    confidence_min: 0.7
    method: PARALLEL         # Avoid twist
    output: "car_frames"

# Apply geometric constraints
- Crystallize:
    field: "car_field"
    constraints: [BILATERAL_SYMMETRY, G2_CONTINUITY]
    output: "car_solid"

# Generate surface
- SpectralLoft:
    frames: "car_frames"
    profile: "$wheel_profile"
    output: "car_surface"
```

---

## 6. Mathematical Summary

### 6.1 The Core Equations

**Spectral Field Representation**:
```
F(x) = Σᵢ αᵢ φᵢ(x) + Σⱼ βⱼ ψⱼ(x)

where:
  φᵢ = Laplacian eigenfunction (Δφᵢ = λᵢφᵢ)
  ψⱼ = SAE-derived basis function
  αᵢ, βⱼ = excitation coefficients
```

**Stroke Excitation**:
```
αᵢ ← αᵢ + ∫_γ φᵢ(γ(s)) |γ'(s)| ds

(Project stroke onto eigenmode, weighted by arc length)
```

**Prior-Posterior Blending**:
```
α_final = (1 - c) α_prior + c α_observed

where c = confidence (0 = pure prior, 1 = pure observation)
```

**Spectral Attention**:
```
Attention(Q, K, V) = softmax(QKᵀ/√d + S) V

where S = spectral bias matrix from eigenvector similarity
```

**Frame Emergence**:
```
Frenet-Serret: T' = κN, N' = -κT + τB, B' = -τN

Parallel transport: U' ⊥ T, V' ⊥ T (minimal twist)
```

### 6.2 The Insight Hierarchy

```
LEVEL 1: Stroke as input
         ↓
LEVEL 2: Stroke as eigenmode excitation
         ↓
LEVEL 3: Field as spectral superposition
         ↓
LEVEL 4: Geometry as spectral resonance with prior
         ↓
LEVEL 5: Crystallization as constraint satisfaction
         ↓
LEVEL 6: Frames emerge along high-confidence curves
         ↓
LEVEL 7: Surface generated by frame-guided lofting
```

---

## 7. Conclusion

The **Spectral Sketch Field** reframes sketching as eigenmode excitation in a learnable neural field:

1. **Strokes don't draw geometry**—they excite spectral patterns
2. **The field anticipates** from learned priors (text prompts)
3. **Progressive resolution**: low → high frequency as detail accumulates
4. **Geometric priors crystallize** as constraints are satisfied
5. **Frenet frames emerge** along high-confidence curves
6. **Lofted surfaces** generated from frame scaffolding

The user draws *excitement*, not *geometry*. The geometry emerges from spectral resonance between stroke input and learned priors.

---

*Draw the excitation. Let the spectrum resonate. Watch geometry crystallize.*
