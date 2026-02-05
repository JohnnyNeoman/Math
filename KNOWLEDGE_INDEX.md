# Knowledge Index: 3D Tools ML Hybrid

> **Portable Index for Functional Parametric L-System Architecture**
> Version 1.1 | Last Updated: 2026-01-27

---

## Quick Navigation

| Entry Point | Purpose | Start Here If... |
|-------------|---------|------------------|
| [ABSTRACT.md](ABSTRACT.md) | Vision & roadmap | Sharing the project, understanding trajectory |
| [CLAUDE_CODE_HANDOFF.md](CLAUDE_CODE_HANDOFF.md) | **Implementation bootstrap** | Starting Claude Code session |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | One-page cheat sheet | Quick lookup |
| [math_foundations.md](core/math_foundations.md) | Core math layer | Understanding the algebra |
| [SKELETAL_SINGLETON_TREE.md](core/SKELETAL_SINGLETON_TREE.md) | Architecture overview | Understanding the system |
| [node_algebra.md](core/node_algebra.md) | Functional alphabet | Composing operations |
| [state_schema.md](core/state_schema.md) | Matrix stack | Working with transforms |
| [mutation_schema.md](core/mutation_schema.md) | Geometry operations | Adding primitives/CSG |
| [transpiler_spec.md](core/transpiler_spec.md) | Platform mappings | Implementing for Maya/Blender/Unreal |

---

## The Core Insight

```
Traditional L-System:  "F+F-F"   →  string rewriting  →  drawing commands
Functional L-System:   Node tree  →  tree traversal   →  matrix × geometry

The strings ARE the expressions. The expressions produce geometry.
The math layer is so compact that the code writes itself.
```

---

## File Registry

| ID | Path | Category | Priority | Tags |
|----|------|----------|----------|------|
| f001 | `core/SKELETAL_SINGLETON_TREE.md` | theory | 1 | `core`, `architecture` |
| f002 | `core/state_schema.md` | theory | 1 | `state`, `matrix` |
| f003 | `core/mutation_schema.md` | theory | 1 | `mutation`, `geometry` |
| f004 | `core/transpiler_spec.md` | implementation | 2 | `maya`, `blender`, `unreal` |
| f005 | `core/math_foundations.md` | theory | 1 | `math`, `gram-schmidt`, `symmetry` |
| f006 | `core/node_algebra.md` | theory | 1 | `nodes`, `algebra`, `generative` |
| f007 | `QUICK_REFERENCE.md` | navigation | 2 | `reference`, `cheatsheet` |
| f009 | `ABSTRACT.md` | vision | 1 | `abstract`, `vision`, `ml`, `roadmap` |
| f011 | `CLAUDE_CODE_HANDOFF.md` | implementation | 1 | `handoff`, `bootstrap`, `ue5`, `mcp` |
| f012 | `core/extended_state_algebra.md` | theory | 1 | `spread`, `frame`, `field`, `topology` |
| f013 | `core/ml_integration.md` | theory | 1 | `ml`, `neural-operators`, `gcnn`, `ghost` |
| f014 | `core/rule_patterns.md` | theory | 1 | `rules`, `patterns`, `lsystem`, `library` |
| f015 | `core/plugin_bridges.md` | theory | 1 | `bridges`, `polyflow`, `integration` |

---

## Topic Index

### A
- **Algebra** (composition rules) → [math_foundations.md](core/math_foundations.md#3-affine-transform-algebra)
- **Alignment** (smart placement) → [math_foundations.md](core/math_foundations.md#1-gram-schmidt-alignment)
- **Architecture** → [SKELETAL_SINGLETON_TREE.md](core/SKELETAL_SINGLETON_TREE.md)

### C
- **Collapse** (spread → geometry) → [extended_state_algebra.md](core/extended_state_algebra.md#4-topology-aware-mutations)
- **Composition** → [node_algebra.md](core/node_algebra.md#6-the-generative-property)
- **CSG** (boolean ops) → [mutation_schema.md](core/mutation_schema.md#constructive-solid-geometry-σ_csg)

### F
- **Field** (continuous deformation) → [extended_state_algebra.md](core/extended_state_algebra.md#3-field-operations)
- **Frame** (topology → matrices) → [extended_state_algebra.md](core/extended_state_algebra.md#1-topological-frame-binding)
- **Frenet Frame** (curve basis) → [extended_state_algebra.md](core/extended_state_algebra.md#frame-modes)

### G
- **Gram-Schmidt** → [math_foundations.md](core/math_foundations.md#1-gram-schmidt-alignment)
- **Generative Property** → [node_algebra.md](core/node_algebra.md#6-the-generative-property)

### M
- **Matrix Stack** → [state_schema.md](core/state_schema.md)
- **Mirror Node** → [node_algebra.md](core/node_algebra.md#mirror-node-bifurcation)
- **Mutation** → [mutation_schema.md](core/mutation_schema.md)

### N
- **Node Algebra** → [node_algebra.md](core/node_algebra.md)
- **Node Types** → [node_algebra.md](core/node_algebra.md#3-node-types-the-functional-grammar)

### P
- **Platform Symmetry** → [math_foundations.md](core/math_foundations.md#2-platform-symmetry-portable-world-center)
- **Push/Pop** → [state_schema.md](core/state_schema.md#push-scope-begin)
- **Primitives** → [mutation_schema.md](core/mutation_schema.md#primitive-operations-σ_primitive)

### R
- **Radial Node** → [node_algebra.md](core/node_algebra.md#radial-node)
- **Reflection Formula** → [math_foundations.md](core/math_foundations.md#the-formula)

### L
- **Loft** (spreads → surface) → [extended_state_algebra.md](core/extended_state_algebra.md#4-topology-aware-mutations)

### S
- **Sample** (geometry → spread) → [extended_state_algebra.md](core/extended_state_algebra.md#2-spread-operations)
- **Scope Isolation** → [state_schema.md](core/state_schema.md#scope-rules)
- **Spread** (array of matrices) → [extended_state_algebra.md](core/extended_state_algebra.md#2-spread-operations)
- **State** → [state_schema.md](core/state_schema.md)
- **Symmetry** → [math_foundations.md](core/math_foundations.md#2-platform-symmetry-portable-world-center)

### T
- **Transpilation** → [transpiler_spec.md](core/transpiler_spec.md)
- **Transform Algebra** → [math_foundations.md](core/math_foundations.md#3-affine-transform-algebra)

### W
- **Walker** (execution state) → [node_algebra.md](core/node_algebra.md#2-the-walker-execution-state)

---

## Concept Map

```
                         MATH FOUNDATIONS
                               │
           ┌───────────────────┼───────────────────┐
           │                   │                   │
     Gram-Schmidt        Affine Algebra      Platform Symmetry
     (alignment)         (T, R, S)           (M = P×S×P⁻¹)
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                    SKELETAL SINGLETON TREE
                               │
               ┌───────────────┴───────────────┐
               │                               │
            STATE                          MUTATION
         (Matrix Stack)                   (Geometry)
               │                               │
        ┌──────┴──────┐                 ┌──────┴──────┐
        │             │                 │             │
     Transforms    Scopes           Primitives     CSG
     T, R, S       [ ]              □ ○ △         ⊕ ⊖ ⊗
               │
               ▼
         NODE ALGEBRA
        (Functional Σ)
               │
    ┌──────────┼──────────┐
    │          │          │
  Mirror    Radial    Instance
 (bifurcate) (replicate) (emit)
```

---

## Retrieval Patterns

### Pattern 1: "I want to understand the math layer"
**Sequence**: f005 → f006 → f001
1. Math foundations (algebra)
2. Node algebra (composition)
3. SST architecture (overview)

### Pattern 2: "I want to implement symmetry"
**Sequence**: f005 → f006
1. Platform reflection formula
2. Mirror node implementation

### Pattern 3: "I want to add a new operation"
**Sequence**: f006 → f003 → f004
1. Study node algebra structure
2. Check mutation schema
3. Add platform mappings

### Pattern 4: "Quick reference for formulas"
**Sequence**: QUICK_REFERENCE.md
1. One-page cheat sheet

### Pattern 5: "I want Houdini-style operations (spreads, fields, topology)"
**Sequence**: f012 → f006 → f005
1. Extended state algebra (spreads, frames, fields)
2. Node algebra (composition)
3. Math foundations (frame calculations)

---

## Project Structure

```
3D_tools_ML_hybrid/
├── ABSTRACT.md                 # Vision document & roadmap
├── CLAUDE_CODE_HANDOFF.md      # ★ Implementation bootstrap (give to Claude Code)
├── KNOWLEDGE_INDEX.md          # Human-readable index (this file)
├── QUICK_REFERENCE.md          # One-page cheat sheet
├── index.jsonl                 # Machine-readable index
├── core/                       # SPECIFICATIONS
│   ├── math_foundations.md     # ★ Core algebra (start here for math)
│   ├── node_algebra.md         # Functional alphabet, Walker
│   ├── extended_state_algebra.md # Spreads, Frames, Fields
│   ├── ml_integration.md       # ★ Neural operators, Ghost scaffolding
│   ├── rule_patterns.md        # L-system rewriting, pattern library
│   ├── plugin_bridges.md       # External tool integration
│   ├── SKELETAL_SINGLETON_TREE.md
│   ├── state_schema.md
│   ├── mutation_schema.md
│   └── transpiler_spec.md
├── scripts/
│   └── generate_index.py
└── implementations/            # PLATFORM CODE (Claude Code builds these)
    ├── universal/              # Pure math, no dependencies
    ├── unreal/                 # UE5 plugin (SymmetryKit)
    ├── blender/                # Blender addon
    └── maya/                   # Maya plugin
```

---

## Status

### Core Architecture
| Component | Status | Notes |
|-----------|--------|-------|
| Math Foundations | ✅ Complete | Gram-Schmidt, platform reflection, composition algebra |
| SST Architecture | ✅ Complete | Core theory documented |
| Node Algebra | ✅ Complete | Functional alphabet, generative property |
| State Schema | ✅ Complete | Matrix stack formalized |
| Mutation Schema | ✅ Complete | Operations catalogued |
| **Extended Algebra** | ✅ Complete | Spreads, Frames, Fields (Phase 6) |
| Quick Reference | ✅ Complete | One-page cheat sheet |
| Abstract/Vision | ✅ Complete | Roadmap documented |
| Transpiler Spec | 🔲 Skeleton | Platform mappings needed |
| Implementations | 🟡 In Progress | UE5 SymmetryKit started |

### Advanced Extensions
| Component | Status | Notes |
|-----------|--------|-------|
| **ML Integration** | ✅ Complete | FNO/GNO/G-CNN, Ghost scaffolding, Epipolar ML |
| **Rule Patterns** | ✅ Complete | L-system rewriting, pattern library |
| **Plugin Bridges** | ✅ Complete | PolyFlow, Blender, Maya, Web services |
| Collaboration | 🔲 Planned | Multi-user state sync |
| Validation | 🔲 Planned | Topological guardrails |

### ML Vision (Future)
| Component | Status | Notes |
|-----------|--------|-------|
| Neural Operators (FNO/GNO/GNP) | 🔲 Planned | Sketch-based modeling |
| Epipolar Geometry ML | 🔲 Planned | Stereo-Photogrammetric Sketcher |
| G-CNN Ghost Scaffolding | 🔲 Planned | Predictive symmetry |
| Topological Guardrails | 🔲 Planned | Persistent homology validation |

---

## Key Formulas (Quick Lookup)

**Gram-Schmidt Alignment**:
```
Y = normalize(N), X = normalize(cross(Up, Y)), Z = cross(X, Y)
```

**Platform Reflection**:
```
M_reflect = P × S × P⁻¹    where S = diag(-1,1,1,1)
```

**Matrix Stack**:
```
push(): append(copy(top)), pop(): remove top, transform(T): top = top × T
```

**Composition**:
```
Sequential: A ∘ B = A × B    (right-to-left)
Scoped: [ A ] isolates A     (push/pop)
```

---

*Index regenerated with: `python scripts/generate_index.py --root . --output index.jsonl`*
