# LLM-to-SST Bridge: Prompt-Driven Geometry

> **Minimal viable pipeline:** User prompt (text or image) → LLM → SST JSON → Walker → Geometry

---

## The Idea in One Sentence

An LLM reads a schema that teaches it the SST grammar, then translates natural language (or an image description) into a valid SST JSON tree that the Walker executes into 3D geometry.

---

## Why This Works

The SST node system is already a **structured, finite grammar**:

```
Node types:  T, R, S, Scope, Platform, Mirror, Radial, Instance,
             Spread, Field, Connect, Collapse
```

Each node has a small, well-defined parameter set. An LLM doesn't need to understand 3D math — it needs to understand the **vocabulary** and **composition rules**, then map intent to structure. The Walker handles all the actual math.

This is essentially **code generation** where the target language is SST JSON.

---

## Architecture

```
                                 ┌──────────────┐
  "make a fence with            │              │
   10 posts mirrored            │     LLM      │   Reads schema once,
   across X"                    │  (any model)  │   outputs SST JSON
         │                      │              │
         ▼                      └──────┬───────┘
  ┌──────────────┐                     │
  │  Prompt +    │                     ▼
  │  Schema      │──────────►  ┌──────────────┐
  │  (system     │             │  SST JSON    │
  │   context)   │             │  Output      │
  └──────────────┘             └──────┬───────┘
                                      │
                                      ▼ validate
                               ┌──────────────┐
                               │  SST Walker  │   Already built
                               │  (execute)   │
                               └──────┬───────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │  Geometry    │   Emit buffer →
                               │  Buffer      │   platform rendering
                               └──────────────┘
```

---

## The Schema Document

The LLM needs a **system prompt schema** — a concise description of every node type, its parameters, and composition rules. This is the translation dictionary.

### Schema Format (YAML for readability, served as JSON)

```yaml
# sst_schema.yaml — LLM translation reference
# This file teaches an LLM how to write SST trees.

meta:
  name: "SST Node Grammar"
  version: "1.0"
  description: >
    Structured Symmetry Tree — a JSON format for procedural 3D geometry.
    Each node produces a matrix transform or geometry emission.
    Nodes compose as a tree: parent transforms apply to all children.

output_format: "JSON"
wrapper: '{ "root": [ ...nodes... ] }'

# ─────────────────────────────────────────────
# Node Catalog
# ─────────────────────────────────────────────

nodes:

  # --- Transform Nodes (move/rotate/scale the cursor) ---

  Translate:
    short: "T"
    params:
      v: [x, y, z]           # units: centimeters
    example: { "type": "T", "v": [100, 0, 0] }
    plain: "Move 100cm along X"

  Rotate:
    short: "R"
    params:
      axis: "X" | "Y" | "Z"
      deg: number             # degrees
    example: { "type": "R", "axis": "Z", "deg": 45 }
    plain: "Rotate 45 degrees around Z"

  Scale:
    short: "S"
    params:
      v: [sx, sy, sz]        # multipliers (1 = no change)
    example: { "type": "S", "v": [2, 1, 1] }
    plain: "Stretch to 2x width"

  # --- Structure Nodes ---

  Scope:
    params: none
    effect: "Push/pop — children don't affect siblings"
    example: { "type": "Scope", "children": [...] }
    plain: "Isolate these transforms"

  Platform:
    params:
      id: string              # name for this origin point
    example: { "type": "Platform", "id": "P1" }
    plain: "Mark current position as a named origin"

  # --- Symmetry / Flow Nodes ---

  Mirror:
    params:
      axis: "X" | "Y" | "Z" | "XY" | "XZ" | "YZ"
    requires: "Platform node must exist (defines mirror plane)"
    effect: "Runs children twice: once normal, once reflected"
    example: { "type": "Mirror", "axis": "X", "children": [...] }
    plain: "Mirror everything inside across the X axis"

  Radial:
    params:
      n: integer              # number of copies
      axis: "X" | "Y" | "Z"  # rotation axis (default Z)
    effect: "Runs children n times, each rotated by 360/n degrees"
    example: { "type": "Radial", "n": 6, "axis": "Z", "children": [...] }
    plain: "Repeat in a circle, 6 copies around Z"

  # --- Emission (produce actual geometry) ---

  Instance:
    params:
      mesh: string            # mesh asset path or name
    effect: "Emit geometry at current cursor position"
    example: { "type": "Instance", "mesh": "Cube" }
    plain: "Place a cube here"

  # --- Array / Spread Nodes (Phase 6) ---

  Spread:
    params:
      source: "Linear" | "Radial" | "Grid"
      params:
        # Linear:
        start: [x, y, z]
        end: [x, y, z]
        n: integer
        # Radial:
        radius: number
        n: integer
        axis: "X" | "Y" | "Z"
        # Grid:
        dims: [nx, ny]
        spacing: [sx, sy, sz]
    effect: "Generate array of positions, run children at each"
    example:
      linear: |
        { "type": "Spread", "source": "Linear",
          "params": { "start": [0,0,0], "end": [500,0,0], "n": 10 },
          "children": [...] }
      grid: |
        { "type": "Spread", "source": "Grid",
          "params": { "dims": [5, 5], "spacing": [100, 100, 0] },
          "children": [...] }
    plain: "Distribute children in a line / circle / grid"

  Collapse:
    params:
      collapseType: "INSTANCES" | "TUBE" | "POLYLINE" | "POINTS"
      mesh: string            # for INSTANCES
      radius: number          # for TUBE
      segments: integer       # for TUBE
    requires: "Must be inside a Spread"
    effect: "Convert the spread positions into geometry"
    example: |
      { "type": "Collapse", "collapseType": "INSTANCES", "mesh": "Post" }
    plain: "Turn the array into placed objects (or a tube/line)"

  Field:
    params:
      fieldType: "Noise" | "Attractor" | "Repel" | "Vortex"
      operation: "ADVECT" | "ALIGN" | "SCALE"
      intensity: number
      scale: number           # for Noise
      target: [x, y, z]      # for Attractor/Repel/Vortex
    effect: "Warp positions based on a spatial field"
    example: |
      { "type": "Field", "fieldType": "Noise",
        "operation": "ADVECT", "intensity": 50, "scale": 0.005 }
    plain: "Add organic randomness to positions"

  Connect:
    params:
      strategy: "SEQUENTIAL" | "LOOP" | "GRID"
    effect: "Declare how spread elements connect (for tubes/polylines)"
    example: { "type": "Connect", "strategy": "LOOP" }
    plain: "Connect the array elements end-to-end (or in a loop)"

# ─────────────────────────────────────────────
# Composition Rules
# ─────────────────────────────────────────────

rules:
  - "Nodes nest via 'children' arrays"
  - "Transforms accumulate: parent T + child T = combined offset"
  - "Scope isolates: transforms inside don't leak out"
  - "Mirror needs a Platform sibling defined first"
  - "Spread must contain a Collapse to produce output"
  - "Instance is a leaf — it emits geometry, no children"
  - "Default units: centimeters, degrees"

# ─────────────────────────────────────────────
# Common Patterns
# ─────────────────────────────────────────────

patterns:

  fence_row:
    description: "Line of evenly spaced objects"
    template: |
      { "root": [
        { "type": "Spread", "source": "Linear",
          "params": { "start": [0,0,0], "end": [500,0,0], "n": 6 },
          "children": [
            { "type": "Collapse", "collapseType": "INSTANCES", "mesh": "FencePost" }
          ]}
      ]}

  mirrored_arm:
    description: "Object mirrored across X axis"
    template: |
      { "root": [
        { "type": "Platform", "id": "Center" },
        { "type": "Mirror", "axis": "X", "children": [
          { "type": "T", "v": [80, 0, 0] },
          { "type": "Instance", "mesh": "Arm" }
        ]}
      ]}

  radial_pillars:
    description: "Pillars arranged in a circle"
    template: |
      { "root": [
        { "type": "Radial", "n": 8, "axis": "Z", "children": [
          { "type": "T", "v": [200, 0, 0] },
          { "type": "Instance", "mesh": "Pillar" }
        ]}
      ]}

  organic_scatter:
    description: "Grid of objects with noise displacement"
    template: |
      { "root": [
        { "type": "Spread", "source": "Grid",
          "params": { "dims": [5, 5], "spacing": [120, 120, 0] },
          "children": [
            { "type": "Field", "fieldType": "Noise",
              "operation": "ADVECT", "intensity": 30, "scale": 0.01 },
            { "type": "Collapse", "collapseType": "INSTANCES", "mesh": "Rock" }
          ]}
      ]}

  tube_along_path:
    description: "Tube/pipe following a curved path"
    template: |
      { "root": [
        { "type": "Spread", "source": "Linear",
          "params": { "start": [0,0,0], "end": [400,0,200], "n": 20 },
          "children": [
            { "type": "Field", "fieldType": "Noise",
              "operation": "ADVECT", "intensity": 40, "scale": 0.008 },
            { "type": "Connect", "strategy": "SEQUENTIAL" },
            { "type": "Collapse", "collapseType": "TUBE",
              "radius": 5, "segments": 8 }
          ]}
      ]}

  symmetric_building:
    description: "Building facade with mirrored and radial elements"
    template: |
      { "root": [
        { "type": "Platform", "id": "Base" },
        { "type": "Mirror", "axis": "X", "children": [
          { "type": "Spread", "source": "Linear",
            "params": { "start": [50,0,0], "end": [50,0,400], "n": 5 },
            "children": [
              { "type": "Collapse", "collapseType": "INSTANCES", "mesh": "Window" }
            ]}
        ]}
      ]}
```

---

## Prompt Engineering

### System Prompt Template

The LLM receives the schema above as system context, plus this instruction:

```
You are a 3D geometry generator. You translate natural language descriptions
into SST JSON trees. The SST (Structured Symmetry Tree) is a node-based
format where each node is a transform, symmetry operation, or geometry
emission.

RULES:
- Output ONLY valid JSON matching the SST format. No explanation.
- Wrap output in { "root": [ ... ] }
- Use the node types and parameters exactly as defined in the schema.
- When the user says "mirror", use a Platform + Mirror node pair.
- When the user says "array", "line of", "row of", use Spread + Collapse.
- When the user says "circle of", "ring of", use Radial or Spread(Radial).
- When the user says "organic" or "natural", add a Field(Noise) node.
- Default mesh names to simple descriptors: "Cube", "Sphere", "Cylinder",
  "Post", "Beam", "Pillar", etc. unless the user specifies otherwise.
- Think in centimeters. A person is ~170cm tall. A door is ~200cm x 90cm.
- Compose patterns: you can nest Mirror inside Spread, Spread inside Mirror,
  Radial containing Spread, etc.

EXAMPLES:
[include 3-5 patterns from the schema]
```

### User Prompt Examples

| User says | LLM should produce |
|-----------|-------------------|
| "A row of 8 fence posts" | Spread(Linear, n=8) → Collapse(INSTANCES, "Post") |
| "A ring of 12 columns" | Radial(12, Z) → T(300,0,0) → Instance("Column") |
| "Mirror a chair across X" | Platform + Mirror(X) → T + Instance("Chair") |
| "5x5 grid of trees with some randomness" | Spread(Grid, 5x5) → Field(Noise) → Collapse(INSTANCES, "Tree") |
| "A curving pipe from A to B" | Spread(Linear) → Field(Noise) → Connect → Collapse(TUBE) |

---

## Image-to-SST Flow

When the user provides an image instead of text:

```
  ┌──────────┐     ┌──────────────┐     ┌──────────┐
  │  Image   │────►│  Vision LLM  │────►│  SST     │
  │  (photo, │     │  (describe   │     │  JSON    │
  │  sketch, │     │   structure) │     │          │
  │  render) │     └──────────────┘     └────┬─────┘
  └──────────┘                               │
                                             ▼
                                       ┌──────────┐
                                       │  Walker  │
                                       │  Execute │
                                       └──────────┘
```

**Two-stage approach:**

1. **Stage 1 — Describe:** The vision model sees the image and outputs a structural description:
   > "I see a symmetrical facade with 4 floors, each floor has 5 evenly spaced windows, mirrored across the center axis. Ground floor has a larger arched entrance."

2. **Stage 2 — Translate:** The same (or second) LLM translates that description into SST JSON using the schema.

This two-stage approach is more reliable than directly outputting JSON from an image because it separates perception from generation.

**Single-stage alternative:** A capable multimodal model (GPT-4o, Claude with vision) can do both in one pass with the right system prompt:

```
Look at this image. Identify the structural patterns:
- Repetition (arrays, rows, grids)
- Symmetry (mirrors, radial)
- Hierarchy (groups of groups)
- Organic variation (randomness, noise)

Then output an SST JSON tree that would approximate this structure.
```

---

## Validation Layer

The LLM will sometimes produce invalid JSON or use wrong parameter names. A thin validation layer catches this before the Walker:

```python
def validate_sst(tree: dict) -> tuple[bool, list[str]]:
    """Validate an SST JSON tree. Returns (valid, errors)."""
    errors = []

    if "root" not in tree:
        errors.append("Missing 'root' array")
        return False, errors

    VALID_TYPES = {
        "T", "Translate", "R", "Rotate", "S", "Scale",
        "Scope", "Platform", "Mirror", "Radial", "Instance",
        "Spread", "Field", "Connect", "Collapse"
    }

    def check_node(node, path="root"):
        if "type" not in node:
            errors.append(f"{path}: missing 'type'")
            return
        if node["type"] not in VALID_TYPES:
            errors.append(f"{path}: unknown type '{node['type']}'")

        # Type-specific checks
        t = node["type"]
        if t in ("T", "Translate") and "v" not in node:
            errors.append(f"{path}: Translate needs 'v' array")
        if t in ("R", "Rotate") and ("axis" not in node or "deg" not in node):
            errors.append(f"{path}: Rotate needs 'axis' and 'deg'")
        if t == "Instance" and "mesh" not in node:
            errors.append(f"{path}: Instance needs 'mesh'")
        if t == "Mirror" and "axis" not in node:
            errors.append(f"{path}: Mirror needs 'axis'")
        if t == "Radial" and "n" not in node:
            errors.append(f"{path}: Radial needs 'n'")

        for i, child in enumerate(node.get("children", [])):
            check_node(child, f"{path}[{i}]")

    for i, node in enumerate(tree["root"]):
        check_node(node, f"root[{i}]")

    return len(errors) == 0, errors
```

### Auto-Repair

For common LLM mistakes, attempt auto-repair before rejecting:

| Mistake | Fix |
|---------|-----|
| `"type": "Translate"` | Accept (alias for "T") |
| Missing `"v"` on Translate | Default to `[0, 0, 0]` |
| `"axis": "x"` (lowercase) | Uppercase it |
| `"n": "8"` (string) | Cast to int |
| Mirror without Platform | Insert Platform("auto") before it |
| Collapse outside Spread | Warn, skip node |
| Extra fields | Ignore them |

---

## Feedback Loop (Refinement)

The user sees the generated geometry and can refine:

```
User:  "Make the spacing wider"
       "Add more copies"
       "Rotate the whole thing 45 degrees"
       "Make it more organic"
```

The LLM receives the **previous SST JSON** as context plus the refinement instruction, and outputs a modified tree. This is simple because the SST is small, structured, and the LLM can read it back.

```
System: Here is the current SST tree: { ... }
User:   "Double the number of columns and make them taller"
LLM:    [outputs modified SST JSON with n doubled and scale increased]
```

---

## Minimal Implementation Plan

### What exists already
- SST Walker + all node types (Python: `sst_nodes.py`)
- JSON parser (`parse_tree`, `execute_tree`)
- Full node algebra with tests (20 tests passing)

### What to build

**Step 1 — Schema file** (30 min)
- Convert the YAML schema above into `sst_schema.json`
- This is the LLM's reference document

**Step 2 — Validation function** (30 min)
- `validate_sst()` as shown above
- Add to `implementations/universal/sst_validator.py`

**Step 3 — Prompt wrapper** (minimal code)
- Function that assembles: system prompt (schema + rules) + user message
- Calls any LLM API (OpenAI, Anthropic, local model)
- Parses JSON response
- Validates → executes → returns geometry buffer

```python
def prompt_to_geometry(user_prompt: str,
                       model: str = "claude-sonnet",
                       image: bytes = None) -> list[dict]:
    """
    Natural language → SST JSON → geometry buffer.

    This is the entire pipeline in one function.
    """
    schema = load_schema("sst_schema.json")
    system = build_system_prompt(schema)

    response = call_llm(system, user_prompt, image=image, model=model)

    tree = json.loads(response)

    valid, errors = validate_sst(tree)
    if not valid:
        tree = auto_repair(tree, errors)

    return execute_tree(json.dumps(tree))
```

**Step 4 — CLI interface** (optional, for testing)
```bash
python sst_prompt.py "a ring of 6 pillars around a central platform"
# → outputs SST JSON + geometry buffer summary
```

---

## What Makes This Different from Generic LLM Code-Gen

1. **Tiny target language.** SST has ~12 node types. The LLM doesn't need to write arbitrary code — just compose a small vocabulary.

2. **Deterministic execution.** The Walker is pure math. Same JSON → same geometry, every time. No runtime errors, no dependencies.

3. **Compositional patterns.** The few-shot examples in the schema teach the LLM reusable patterns (mirror + spread, radial + translate) that combine predictably.

4. **Refinement-friendly.** The SST JSON is short enough to fit in context alongside a refinement prompt. The LLM can read its own output and modify it.

5. **Image grounding.** A vision model can identify structural patterns (repetition, symmetry, hierarchy) that map directly to SST node types. It doesn't need to understand mesh topology — just spatial composition.

---

## Dormant Integration Point

This system is designed to be **dormant until activated**. The existing codebase doesn't depend on any LLM. The bridge is:

- **sst_schema.json** — static file, no runtime cost
- **sst_validator.py** — pure Python, no external deps
- **prompt_to_geometry()** — only called when user invokes LLM mode

The LLM call is the only part that requires an API key or network access. Everything else is local, deterministic, and already tested.

```
┌─────────────────────────────────────────────────┐
│              Existing System                     │
│                                                  │
│  math_core.py ──► sst_nodes.py ──► Walker       │
│       ▲               ▲                          │
│       │               │                          │
│       │        parse_tree(json)                   │
│       │               ▲                          │
│       │               │ ◄── dormant bridge ──┐   │
│       │               │                      │   │
│       │       validate_sst()                 │   │
│       │               ▲                      │   │
│       │               │                      │   │
│       │        LLM response (JSON)           │   │
│       │               ▲                      │   │
│       │               │                      │   │
│       │     ┌─────────┴──────────┐           │   │
│       │     │  sst_schema.json   │           │   │
│       │     │  (system prompt)   │           │   │
│       │     └────────────────────┘           │   │
│       │                                      │   │
└───────┼──────────────────────────────────────┘   │
        │                                          │
        │         Only active when user             │
        │         invokes LLM mode ─────────────────┘
```

---

## Reference

- **Node types:** `core/node_algebra.md`
- **Walker math:** `core/math_foundations.md`
- **Phase 6 extensions:** `core/extended_state_algebra.md`
- **Python implementation:** `implementations/universal/sst_nodes.py`
- **Existing ML vision (theoretical):** `core/ml_integration.md`
