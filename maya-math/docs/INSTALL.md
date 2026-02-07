# SymmetryKit — Installation & Setup

> Procedural symmetry toolkit with multi-platform targets

---

## Prerequisites

| Tool | Version | Required For |
|------|---------|-------------|
| Python | 3.10+ | Universal math core, tests, Blender/Maya bindings |
| Unreal Engine | 5.3+ (tested 5.4/5.5) | UE5 plugin |
| Git | any | Version control |
| Blender | 3.6+ / 4.0+ | Blender addon (future) |
| Maya | 2024+ | Maya plugin (future) |

Only **Python** is required to run the universal reference implementation and tests.
UE5 is only needed if you're building the Unreal plugin.

---

## Quick Start (Universal / Python)

### 1. Verify Python

```bash
python --version
# Python 3.10+ required
```

### 2. Run the Setup Script

```bash
python scripts/setup.py
```

This will:
- Verify Python version
- Run math core self-tests (10 tests)
- Run SST node system self-tests (10 tests)
- Validate project structure
- Report status

### 3. Manual Verification (Optional)

```bash
cd implementations/universal
python math_core.py       # 10 math tests
python sst_nodes.py       # 10 node system tests
```

No external dependencies are needed — the universal layer uses only the Python standard library.

---

## Unreal Engine 5 Plugin

### 1. Locate Your UE5 Project Plugins Folder

```
YourProject/
  Plugins/          <-- target directory
  Source/
  Content/
  YourProject.uproject
```

### 2. Copy or Symlink the Plugin

**Option A — Copy:**
```bash
# From the repo root:
python scripts/setup.py --install-ue5 "C:/Path/To/YourProject"
```

**Option B — Manual Copy:**
Copy the entire `implementations/unreal/SymmetryKit/` folder into your project's `Plugins/` directory:

```
YourProject/
  Plugins/
    SymmetryKit/           <-- copy here
      SymmetryKit.uplugin
      SymmetryKit.Build.cs
      Source/
        SymmetryKit/
          Public/
          Private/
      Content/
```

**Option C — Symlink (development):**
```bash
# Windows (PowerShell as Admin):
New-Item -ItemType Junction -Path "C:\Path\To\YourProject\Plugins\SymmetryKit" -Target "C:\Path\To\This\Repo\implementations\unreal\SymmetryKit"

# macOS/Linux:
ln -s /path/to/repo/implementations/unreal/SymmetryKit /path/to/project/Plugins/SymmetryKit
```

### 3. Enable Required Plugins

In your `.uproject` file, ensure these plugins are enabled:

```json
{
  "Plugins": [
    { "Name": "ModelingToolsEditorMode", "Enabled": true },
    { "Name": "GeometryScripting", "Enabled": true },
    { "Name": "SymmetryKit", "Enabled": true }
  ]
}
```

Or enable them in the Editor: **Edit > Plugins**, search for "SymmetryKit".

### 4. Regenerate Project Files

```bash
# Windows (right-click .uproject → "Generate Visual Studio project files")
# Or from command line:
"C:/Program Files/Epic Games/UE_5.4/Engine/Build/BatchFiles/Build.bat" YourProjectEditor Win64 Development "C:/Path/To/YourProject/YourProject.uproject"
```

### 5. Build & Launch

Open the `.sln` in Visual Studio / Rider and build, or launch UE5 Editor directly — it will compile the plugin on startup.

### UE5 Module Dependencies

The plugin depends on these UE5 modules (handled automatically by `Build.cs`):

| Module | Purpose |
|--------|---------|
| Core, CoreUObject, Engine | Standard UE5 |
| UnrealEd | Editor subsystem (MCPBridge) |
| GeometryCore, DynamicMesh | Procedural mesh |
| GeometryFramework, ModelingComponents | Geometry scripting |
| Http, Json, JsonUtilities | MCP bridge communication |

---

## Blender Addon (Planned)

The Blender binding is not yet implemented. When ready:

```bash
python scripts/setup.py --install-blender
```

This will copy the universal math core and node system into a Blender addon package at `implementations/blender/symmetry_kit/`.

**Manual install:** Copy the addon folder to Blender's addons directory:
- **Windows:** `%APPDATA%/Blender Foundation/Blender/4.0/scripts/addons/`
- **macOS:** `~/Library/Application Support/Blender/4.0/scripts/addons/`
- **Linux:** `~/.config/blender/4.0/scripts/addons/`

---

## Maya Plugin (Planned)

The Maya binding is not yet implemented. When ready:

```bash
python scripts/setup.py --install-maya
```

This will copy files into Maya's script path:
- **Windows:** `%USERPROFILE%/Documents/maya/scripts/`
- **macOS:** `~/Library/Preferences/Autodesk/maya/scripts/`
- **Linux:** `~/maya/scripts/`

---

## Project Structure

```
3D_tools_ML_hybrid/
├── core/                          # Specifications (source of truth)
│   ├── math_foundations.md        # Gram-Schmidt, M_reflect = P x S x P^-1
│   ├── node_algebra.md           # Functional alphabet, walker, composition
│   ├── state_schema.md           # Matrix stack semantics
│   ├── mutation_schema.md        # Geometry operations
│   ├── extended_state_algebra.md  # Phase 6: Spreads, Fields, Collapse
│   ├── SKELETAL_SINGLETON_TREE.md # Architecture: State ⊥ Mutation
│   ├── transpiler_spec.md        # Platform API mappings
│   ├── ml_integration.md         # Neural operators (future)
│   ├── rule_patterns.md          # Production rules
│   ├── plugin_bridges.md         # External tool integration
│   └── neural_topological_synthesis.md
│
├── implementations/
│   ├── universal/                 # Platform-agnostic (Python stdlib only)
│   │   ├── math_core.py          # Vec3, Mat4, MatrixStack, platform_reflect
│   │   ├── math_core.hpp         # C++ header-only (stub)
│   │   └── sst_nodes.py          # Full node system + JSON parser + tests
│   │
│   ├── unreal/                    # UE5 Plugin
│   │   └── SymmetryKit/          # Copy to YourProject/Plugins/
│   │
│   ├── blender/                   # Blender addon (planned)
│   └── maya/                      # Maya plugin (planned)
│
├── scripts/
│   ├── setup.py                   # Automated setup & install
│   └── generate_index.py         # Knowledge index generator
│
├── INSTALL.md                     # This file
├── CLAUDE_CODE_HANDOFF.md         # Implementation guide
├── QUICK_REFERENCE.md             # One-page formula cheat sheet
├── KNOWLEDGE_INDEX.md             # Human navigation index
└── index.jsonl                    # Machine-readable index
```

---

## Running Tests

```bash
# All tests (via setup script)
python scripts/setup.py --test

# Individual test suites
python implementations/universal/math_core.py     # Math core: 10 tests
python implementations/universal/sst_nodes.py      # Node system: 10 tests
```

### What the Tests Cover

**Math Core (10 tests):**
- Identity multiplication
- Translation round-trip
- Matrix inverse (A @ A^-1 = I)
- X-axis reflection
- Gram-Schmidt orthonormality
- Matrix stack push/pop
- Platform reflection formula
- Scale composition
- Rotation matrices
- Transform point/vector

**Node System (10 tests):**
- Translate + Instance emission
- Platform + Mirror (M_reflect = P x S x P^-1)
- Radial(4) replication
- Spread(Linear) + Collapse(INSTANCES)
- Spread(Grid, 3x3)
- Nested Spread (3 x 4 = 12)
- Mirror + Spread combination
- Spread + Connect + Collapse(TUBE)
- Field(Noise) displacement
- Scope isolation

---

## MCP Bridge (Phase 4)

The MCP bridge allows Claude to control SymmetryKit via JSON-RPC. Currently a skeleton — the HTTP/WebSocket server is a placeholder (`StartMCPServer`/`StopMCPServer` in `MCPBridge.cpp`).

**Endpoints defined:**

| Method | Description |
|--------|-------------|
| `symmetry.setPlane` | Create/update symmetry plane |
| `symmetry.mirrorSelection` | Mirror selected actors |
| `sst.execute` | Execute JSON SST tree |

---

## Troubleshooting

### Python tests fail with ImportError
Make sure you run from the `implementations/universal/` directory or from the repo root via `scripts/setup.py`:
```bash
cd implementations/universal && python sst_nodes.py
```

### UE5 "Missing module" errors
Ensure `ModelingToolsEditorMode` and `GeometryScripting` plugins are enabled in your project before adding SymmetryKit.

### UE5 compile errors with Http/Json
These modules are included for the MCP bridge. If you don't need MCP integration yet, you can comment out `Http`, `Json`, `JsonUtilities` from `Build.cs` and remove `MCPBridge.h/.cpp`.

### Symlink not working on Windows
Run PowerShell as Administrator. Junction links require elevated permissions on some Windows configurations.

---

## Implementation Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Universal Math Core | Complete |
| 2 | UE5 Plugin Structure | Complete |
| 3 | Node System + Parser | Complete |
| 4 | MCP Bridge Server | Skeleton |
| 5 | Editor Widget | Planned |
| 6 | Extended Algebra (Spread/Field/Collapse) | Complete (Python), UE5 ready |

---

*The math layer is universal. Only the emit() step is platform-specific.*
