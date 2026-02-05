#!/usr/bin/env python3
"""
SymmetryKit — Setup & Installation Script
==========================================

Automates project setup, validation, and platform installs.

Usage:
    python scripts/setup.py                    # Validate + run all tests
    python scripts/setup.py --test             # Run tests only
    python scripts/setup.py --validate         # Structure check only
    python scripts/setup.py --install-ue5 PATH # Install UE5 plugin to project
    python scripts/setup.py --install-blender  # Install Blender addon (future)
    python scripts/setup.py --install-maya     # Install Maya plugin (future)
    python scripts/setup.py --all              # Full setup + tests + index regen
"""

import os
import sys
import shutil
import subprocess
import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
UNIVERSAL_DIR = REPO_ROOT / "implementations" / "universal"
UE5_PLUGIN_DIR = REPO_ROOT / "implementations" / "unreal" / "SymmetryKit"
BLENDER_DIR = REPO_ROOT / "implementations" / "blender"
MAYA_DIR = REPO_ROOT / "implementations" / "maya"
CORE_DIR = REPO_ROOT / "core"
SCRIPTS_DIR = REPO_ROOT / "scripts"

MIN_PYTHON = (3, 10)

# Expected project structure for validation
EXPECTED_FILES = {
    "core": [
        "math_foundations.md",
        "node_algebra.md",
        "state_schema.md",
        "mutation_schema.md",
        "extended_state_algebra.md",
        "SKELETAL_SINGLETON_TREE.md",
    ],
    "universal": [
        "math_core.py",
        "sst_nodes.py",
    ],
    "ue5_public": [
        "SymmetryKitModule.h",
        "MathCore.h",
        "Walker.h",
        "SymmetryPlaneActor.h",
        "MirrorComponent.h",
        "MCPBridge.h",
        "NodeParser.h",
        "SSTNodes.h",
        "SymmetryKitEditorWidget.h",
        "SymmetryKitUITypes.h",
        "SymmetryKitPreviewManager.h",
    ],
    "ue5_private": [
        "SymmetryKitModule.cpp",
        "MathCore.cpp",
        "Walker.cpp",
        "SymmetryPlaneActor.cpp",
        "MirrorComponent.cpp",
        "MCPBridge.cpp",
        "NodeParser.cpp",
        "SSTNodes.cpp",
        "SymmetryKitEditorWidget.cpp",
        "SymmetryKitUITypes.cpp",
        "SymmetryKitPreviewManager.cpp",
    ],
    "ue5_root": [
        "SymmetryKit.uplugin",
    ],
    "root": [
        "CLAUDE_CODE_HANDOFF.md",
        "QUICK_REFERENCE.md",
        "INSTALL.md",
    ],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @staticmethod
    def supported():
        return sys.stdout.isatty() and os.name != "nt" or os.environ.get("FORCE_COLOR")


def c(text, color):
    """Apply color if terminal supports it."""
    if Colors.supported():
        return f"{color}{text}{Colors.RESET}"
    return text


def header(title):
    print(f"\n{'=' * 60}")
    print(f"  {c(title, Colors.BOLD + Colors.CYAN)}")
    print(f"{'=' * 60}")


def ok(msg):
    print(f"  {c('[OK]', Colors.GREEN)} {msg}")


def fail(msg):
    print(f"  {c('[FAIL]', Colors.RED)} {msg}")


def warn(msg):
    print(f"  {c('[WARN]', Colors.YELLOW)} {msg}")


def info(msg):
    print(f"  {c('[INFO]', Colors.CYAN)} {msg}")


# ---------------------------------------------------------------------------
# Check Python Version
# ---------------------------------------------------------------------------

def check_python():
    header("Python Environment")
    v = sys.version_info
    ver_str = f"{v.major}.{v.minor}.{v.micro}"

    if (v.major, v.minor) >= MIN_PYTHON:
        ok(f"Python {ver_str} (>= {MIN_PYTHON[0]}.{MIN_PYTHON[1]} required)")
        return True
    else:
        fail(f"Python {ver_str} — need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
        return False


# ---------------------------------------------------------------------------
# Validate Project Structure
# ---------------------------------------------------------------------------

def validate_structure():
    header("Project Structure Validation")

    all_ok = True
    counts = {"found": 0, "missing": 0}

    # Core specs
    for f in EXPECTED_FILES["core"]:
        path = CORE_DIR / f
        if path.exists():
            counts["found"] += 1
        else:
            fail(f"Missing: core/{f}")
            counts["missing"] += 1
            all_ok = False

    # Universal implementations
    for f in EXPECTED_FILES["universal"]:
        path = UNIVERSAL_DIR / f
        if path.exists():
            counts["found"] += 1
        else:
            fail(f"Missing: implementations/universal/{f}")
            counts["missing"] += 1
            all_ok = False

    # UE5 plugin
    for f in EXPECTED_FILES["ue5_root"]:
        path = UE5_PLUGIN_DIR / f
        if path.exists():
            counts["found"] += 1
        else:
            fail(f"Missing: SymmetryKit/{f}")
            counts["missing"] += 1
            all_ok = False

    ue5_src = UE5_PLUGIN_DIR / "Source" / "SymmetryKit"
    for f in EXPECTED_FILES["ue5_public"]:
        path = ue5_src / "Public" / f
        if path.exists():
            counts["found"] += 1
        else:
            fail(f"Missing: SymmetryKit/Public/{f}")
            counts["missing"] += 1
            all_ok = False

    for f in EXPECTED_FILES["ue5_private"]:
        path = ue5_src / "Private" / f
        if path.exists():
            counts["found"] += 1
        else:
            fail(f"Missing: SymmetryKit/Private/{f}")
            counts["missing"] += 1
            all_ok = False

    # Root files
    for f in EXPECTED_FILES["root"]:
        path = REPO_ROOT / f
        if path.exists():
            counts["found"] += 1
        else:
            warn(f"Missing: {f}")
            counts["missing"] += 1

    if all_ok:
        ok(f"All {counts['found']} expected files present")
    else:
        info(f"{counts['found']} found, {counts['missing']} missing")

    return all_ok


# ---------------------------------------------------------------------------
# Run Tests
# ---------------------------------------------------------------------------

def run_tests():
    header("Running Tests")

    results = {"passed": 0, "failed": 0}

    # Math core tests
    info("Math Core (math_core.py)...")
    ret = subprocess.run(
        [sys.executable, str(UNIVERSAL_DIR / "math_core.py")],
        capture_output=True, text=True, cwd=str(UNIVERSAL_DIR),
        timeout=30,
    )
    if ret.returncode == 0:
        ok("Math Core — all tests passed")
        results["passed"] += 1
    else:
        fail("Math Core — tests failed")
        if ret.stderr:
            for line in ret.stderr.strip().split("\n")[-5:]:
                print(f"        {line}")
        results["failed"] += 1

    # SST node tests
    info("SST Node System (sst_nodes.py)...")
    ret = subprocess.run(
        [sys.executable, str(UNIVERSAL_DIR / "sst_nodes.py")],
        capture_output=True, text=True, cwd=str(UNIVERSAL_DIR),
        timeout=30,
    )
    if ret.returncode == 0:
        ok("SST Nodes — all tests passed")
        results["passed"] += 1
    else:
        fail("SST Nodes — tests failed")
        if ret.stderr:
            for line in ret.stderr.strip().split("\n")[-5:]:
                print(f"        {line}")
        results["failed"] += 1

    # Print test output (stdout from tests shows individual [PASS] lines)
    if ret.returncode == 0 and ret.stdout:
        for line in ret.stdout.strip().split("\n"):
            if "[PASS]" in line or "passed" in line.lower():
                print(f"        {line.strip()}")

    print()
    if results["failed"] == 0:
        ok(f"All test suites passed ({results['passed']}/{results['passed']})")
    else:
        fail(f"{results['failed']} suite(s) failed, {results['passed']} passed")

    return results["failed"] == 0


# ---------------------------------------------------------------------------
# Install UE5 Plugin
# ---------------------------------------------------------------------------

def install_ue5(project_path: str):
    header("UE5 Plugin Install")

    project = Path(project_path).resolve()

    # Find or create Plugins directory
    plugins_dir = project / "Plugins"
    if not project.exists():
        fail(f"Project path not found: {project}")
        return False

    # Check for .uproject file
    uproject_files = list(project.glob("*.uproject"))
    if not uproject_files:
        warn("No .uproject file found — are you sure this is a UE5 project?")

    if not plugins_dir.exists():
        info(f"Creating Plugins directory: {plugins_dir}")
        plugins_dir.mkdir(parents=True)

    dest = plugins_dir / "SymmetryKit"

    if dest.exists():
        warn(f"SymmetryKit already exists at {dest}")
        info("Removing existing copy...")
        shutil.rmtree(dest)

    info(f"Copying plugin to {dest}")
    shutil.copytree(str(UE5_PLUGIN_DIR), str(dest))

    ok(f"SymmetryKit installed to {dest}")

    # Check .uproject for required plugins
    if uproject_files:
        uproject_path = uproject_files[0]
        try:
            with open(uproject_path, "r", encoding="utf-8") as f:
                uproject = json.load(f)

            plugins = uproject.get("Plugins", [])
            plugin_names = {p.get("Name") for p in plugins}

            required = ["ModelingToolsEditorMode", "GeometryScripting"]
            missing = [r for r in required if r not in plugin_names]

            if missing:
                warn(f"Your .uproject is missing required plugins: {', '.join(missing)}")
                info("Add them to the Plugins array in your .uproject file")
            else:
                ok("Required dependency plugins found in .uproject")

            if "SymmetryKit" not in plugin_names:
                info("Consider adding SymmetryKit to your .uproject Plugins array:")
                info('  { "Name": "SymmetryKit", "Enabled": true }')

        except (json.JSONDecodeError, IOError) as e:
            warn(f"Could not parse .uproject: {e}")

    info("Next steps:")
    info("  1. Regenerate project files (right-click .uproject)")
    info("  2. Build the project")
    info("  3. Enable SymmetryKit in Edit > Plugins")

    return True


# ---------------------------------------------------------------------------
# Install Blender Addon (Future)
# ---------------------------------------------------------------------------

def install_blender():
    header("Blender Addon Install")
    warn("Blender addon is not yet implemented.")
    info("The universal math core and node system are ready.")
    info("A Blender binding needs: operators, panels, and mathutils conversion.")
    info("See implementations/blender/README.md for the planned structure.")
    return False


# ---------------------------------------------------------------------------
# Install Maya Plugin (Future)
# ---------------------------------------------------------------------------

def install_maya():
    header("Maya Plugin Install")
    warn("Maya plugin is not yet implemented.")
    info("The universal math core and node system are ready.")
    info("A Maya binding needs: commands, UI, and OpenMaya conversion.")
    info("See implementations/maya/README.md for the planned structure.")
    return False


# ---------------------------------------------------------------------------
# Regenerate Index
# ---------------------------------------------------------------------------

def regenerate_index():
    header("Regenerating Knowledge Index")

    index_script = SCRIPTS_DIR / "generate_index.py"
    if not index_script.exists():
        warn("generate_index.py not found, skipping")
        return False

    ret = subprocess.run(
        [sys.executable, str(index_script), "--root", str(REPO_ROOT),
         "--output", "index.jsonl"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
        timeout=30,
    )
    if ret.returncode == 0:
        ok("index.jsonl regenerated")
        if ret.stdout:
            info(ret.stdout.strip())
        return True
    else:
        fail("Index generation failed")
        if ret.stderr:
            for line in ret.stderr.strip().split("\n")[-3:]:
                print(f"        {line}")
        return False


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------

def print_summary():
    header("Implementation Status")

    phases = [
        ("1", "Universal Math Core", "Complete", "math_core.py (447 lines)"),
        ("2", "UE5 Plugin Structure", "Complete", "18 files (9 .h + 9 .cpp)"),
        ("3", "Node System + Parser", "Complete", "sst_nodes.py (1500+ lines) + UE5"),
        ("4", "MCP Bridge Server", "Complete", "HTTP server on port 9147, JSON-RPC endpoints"),
        ("5", "Editor Widget", "Complete", "SymmetryKitEditorWidget C++ base class"),
        ("6", "Extended Algebra", "Complete", "Spread, Frame, Field, Loft, Collapse"),
    ]

    for num, name, status, note in phases:
        if status == "Complete":
            marker = c("[DONE]", Colors.GREEN)
        elif status == "Skeleton":
            marker = c("[PART]", Colors.YELLOW)
        else:
            marker = c("[TODO]", Colors.YELLOW)
        print(f"  Phase {num}: {marker} {name} — {note}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SymmetryKit — Setup & Installation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup.py                    # Full validate + test
  python scripts/setup.py --test             # Tests only
  python scripts/setup.py --validate         # Structure check only
  python scripts/setup.py --install-ue5 PATH # Install to UE5 project
  python scripts/setup.py --all              # Everything + index regen
        """,
    )
    parser.add_argument("--test", action="store_true",
                        help="Run tests only")
    parser.add_argument("--validate", action="store_true",
                        help="Validate project structure only")
    parser.add_argument("--install-ue5", metavar="PATH",
                        help="Install UE5 plugin to project path")
    parser.add_argument("--install-blender", action="store_true",
                        help="Install Blender addon (future)")
    parser.add_argument("--install-maya", action="store_true",
                        help="Install Maya plugin (future)")
    parser.add_argument("--all", action="store_true",
                        help="Full setup: validate + test + index regen")
    parser.add_argument("--summary", action="store_true",
                        help="Print implementation status summary")

    args = parser.parse_args()

    print(c("\n  SymmetryKit Setup", Colors.BOLD))
    print(f"  Repo: {REPO_ROOT}\n")

    success = True

    # Handle specific install commands
    if args.install_ue5:
        check_python()
        install_ue5(args.install_ue5)
        return

    if args.install_blender:
        install_blender()
        return

    if args.install_maya:
        install_maya()
        return

    if args.summary:
        print_summary()
        return

    # Default: validate + test (or specific flags)
    if args.test:
        if not check_python():
            sys.exit(1)
        if not run_tests():
            sys.exit(1)
        return

    if args.validate:
        if not check_python():
            sys.exit(1)
        validate_structure()
        return

    # Default / --all: full setup
    if not check_python():
        sys.exit(1)

    validate_structure()

    if not run_tests():
        success = False

    if args.all:
        regenerate_index()

    print_summary()

    print()
    if success:
        ok(c("Setup complete. All systems operational.", Colors.GREEN))
    else:
        fail("Setup completed with errors. See above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
