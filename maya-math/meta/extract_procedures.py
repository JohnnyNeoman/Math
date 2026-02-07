#!/usr/bin/env python3
"""
Extract all procedure declarations from Maya MEL scripts.
Generates a comprehensive catalog for organization.
"""

import os
import re
from pathlib import Path
from collections import defaultdict

MAYA_DIR = Path(r"C:\Users\John\Downloads\MCP_Folder\3D_tools_ML_hybrid\MAYA")

def extract_procs_from_file(filepath):
    """Extract procedure declarations from a MEL file."""
    procs = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')

            # Pattern for MEL proc declarations
            # global proc returnType procName(params) or proc returnType procName(params)
            proc_pattern = re.compile(
                r'^(?:\s*)(?:global\s+)?proc\s+(?:(string|int|float|vector|matrix|\[\]|string\s*\[\]|int\s*\[\]|float\s*\[\]|vector\s*\[\])\s+)?(\w+)\s*\(([^)]*)\)',
                re.MULTILINE
            )

            for i, line in enumerate(lines, 1):
                match = proc_pattern.match(line)
                if match:
                    return_type = match.group(1) or 'void'
                    proc_name = match.group(2)
                    params = match.group(3).strip()
                    procs.append({
                        'name': proc_name,
                        'return_type': return_type,
                        'params': params,
                        'line': i,
                        'file': filepath.name
                    })

    except Exception as e:
        print(f"Error reading {filepath}: {e}")

    return procs

def categorize_proc(name, params):
    """Categorize procedure by name pattern."""
    name_lower = name.lower()

    if 'circle' in name_lower or 'circ' in name_lower:
        return 'circle'
    elif 'tangent' in name_lower or 'tan' in name_lower and 'atan' not in name_lower:
        return 'tangent'
    elif 'cam' in name_lower or 'camera' in name_lower or 'plane' in name_lower or 'project' in name_lower:
        return 'sketch'
    elif 'matrix' in name_lower or 'rotation' in name_lower or 'rotate' in name_lower or 'transform' in name_lower:
        return 'matrix'
    elif 'array' in name_lower or 'string' in name_lower or 'float' in name_lower or 'vec' in name_lower or 'index' in name_lower:
        return 'array'
    elif 'poly' in name_lower or 'face' in name_lower or 'edge' in name_lower or 'vert' in name_lower:
        return 'polygon'
    elif 'curve' in name_lower or 'nurb' in name_lower:
        return 'curve'
    else:
        return 'utility'

def determine_sst_layer(category):
    """Map category to SST mathematical layer."""
    if category in ['matrix', 'utility']:
        return 'affine'
    elif category in ['circle', 'tangent', 'curve']:
        return 'conformal'
    elif category in ['sketch']:
        return 'conformal/spectral'
    elif category in ['polygon']:
        return 'affine'
    else:
        return 'affine'

def main():
    all_procs = []
    files_processed = 0

    # Walk through MAYA directory
    for root, dirs, files in os.walk(MAYA_DIR):
        # Skip organized/legacy/meta folders
        if any(skip in root for skip in ['organized', 'legacy', 'meta', '__pycache__']):
            continue

        for filename in files:
            if filename.endswith(('.mel', '.txt')):
                filepath = Path(root) / filename
                procs = extract_procs_from_file(filepath)
                if procs:
                    files_processed += 1
                    for proc in procs:
                        proc['source_path'] = str(filepath.relative_to(MAYA_DIR))
                        proc['category'] = categorize_proc(proc['name'], proc['params'])
                        proc['sst_layer'] = determine_sst_layer(proc['category'])
                        all_procs.append(proc)

    # Sort by category, then name
    all_procs.sort(key=lambda x: (x['category'], x['name'].lower()))

    # Group by category
    by_category = defaultdict(list)
    for proc in all_procs:
        by_category[proc['category']].append(proc)

    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCEDURE EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Files processed: {files_processed}")
    print(f"Total procedures found: {len(all_procs)}")
    print(f"\nBy Category:")
    for cat, procs in sorted(by_category.items()):
        print(f"  {cat}: {len(procs)}")

    # Find duplicates
    name_counts = defaultdict(list)
    for proc in all_procs:
        name_counts[proc['name']].append(proc)

    duplicates = {name: procs for name, procs in name_counts.items() if len(procs) > 1}
    if duplicates:
        print(f"\nDuplicate procedure names: {len(duplicates)}")
        for name, procs in list(duplicates.items())[:10]:
            files = [p['source_path'] for p in procs]
            print(f"  {name}: {len(procs)} occurrences")

    return all_procs, by_category, duplicates

if __name__ == "__main__":
    all_procs, by_category, duplicates = main()

    # Store for use by catalog generator
    import json
    output_file = MAYA_DIR / 'procedure_data.json'
    with open(output_file, 'w') as f:
        json.dump({
            'procedures': all_procs,
            'by_category': {k: v for k, v in by_category.items()},
            'duplicates': {k: v for k, v in duplicates.items()}
        }, f, indent=2)
    print(f"\nData saved to: {output_file}")
