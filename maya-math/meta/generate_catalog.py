#!/usr/bin/env python3
"""
Generate PROCEDURE_CATALOG.md from extracted procedure data.
"""

import json
from pathlib import Path
from collections import defaultdict

MAYA_DIR = Path(r"C:\Users\John\Downloads\MCP_Folder\3D_tools_ML_hybrid\MAYA")

def load_data():
    with open(MAYA_DIR / 'procedure_data.json', 'r') as f:
        return json.load(f)

def generate_catalog(data):
    procs = data['procedures']
    by_category = defaultdict(list)
    for proc in procs:
        by_category[proc['category']].append(proc)

    # Find unique procedures (deduplicated by name, keep first occurrence)
    seen_names = set()
    unique_procs = []
    for proc in procs:
        if proc['name'] not in seen_names:
            seen_names.add(proc['name'])
            unique_procs.append(proc)

    catalog = []
    catalog.append("# Maya MEL Procedure Catalog")
    catalog.append("")
    catalog.append("> Comprehensive catalog of MEL procedures organized for SST integration")
    catalog.append("> Generated from legacy scripts - annotated for future Python translation")
    catalog.append("")
    catalog.append("---")
    catalog.append("")
    catalog.append("## Summary")
    catalog.append("")
    catalog.append(f"- **Total Procedures**: {len(procs)}")
    catalog.append(f"- **Unique Procedures**: {len(unique_procs)}")
    catalog.append(f"- **Files Processed**: 68")
    catalog.append(f"- **Duplicate Names**: {len(procs) - len(unique_procs)}")
    catalog.append("")
    catalog.append("### By Category")
    catalog.append("")
    catalog.append("| Category | Count | SST Layer | Description |")
    catalog.append("|----------|-------|-----------|-------------|")

    category_info = {
        'circle': ('conformal', 'Circle creation, 3-point circles, packing'),
        'tangent': ('conformal', 'Point-to-circle tangents, tangent circles'),
        'sketch': ('conformal/spectral', 'CAM-based curve sketching, projection'),
        'curve': ('conformal', 'NURBS curves, arc operations'),
        'matrix': ('affine', 'Matrix operations, rotations'),
        'polygon': ('affine', 'Polygon operations, faces, edges'),
        'array': ('affine', 'Array manipulation, sorting, conversion'),
        'utility': ('affine', 'General utilities, selection, cleanup'),
    }

    for cat in ['circle', 'tangent', 'sketch', 'curve', 'matrix', 'polygon', 'array', 'utility']:
        if cat in by_category:
            layer, desc = category_info.get(cat, ('affine', ''))
            catalog.append(f"| **{cat}** | {len(by_category[cat])} | {layer} | {desc} |")

    catalog.append("")
    catalog.append("---")
    catalog.append("")

    # Detailed sections by category
    for cat in ['circle', 'tangent', 'sketch', 'curve', 'matrix', 'array', 'polygon', 'utility']:
        if cat not in by_category:
            continue

        layer, desc = category_info.get(cat, ('affine', ''))
        catalog.append(f"## {cat.upper()} Procedures")
        catalog.append("")
        catalog.append(f"**SST Layer**: {layer}")
        catalog.append(f"**Description**: {desc}")
        catalog.append("")

        # Deduplicate within category
        cat_procs = by_category[cat]
        seen = set()
        unique_cat = []
        for p in cat_procs:
            if p['name'] not in seen:
                seen.add(p['name'])
                unique_cat.append(p)

        # Sort by name
        unique_cat.sort(key=lambda x: x['name'].lower())

        catalog.append("| Procedure | Return | Parameters | Source |")
        catalog.append("|-----------|--------|------------|--------|")

        for proc in unique_cat[:50]:  # Limit to 50 per category for readability
            name = proc['name']
            ret = proc['return_type'][:10] if proc['return_type'] else 'void'
            params = proc['params'][:40] + '...' if len(proc['params']) > 40 else proc['params']
            params = params.replace('|', '\\|')  # Escape pipes for markdown
            source = proc['source_path'].split('/')[-1][:30]
            catalog.append(f"| `{name}` | {ret} | `{params}` | {source} |")

        if len(unique_cat) > 50:
            catalog.append(f"| ... | | | *{len(unique_cat) - 50} more* |")

        catalog.append("")
        catalog.append("---")
        catalog.append("")

    # Duplicates section
    catalog.append("## Duplicate Procedures (Consolidation Candidates)")
    catalog.append("")
    catalog.append("These procedures appear in multiple files and should be consolidated:")
    catalog.append("")

    duplicates = data.get('duplicates', {})
    dup_list = sorted(duplicates.items(), key=lambda x: -len(x[1]))[:30]

    catalog.append("| Procedure | Occurrences | Files |")
    catalog.append("|-----------|-------------|-------|")

    for name, occurrences in dup_list:
        count = len(occurrences)
        files = list(set(o['source_path'].split('/')[-1][:20] for o in occurrences))[:3]
        files_str = ', '.join(files)
        if len(set(o['source_path'].split('/')[-1] for o in occurrences)) > 3:
            files_str += '...'
        catalog.append(f"| `{name}` | {count} | {files_str} |")

    catalog.append("")
    catalog.append("---")
    catalog.append("")

    # Key procedures for SST integration
    catalog.append("## Key Procedures for SST Integration")
    catalog.append("")
    catalog.append("### Python Translation Candidates")
    catalog.append("")
    catalog.append("High-priority procedures for translation to Python `math_core.py`:")
    catalog.append("")

    key_procs = [
        ('Circle3Point', 'circle', 'Create circle from 3 points - core conformal operation'),
        ('PointToCircleTangents', 'tangent', 'Calculate tangent lines from point to circle'),
        ('TangentCircles', 'tangent', 'Find tangent lines between two circles'),
        ('CircleFromCurve', 'circle', 'Extract circle from NURBS curve'),
        ('xyzRotation', 'matrix', 'Quaternion-based 3D rotation'),
        ('GetRotationFromDirection', 'matrix', 'Rotation matrix from direction vectors'),
        ('ProjectToCameraPlane', 'sketch', 'Project point/curve to camera plane'),
        ('IntersectTwoCircles', 'circle', 'Circle intersection calculation'),
    ]

    catalog.append("| Procedure | Category | Purpose | SST Node |")
    catalog.append("|-----------|----------|---------|----------|")
    for name, cat, purpose in key_procs:
        node = 'TransformNode' if cat == 'matrix' else 'MutationNode'
        catalog.append(f"| `{name}` | {cat} | {purpose} | {node} |")

    catalog.append("")
    catalog.append("---")
    catalog.append("")
    catalog.append("## Naming Conventions")
    catalog.append("")
    catalog.append("| Suffix | Meaning |")
    catalog.append("|--------|---------|")
    catalog.append("| `*Z` | Z-plane focused operations |")
    catalog.append("| `*Vec` | Vector-based operations |")
    catalog.append("| `*Float` | Float coordinate input |")
    catalog.append("| `*String` | String array input |")
    catalog.append("| `*2`, `*3` | Iterative improvements |")
    catalog.append("| `*TF` | Returns True/False |")
    catalog.append("")
    catalog.append("---")
    catalog.append("")
    catalog.append("*Generated by extract_procedures.py and generate_catalog.py*")

    return '\n'.join(catalog)

def main():
    data = load_data()
    catalog = generate_catalog(data)

    output_path = MAYA_DIR / 'PROCEDURE_CATALOG.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(catalog)

    print(f"Catalog generated: {output_path}")
    print(f"Size: {len(catalog)} bytes")

if __name__ == "__main__":
    main()
