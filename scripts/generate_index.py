#!/usr/bin/env python3
"""
Index Generator for 3D Tools ML Hybrid
Scans directory and generates index.jsonl

Usage:
    python generate_index.py --root . --output index.jsonl
"""

import os
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

# Category inference rules
CATEGORY_PATTERNS = {
    'theory': ['schema', 'spec', 'architecture', 'SKELETAL'],
    'implementation': ['impl', 'transpiler', 'emitter'],
    'navigation': ['INDEX', 'index'],
    'reference': ['ref', 'api'],
}

# Tag inference rules
TAG_PATTERNS = {
    'core': ['SKELETAL', 'SST', 'architecture'],
    'state': ['state', 'matrix', 'transform', 'scope'],
    'mutation': ['mutation', 'geometry', 'primitive', 'csg', 'deform'],
    'maya': ['maya', 'mel', 'cmds'],
    'blender': ['blender', 'bpy'],
    'unreal': ['unreal', 'ue4', 'ue5', 'blueprint'],
    'l-system': ['l-system', 'lsystem', 'axiom', 'production'],
}


def generate_file_id(filepath: str, root: str) -> str:
    """Generate stable ID from path hash."""
    rel_path = os.path.relpath(filepath, root)
    hash_input = rel_path.encode('utf-8')
    return 'f' + hashlib.sha256(hash_input).hexdigest()[:8]


def infer_category(filepath: str, content: str) -> str:
    """Infer category from filepath and content."""
    combined = filepath + ' ' + content[:500]
    for category, patterns in CATEGORY_PATTERNS.items():
        if any(p.lower() in combined.lower() for p in patterns):
            return category
    return 'general'


def infer_tags(filepath: str, content: str) -> list:
    """Infer tags from filepath and content."""
    combined = filepath + ' ' + content[:1000]
    tags = []
    for tag, patterns in TAG_PATTERNS.items():
        if any(p.lower() in combined.lower() for p in patterns):
            tags.append(tag)
    return tags if tags else ['general']


def extract_title(content: str) -> str:
    """Extract title from markdown content."""
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('# '):
            return line[2:].strip()
    return ''


def scan_file(filepath: str, root: str) -> dict:
    """Generate file record from path."""
    rel_path = os.path.relpath(filepath, root).replace('\\', '/')
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        content = ''
    
    file_id = generate_file_id(filepath, root)
    category = infer_category(rel_path, content)
    tags = infer_tags(rel_path, content)
    title = extract_title(content)
    
    # Priority: 1=theory/core, 2=implementation, 3=navigation/other
    priority = 1 if category == 'theory' else (2 if category == 'implementation' else 3)
    
    return {
        'type': 'file',
        'id': file_id,
        'path': rel_path,
        'category': category,
        'tags': tags,
        'priority': priority,
        'title': title,
        'format': Path(filepath).suffix[1:] if Path(filepath).suffix else 'unknown',
        'size_kb': os.path.getsize(filepath) / 1024,
    }


def generate_tag_indexes(file_records: list) -> list:
    """Generate tag index records from file records."""
    tag_map = {}
    for record in file_records:
        for tag in record.get('tags', []):
            if tag not in tag_map:
                tag_map[tag] = []
            tag_map[tag].append(record['id'])
    
    return [
        {'type': 'tag_index', 'tag': tag, 'files': files}
        for tag, files in sorted(tag_map.items())
    ]


def main():
    parser = argparse.ArgumentParser(description='Generate JSONL index')
    parser.add_argument('--root', default='.', help='Root directory to scan')
    parser.add_argument('--output', default='index.jsonl', help='Output file')
    args = parser.parse_args()
    
    root = os.path.abspath(args.root)
    records = []
    
    # Meta record
    records.append({
        'type': 'meta',
        'version': '1.0',
        'created': datetime.now().strftime('%Y-%m-%d'),
        'schema': 'ukis-v2',
        'description': '3D Tools ML Hybrid - Auto-generated index',
    })
    
    # Scan files
    file_records = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip hidden and script directories for file records
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
        
        for filename in filenames:
            if filename.startswith('.'):
                continue
            if filename == args.output:
                continue
            if not filename.endswith(('.md', '.jsonl', '.json', '.py')):
                continue
                
            filepath = os.path.join(dirpath, filename)
            record = scan_file(filepath, root)
            file_records.append(record)
    
    records.extend(file_records)
    
    # Generate tag indexes
    records.extend(generate_tag_indexes(file_records))
    
    # Write output
    output_path = os.path.join(root, args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')
    
    print(f"Generated {len(records)} records in {output_path}")


if __name__ == '__main__':
    main()
