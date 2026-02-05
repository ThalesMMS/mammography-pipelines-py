#!/usr/bin/env python3
"""Cross-reference validation script for LaTeX labels and references."""

import re
import sys
from pathlib import Path
from collections import defaultdict

def extract_labels(content, filepath):
    """Extract all \label{} commands from content."""
    pattern = r'\\label\{([^}]+)\}'
    matches = re.findall(pattern, content)
    return [(label, filepath) for label in matches]

def extract_refs(content, filepath):
    """Extract all \ref{} commands from content."""
    pattern = r'\\ref\{([^}]+)\}'
    matches = re.findall(pattern, content)
    return [(ref, filepath) for ref in matches]

def main():
    article_dir = Path('./Article')

    # Find all .tex files in chapters/ and sections/
    tex_files = []
    for subdir in ['chapters', 'sections']:
        subdir_path = article_dir / subdir
        if subdir_path.exists():
            tex_files.extend(subdir_path.glob('*.tex'))

    # Collect all labels and references
    all_labels = {}  # label -> filepath
    all_refs = defaultdict(list)  # ref -> list of filepaths

    for tex_file in tex_files:
        try:
            content = tex_file.read_text(encoding='utf-8')

            # Extract labels
            for label, filepath in extract_labels(content, tex_file):
                if label in all_labels:
                    print(f"⚠️  WARNING: Duplicate label '{label}'")
                    print(f"   First: {all_labels[label]}")
                    print(f"   Also in: {filepath}")
                else:
                    all_labels[label] = filepath

            # Extract references
            for ref, filepath in extract_refs(content, tex_file):
                all_refs[ref].append(filepath)

        except Exception as e:
            print(f"❌ Error reading {tex_file}: {e}")
            continue

    # Validate: check for missing labels
    missing_labels = []
    for ref, filepaths in all_refs.items():
        if ref not in all_labels:
            missing_labels.append((ref, filepaths))

    # Validate: check for unused labels
    unused_labels = []
    for label in all_labels:
        if label not in all_refs:
            unused_labels.append(label)

    # Print report
    print("=" * 70)
    print("CROSS-REFERENCE VALIDATION REPORT")
    print("=" * 70)
    print()

    print(f"📊 Statistics:")
    print(f"   Total labels: {len(all_labels)}")
    print(f"   Total references: {sum(len(fps) for fps in all_refs.values())}")
    print(f"   Unique referenced labels: {len(all_refs)}")
    print()

    if missing_labels:
        print(f"❌ ERRORS: {len(missing_labels)} reference(s) without corresponding label:")
        print()
        for ref, filepaths in missing_labels:
            print(f"   \\ref{{{ref}}}")
            for fp in filepaths:
                print(f"      in {fp}")
        print()
    else:
        print("✅ All references have corresponding labels!")
        print()

    if unused_labels:
        print(f"⚠️  WARNING: {len(unused_labels)} label(s) defined but never referenced:")
        print()
        for label in sorted(unused_labels):
            print(f"   \\label{{{label}}} in {all_labels[label]}")
        print()
    else:
        print("✅ All labels are referenced!")
        print()

    # Print label types summary
    label_types = defaultdict(int)
    for label in all_labels:
        if ':' in label:
            prefix = label.split(':')[0]
            label_types[prefix] += 1
        else:
            label_types['(no-prefix)'] += 1

    print("📋 Label types:")
    for prefix, count in sorted(label_types.items()):
        print(f"   {prefix}: {count}")
    print()

    # Exit with error if missing labels
    if missing_labels:
        print("=" * 70)
        print("❌ VALIDATION FAILED: Missing labels detected")
        print("=" * 70)
        sys.exit(1)
    else:
        print("=" * 70)
        print("✅ VALIDATION PASSED")
        print("=" * 70)
        sys.exit(0)

if __name__ == '__main__':
    main()
