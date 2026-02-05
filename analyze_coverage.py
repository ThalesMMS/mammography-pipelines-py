#!/usr/bin/env python3
"""Analyze coverage.json and generate coverage gaps report."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

def analyze_coverage(coverage_file: str = "coverage.json") -> Tuple[Dict, List]:
    """Analyze coverage data and identify gaps."""
    with open(coverage_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    files = data.get('files', {})
    totals = data.get('totals', {})

    # Categorize files by module
    modules = {}
    for filepath, file_data in files.items():
        # Only process source files
        if 'src/mammography/' not in filepath:
            continue

        summary = file_data.get('summary', {})
        coverage_pct = summary.get('percent_covered', 0)
        missing_lines = file_data.get('missing_lines', [])
        num_statements = summary.get('num_statements', 0)
        covered_lines = summary.get('covered_lines', 0)

        # Normalize path
        module_path = filepath.replace('\\', '/')

        # Determine priority based on module directory
        priority = 'LOW'
        if any(x in module_path for x in ['/data/', '/models/', '/training/', '/io/']):
            priority = 'CRITICAL'
        elif any(x in module_path for x in ['/commands/', 'config.py', 'cli.py']):
            priority = 'HIGH'
        elif any(x in module_path for x in ['/vis/', '/clustering/', '/eval']):
            priority = 'MEDIUM'

        # Only include files under 80% coverage
        if coverage_pct < 80:
            modules[filepath] = {
                'path': module_path,
                'coverage': coverage_pct,
                'missing_lines': missing_lines,
                'num_statements': num_statements,
                'covered_lines': covered_lines,
                'priority': priority
            }

    # Sort by priority (CRITICAL first), then by coverage (lowest first)
    priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
    sorted_modules = sorted(
        modules.items(),
        key=lambda x: (priority_order[x[1]['priority']], x[1]['coverage'])
    )

    return totals, sorted_modules

def generate_markdown_report(totals: Dict, modules: List) -> str:
    """Generate markdown report of coverage gaps."""

    lines = []
    lines.append("# Coverage Gaps Report")
    lines.append("")
    lines.append("**Generated:** 2026-02-04")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total Coverage:** {totals.get('percent_covered', 0):.2f}%")
    lines.append(f"- **Covered Lines:** {totals.get('covered_lines', 0):,} / {totals.get('num_statements', 0):,}")
    lines.append(f"- **Modules Under 80%:** {len(modules)}")
    lines.append("")
    lines.append("## Priority Legend")
    lines.append("")
    lines.append("- 🔴 **CRITICAL**: Core modules (data, models, training, io) - Must be tested first")
    lines.append("- 🟠 **HIGH**: CLI and configuration - Important for user interaction")
    lines.append("- 🟡 **MEDIUM**: Visualization, clustering, evaluation - Important but less critical")
    lines.append("- 🟢 **LOW**: Utility and helper modules")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Group by priority
    priority_groups = {'CRITICAL': [], 'HIGH': [], 'MEDIUM': [], 'LOW': []}
    for filepath, info in modules:
        priority_groups[info['priority']].append((filepath, info))

    # Generate sections for each priority
    priority_symbols = {
        'CRITICAL': '🔴',
        'HIGH': '🟠',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }

    for priority in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        if not priority_groups[priority]:
            continue

        lines.append(f"## {priority_symbols[priority]} {priority} Priority Modules")
        lines.append("")

        for filepath, info in priority_groups[priority]:
            # Extract module name
            module_name = info['path'].split('src/mammography/')[-1]

            lines.append(f"### `{module_name}`")
            lines.append("")
            lines.append(f"- **Coverage:** {info['coverage']:.2f}%")
            lines.append(f"- **Lines:** {info['covered_lines']} / {info['num_statements']} covered")
            lines.append(f"- **Missing:** {len(info['missing_lines'])} uncovered lines")
            lines.append("")

            # Show missing line ranges
            if info['missing_lines']:
                lines.append("**Uncovered Lines:**")
                lines.append("")
                lines.append("```")
                # Format missing lines into ranges
                missing = sorted(info['missing_lines'])
                ranges = []
                start = missing[0]
                end = missing[0]

                for line in missing[1:]:
                    if line == end + 1:
                        end = line
                    else:
                        if start == end:
                            ranges.append(f"{start}")
                        else:
                            ranges.append(f"{start}-{end}")
                        start = line
                        end = line

                # Add final range
                if start == end:
                    ranges.append(f"{start}")
                else:
                    ranges.append(f"{start}-{end}")

                # Display in chunks
                chunk_size = 10
                for i in range(0, len(ranges), chunk_size):
                    chunk = ranges[i:i+chunk_size]
                    lines.append(", ".join(chunk))
                    if i + chunk_size < len(ranges):
                        lines.append("")

                lines.append("```")
                lines.append("")

            lines.append("---")
            lines.append("")

    # Add recommendations section
    lines.append("## Recommendations")
    lines.append("")
    lines.append("### Phase 1: Critical Modules (Target: 80%+ coverage)")
    lines.append("")

    critical_count = len(priority_groups['CRITICAL'])
    if critical_count > 0:
        lines.append(f"Focus on **{critical_count} CRITICAL** modules first:")
        lines.append("")
        for filepath, info in priority_groups['CRITICAL'][:10]:
            module_name = info['path'].split('src/mammography/')[-1]
            lines.append(f"1. `{module_name}` ({info['coverage']:.1f}% → 80%)")
        if critical_count > 10:
            lines.append(f"   ... and {critical_count - 10} more critical modules")
    else:
        lines.append("✅ No critical modules under 80%")

    lines.append("")
    lines.append("### Phase 2: High Priority Modules")
    lines.append("")

    high_count = len(priority_groups['HIGH'])
    if high_count > 0:
        lines.append(f"Address **{high_count} HIGH** priority modules:")
        lines.append("")
        for filepath, info in priority_groups['HIGH'][:5]:
            module_name = info['path'].split('src/mammography/')[-1]
            lines.append(f"1. `{module_name}` ({info['coverage']:.1f}% → 80%)")
        if high_count > 5:
            lines.append(f"   ... and {high_count - 5} more high priority modules")
    else:
        lines.append("✅ No high priority modules under 80%")

    lines.append("")
    lines.append("### Phase 3: Medium & Low Priority")
    lines.append("")

    medium_low_count = len(priority_groups['MEDIUM']) + len(priority_groups['LOW'])
    lines.append(f"Complete remaining **{medium_low_count}** modules to reach project-wide 80%+ coverage.")
    lines.append("")

    lines.append("## Testing Strategy")
    lines.append("")
    lines.append("### For Each Module:")
    lines.append("")
    lines.append("1. **Review uncovered lines** - Understand what code needs testing")
    lines.append("2. **Identify test patterns** - Unit tests, integration tests, or both?")
    lines.append("3. **Create fixtures** - Reusable test data for the module")
    lines.append("4. **Write tests** - Cover happy paths, edge cases, and error handling")
    lines.append("5. **Verify coverage** - Run `pytest --cov=<module>` to confirm improvement")
    lines.append("")
    lines.append("### Test Patterns to Use:")
    lines.append("")
    lines.append("- **Config/Data modules**: Use parametrized tests with `@pytest.mark.parametrize`")
    lines.append("- **Model modules**: Test instantiation, forward passes, output shapes")
    lines.append("- **Training modules**: Mock expensive operations, test with minimal epochs")
    lines.append("- **I/O modules**: Use `tmp_path` fixture and mock DICOM data")
    lines.append("- **CLI modules**: Use subprocess to test commands with synthetic data")
    lines.append("")

    return "\n".join(lines)

if __name__ == "__main__":
    print("Analyzing coverage data...")
    totals, modules = analyze_coverage()

    print(f"Found {len(modules)} modules under 80% coverage")

    # Generate markdown report
    report = generate_markdown_report(totals, modules)

    # Write to file
    output_file = "tests/coverage_gaps.md"
    Path("tests").mkdir(exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✅ Coverage gaps report written to: {output_file}")
    print(f"   Total modules under 80%: {len(modules)}")
