#!/bin/bash
# Cross-reference validation script for LaTeX labels and references

set -e

cd ./Article

echo "======================================================================"
echo "CROSS-REFERENCE VALIDATION REPORT"
echo "======================================================================"
echo ""

# Extract all labels
echo "Extracting labels..."
labels_file=$(mktemp)
grep -roh '\\label{[^}]\+}' chapters/ sections/ 2>/dev/null | sed 's/\\label{\([^}]*\)}/\1/' | sort > "$labels_file"

# Extract all references
echo "Extracting references..."
refs_file=$(mktemp)
grep -roh '\\ref{[^}]\+}' chapters/ sections/ 2>/dev/null | sed 's/\\ref{\([^}]*\)}/\1/' | sort > "$refs_file"

# Count statistics
total_labels=$(wc -l < "$labels_file")
total_refs=$(wc -l < "$refs_file")
unique_labels=$(sort -u "$labels_file" | wc -l)
unique_refs=$(sort -u "$refs_file" | wc -l)

echo "📊 Statistics:"
echo "   Total labels: $total_labels"
echo "   Unique labels: $unique_labels"
echo "   Total references: $total_refs"
echo "   Unique references: $unique_refs"
echo ""

# Find missing labels (refs without corresponding label)
echo "Checking for missing labels..."
missing_file=$(mktemp)
sort -u "$refs_file" | while read -r ref; do
    if ! grep -Fxq "$ref" "$labels_file"; then
        echo "$ref"
    fi
done > "$missing_file"

missing_count=$(wc -l < "$missing_file")

if [ "$missing_count" -gt 0 ]; then
    echo "❌ ERRORS: $missing_count reference(s) without corresponding label:"
    echo ""
    while read -r ref; do
        echo "   \\ref{$ref}"
        grep -rn "\\ref{$ref}" chapters/ sections/ 2>/dev/null | head -3 | sed 's/^/      /'
    done < "$missing_file"
    echo ""
else
    echo "✅ All references have corresponding labels!"
    echo ""
fi

# Find unused labels (labels never referenced)
echo "Checking for unused labels..."
unused_file=$(mktemp)
sort -u "$labels_file" | while read -r label; do
    if ! grep -Fxq "$label" "$refs_file"; then
        echo "$label"
    fi
done > "$unused_file"

unused_count=$(wc -l < "$unused_file")

if [ "$unused_count" -gt 0 ]; then
    echo "⚠️  WARNING: $unused_count label(s) defined but never referenced:"
    echo ""
    while read -r label; do
        file=$(grep -rl "\\label{$label}" chapters/ sections/ 2>/dev/null | head -1)
        echo "   \\label{$label} in $file"
    done < "$unused_file"
    echo ""
else
    echo "✅ All labels are referenced!"
    echo ""
fi

# Analyze label types
echo "📋 Label types:"
sort -u "$labels_file" | sed 's/:.*//;s/^/   /' | uniq -c | sort -rn

echo ""

# Check for duplicate labels
echo "Checking for duplicate labels..."
dup_file=$(mktemp)
sort "$labels_file" | uniq -d > "$dup_file"
dup_count=$(wc -l < "$dup_file")

if [ "$dup_count" -gt 0 ]; then
    echo "⚠️  WARNING: $dup_count duplicate label(s) found:"
    echo ""
    while read -r dup; do
        echo "   \\label{$dup}"
        grep -rn "\\label{$dup}" chapters/ sections/ 2>/dev/null | sed 's/^/      /'
    done < "$dup_file"
    echo ""
else
    echo "✅ No duplicate labels found!"
    echo ""
fi

# Cleanup temp files
rm -f "$labels_file" "$refs_file" "$missing_file" "$unused_file" "$dup_file"

# Final result
echo "======================================================================"
if [ "$missing_count" -gt 0 ]; then
    echo "❌ VALIDATION FAILED: Missing labels detected"
    echo "======================================================================"
    exit 1
else
    echo "✅ VALIDATION PASSED"
    echo "======================================================================"
    exit 0
fi
