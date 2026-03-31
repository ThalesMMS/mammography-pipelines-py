#!/usr/bin/env bash
# Helper script to find Python executable on Windows

# Try common Python locations on Windows
for python_cmd in \
    "/c/Users/$USER/AppData/Local/Programs/Python313/python.exe" \
    "/c/Users/$USER/AppData/Local/Programs/Python312/python.exe" \
    "/c/Python313/python.exe" \
    "/c/Python312/python.exe" \
    "/c/Python311/python.exe" \
    "/c/Python310/python.exe" \
    "/c/Program Files/Python313/python.exe" \
    "/c/Program Files/Python312/python.exe" \
    "/c/Program Files/Python311/python.exe" \
    "/c/Users/$USER/AppData/Local/Programs/Python/Python313/python.exe" \
    "/c/Users/$USER/AppData/Local/Programs/Python/Python312/python.exe" \
    "/c/Users/$USER/AppData/Local/Programs/Python/Python311/python.exe" \
    "/c/ProgramData/anaconda3/python.exe" \
    "/c/Users/$USER/anaconda3/python.exe" \
    "/c/Users/$USER/miniconda3/python.exe"
do
    if [ -f "$python_cmd" ]; then
        echo "$python_cmd"
        exit 0
    fi
done

# If nothing found, output error
echo "No Python found" >&2
exit 1
