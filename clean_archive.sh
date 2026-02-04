#!/usr/bin/env bash

set -euo pipefail

base_dir="${1:-./archive}"

if [[ ! -d "$base_dir" ]]; then
  printf 'Diretório base "%s" não encontrado.\n' "$base_dir" >&2
  exit 1
fi

# Garante que globs vazios resultem em listas vazias (não literais) e incluam diretórios ocultos.
shopt -s nullglob dotglob
trap 'shopt -u nullglob dotglob' EXIT

# Define ordenação lexicográfica consistente para os globs.
export LC_COLLATE=C

for dir in "$base_dir"/*/; do
  [[ -d "$dir" ]] || continue

  png_files=("$dir"*.png)
  if ((${#png_files[@]})); then
    rm -f -- "${png_files[@]}"
  fi

  dcm_files=("$dir"*.dcm)
  if ((${#dcm_files[@]} > 1)); then
    keep="${dcm_files[0]}"
    for file in "${dcm_files[@]:1}"; do
      rm -f -- "$file"
    done
  fi
done
