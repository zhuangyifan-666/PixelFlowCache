#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "Repository: ${ROOT}"

repos=(third_party/JiT third_party/DeCo third_party/PixelGen)

if [[ -f .gitmodules ]]; then
  echo "Initializing configured submodule metadata..."
  git submodule sync -- "${repos[@]}"
  git submodule init "${repos[@]}"
else
  echo "No .gitmodules found; using existing third_party checkouts if present."
fi

for repo in "${repos[@]}"; do
  if [[ -f .gitmodules ]] && { [[ ! -d "${repo}" ]] || ! git -C "${repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1; }; then
    echo "Initializing/updating submodule ${repo}..."
    git submodule update --init --recursive "${repo}"
  fi

  if [[ ! -d "${repo}" ]]; then
    echo "Missing ${repo}."
    echo "Run one of:"
    echo "  git submodule add https://github.com/LTH14/JiT.git third_party/JiT"
    echo "  git submodule add https://github.com/Zehong-Ma/DeCo.git third_party/DeCo"
    echo "  git submodule add https://github.com/Zehong-Ma/PixelGen.git third_party/PixelGen"
    exit 1
  fi

  if ! git -C "${repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "${repo} exists but is not a valid initialized git repository after submodule update."
    exit 1
  fi

  echo "${repo} commit: $(git -C "${repo}" rev-parse HEAD)"
done

if [[ -f .gitmodules ]]; then
  echo "Submodule status:"
  git submodule status "${repos[@]}" || true
fi

echo "third_party setup check complete."
