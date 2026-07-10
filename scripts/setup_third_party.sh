#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

print_only=false
check_only=false
for arg in "$@"; do
  case "${arg}" in
    --print-only) print_only=true ;;
    --check-only) check_only=true ;;
    -h|--help)
      echo "Usage: scripts/setup_third_party.sh [--print-only | --check-only]"
      exit 0
      ;;
    *) echo "Unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done
if [[ "${print_only}" == true && "${check_only}" == true ]]; then
  echo "Choose only one of --print-only and --check-only." >&2
  exit 2
fi

repos=(
  third_party/JiT
  third_party/DeCo
  third_party/PixelGen
  third_party/PixelDiT
)
declare -A urls=(
  [third_party/JiT]="https://github.com/LTH14/JiT.git"
  [third_party/DeCo]="https://github.com/Zehong-Ma/DeCo.git"
  [third_party/PixelGen]="https://github.com/Zehong-Ma/PixelGen.git"
  [third_party/PixelDiT]="https://github.com/NVlabs/PixelDiT.git"
)

echo "Repository: ${ROOT}"
for repo in "${repos[@]}"; do
  echo "${repo} -> ${urls[${repo}]}"
done

if [[ "${print_only}" == true ]]; then
  printf 'git submodule sync --recursive --'
  printf ' %q' "${repos[@]}"
  printf '\n'
  printf 'git submodule update --init --recursive --'
  printf ' %q' "${repos[@]}"
  printf '\n'
  exit 0
fi

if [[ ! -f .gitmodules ]]; then
  echo "Missing .gitmodules." >&2
  exit 1
fi

failed=false
for repo in "${repos[@]}"; do
  configured_url="$(git config -f .gitmodules --get "submodule.${repo}.url" || true)"
  if [[ "${configured_url}" != "${urls[${repo}]}" ]]; then
    echo "URL mismatch for ${repo}: ${configured_url:-missing}" >&2
    failed=true
  fi
  if [[ ! -d "${repo}" ]] || ! git -C "${repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Uninitialized submodule: ${repo}" >&2
    failed=true
    continue
  fi
  if [[ ! -f "${repo}/README.md" ]] && [[ ! -f "${repo}/README.MD" ]]; then
    echo "Missing README/source marker in ${repo}" >&2
    failed=true
  fi
  echo "${repo} commit: $(git -C "${repo}" rev-parse HEAD)"
done

if [[ "${check_only}" == true ]]; then
  [[ "${failed}" == false ]]
  git submodule status --recursive -- "${repos[@]}"
  echo "third_party check-only complete; no network operation was attempted."
  exit 0
fi

git submodule sync --recursive -- "${repos[@]}"
git submodule update --init --recursive -- "${repos[@]}"

for repo in "${repos[@]}"; do
  if ! git -C "${repo}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Submodule update did not initialize ${repo}." >&2
    exit 1
  fi
  echo "${repo} commit: $(git -C "${repo}" rev-parse HEAD)"
done
git submodule status --recursive -- "${repos[@]}"
echo "third_party setup complete."
