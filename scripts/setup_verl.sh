#!/usr/bin/env bash

set -euo pipefail

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
pin_file="$repo_dir/REQUIRED_VERL.txt"
patch_file="$repo_dir/patches/verl-v0.4.1-gcpo.patch"
verl_root=${VERL_ROOT:-"$repo_dir/third_party/verl"}

verl_repository=""
verl_tag=""
verl_commit=""
while IFS='=' read -r key value; do
    case "$key" in
        VERL_REPOSITORY) verl_repository=$value ;;
        VERL_TAG) verl_tag=$value ;;
        VERL_COMMIT) verl_commit=$value ;;
    esac
done < "$pin_file"

if [[ -z "$verl_repository" || -z "$verl_tag" || -z "$verl_commit" ]]; then
    echo "Invalid VERL pin file: $pin_file" >&2
    exit 1
fi

if [[ -e "$verl_root" ]]; then
    if [[ ! -d "$verl_root/.git" ]]; then
        echo "Refusing to use existing non-Git directory: $verl_root" >&2
        exit 1
    fi
else
    mkdir -p "$(dirname "$verl_root")"
    git clone --no-checkout "$verl_repository" "$verl_root"
    git -C "$verl_root" checkout --detach "$verl_commit"
fi

actual_commit=$(git -C "$verl_root" rev-parse HEAD)
if [[ "$actual_commit" != "$verl_commit" ]]; then
    echo "VERL checkout is at $actual_commit; required commit is $verl_commit ($verl_tag)." >&2
    exit 1
fi

if git -C "$verl_root" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
    echo "GCPO patch is already applied to $verl_root"
elif git -C "$verl_root" apply --check "$patch_file"; then
    git -C "$verl_root" apply "$patch_file"
    echo "Applied GCPO patch to $verl_root"
else
    echo "GCPO patch cannot be applied cleanly to $verl_root" >&2
    exit 1
fi

python3 -m pip install --no-build-isolation -e "$repo_dir"
echo "GCPO setup complete."
