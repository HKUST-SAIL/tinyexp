#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/release.sh <version>

Examples:
  scripts/release.sh 0.0.4
  scripts/release.sh v0.0.4
EOF
}

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

input_version="$1"
version="${input_version#v}"
tag="v${version}"

if [[ ! "${version}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "Error: invalid version '${input_version}'. Expected format like 0.0.4 or v0.0.4."
    exit 1
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Error: working tree is not clean. Please commit or stash changes first."
    exit 1
fi

if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null; then
    echo "Error: local tag '${tag}' already exists."
    exit 1
fi

if git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null 2>&1; then
    echo "Error: remote tag '${tag}' already exists on origin."
    exit 1
fi

git_cliff_cmd=()

setup_git_cliff_cmd() {
    if command -v git-cliff >/dev/null 2>&1; then
        git_cliff_cmd=(git-cliff)
        return
    fi

    if command -v uvx >/dev/null 2>&1; then
        git_cliff_cmd=(uvx --from git-cliff git-cliff)
        return
    fi

    echo "Error: git-cliff is not installed and uvx is unavailable."
    exit 1
}

prepend_changelog_entry() {
    local entry_file="$1"
    local tmp
    local tail_tmp
    tmp="$(mktemp)"
    tail_tmp="$(mktemp)"

    if [[ -f CHANGELOG.md ]] && head -n 1 CHANGELOG.md | grep -qx "# Changelog"; then
        tail -n +2 CHANGELOG.md > "${tail_tmp}"
        sed -E '/./,$!d' "${tail_tmp}" > "${tail_tmp}.trimmed"
        {
            echo "# Changelog"
            echo
            cat "${entry_file}"
            echo
            cat "${tail_tmp}.trimmed"
        } > "${tmp}"
        rm -f "${tail_tmp}.trimmed"
    else
        {
            echo "# Changelog"
            echo
            cat "${entry_file}"
            if [[ -f CHANGELOG.md ]]; then
                echo
                cat CHANGELOG.md
            fi
        } > "${tmp}"
    fi

    rm -f "${tail_tmp}"
    mv "${tmp}" CHANGELOG.md
}

generate_changelog_if_missing() {
    local escaped_version
    local entry_file

    escaped_version="${version//./\\.}"
    if [[ -f CHANGELOG.md ]] && grep -qE "^## \[${escaped_version}\] - " CHANGELOG.md; then
        echo "Using existing changelog entry for ${version}"
        return
    fi

    if [[ ! -f cliff.toml ]]; then
        echo "Error: cliff.toml not found."
        exit 1
    fi

    setup_git_cliff_cmd

    entry_file="$(mktemp)"
    "${git_cliff_cmd[@]}" -c cliff.toml --unreleased --tag "${tag}" > "${entry_file}"

    if ! grep -qE '^### ' "${entry_file}"; then
        echo "Error: no unreleased commits found to generate changelog for ${version}."
        rm -f "${entry_file}"
        exit 1
    fi

    prepend_changelog_entry "${entry_file}"

    rm -f "${entry_file}"
    echo "Generated changelog entry for ${version} with git-cliff"
}

update_pyproject_version() {
    local tmp
    tmp="$(mktemp)"
    awk -v new_version="${version}" '
    BEGIN { updated = 0 }
    /^version = "/ && !updated {
        print "version = \"" new_version "\""
        updated = 1
        next
    }
    { print }
    END {
        if (!updated) {
            exit 1
        }
    }
    ' pyproject.toml > "${tmp}"
    mv "${tmp}" pyproject.toml
}

update_uv_lock_version() {
    local tmp
    tmp="$(mktemp)"
    awk -v new_version="${version}" '
    BEGIN {
        in_tinyexp = 0
        updated = 0
    }
    /^\[\[package\]\]$/ {
        in_tinyexp = 0
    }
    /^name = "tinyexp"$/ {
        in_tinyexp = 1
    }
    in_tinyexp && /^version = "/ && !updated {
        print "version = \"" new_version "\""
        updated = 1
        next
    }
    { print }
    END {
        if (!updated) {
            exit 1
        }
    }
    ' uv.lock > "${tmp}"
    mv "${tmp}" uv.lock
}

echo "Updating version to ${version}"
generate_changelog_if_missing
update_pyproject_version
update_uv_lock_version

echo "Running checks and tests"
make check
make test

echo "Building package"
make build

echo "Committing release changes"
git add CHANGELOG.md pyproject.toml uv.lock
git commit -m "release: ${tag}"
git tag -a "${tag}" -m "Release ${tag}"

echo "Publishing to PyPI"
make publish

echo "Pushing commit and tag"
git push origin main
git push origin "${tag}"

echo "Release completed: ${tag}"
