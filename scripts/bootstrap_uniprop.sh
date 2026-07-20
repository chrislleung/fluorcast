#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/bootstrap_uniprop.sh [options]

Create or verify an isolated UniProp/nablaColors environment.

Options:
  --mode cpu|cuda             Bootstrap mode. Default: cpu.
  --python PATH               Python 3.10 executable. Default: python3.10.
  --venv PATH                 Isolated virtualenv path. Default: .venv-uniprop.
  --upstream-dir PATH         Clone path. Default: third_party/nablacolors.
  --revision-file PATH        Revision file. Default: third_party/nablacolors.REVISION.
  --repo-url URL              Override upstream Git URL from revision file.
  --dry-run                   Print planned actions without changing files.
  --json-output PATH          Write a small JSON status report.
  -h, --help                  Show this help.
USAGE
}

MODE="cpu"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR=".venv-uniprop"
UPSTREAM_DIR="third_party/nablacolors"
REVISION_FILE="third_party/nablacolors.REVISION"
REPO_URL=""
DRY_RUN=0
JSON_OUTPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="${2:-}"
            shift 2
            ;;
        --python)
            PYTHON_BIN="${2:-}"
            shift 2
            ;;
        --venv)
            VENV_DIR="${2:-}"
            shift 2
            ;;
        --upstream-dir)
            UPSTREAM_DIR="${2:-}"
            shift 2
            ;;
        --revision-file)
            REVISION_FILE="${2:-}"
            shift 2
            ;;
        --repo-url)
            REPO_URL="${2:-}"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --json-output)
            JSON_OUTPUT="${2:-}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ "$MODE" != "cpu" && "$MODE" != "cuda" ]]; then
    echo "--mode must be cpu or cuda" >&2
    exit 2
fi

read_revision_value() {
    local key="$1"
    local file="$2"
    awk -F= -v wanted="$key" '$1 == wanted {print $2}' "$file" | tail -n 1
}

json_escape() {
    printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

write_json_report() {
    local status="$1"
    local detail="$2"
    if [[ -z "$JSON_OUTPUT" ]]; then
        return
    fi
    local parent
    parent="$(dirname "$JSON_OUTPUT")"
    if [[ "$DRY_RUN" -eq 0 ]]; then
        mkdir -p "$parent"
    elif [[ ! -d "$parent" ]]; then
        echo "Dry run: JSON output parent does not exist: $parent" >&2
        return
    fi
    cat > "$JSON_OUTPUT" <<JSON
{
  "status": "$(json_escape "$status")",
  "detail": "$(json_escape "$detail")",
  "mode": "$(json_escape "$MODE")",
  "python": "$(json_escape "$PYTHON_BIN")",
  "venv_dir": "$(json_escape "$VENV_DIR")",
  "upstream_dir": "$(json_escape "$UPSTREAM_DIR")",
  "revision_file": "$(json_escape "$REVISION_FILE")",
  "pinned_commit": "$(json_escape "${PINNED_COMMIT:-}")"
}
JSON
}

run_or_echo() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf 'DRY-RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

if [[ ! -f "$REVISION_FILE" ]]; then
    echo "Revision file not found: $REVISION_FILE" >&2
    exit 2
fi

PINNED_COMMIT="$(read_revision_value commit "$REVISION_FILE")"
PINNED_REF="$(read_revision_value ref "$REVISION_FILE")"
REVISION_REPO="$(read_revision_value repo "$REVISION_FILE")"
REPO_URL="${REPO_URL:-$REVISION_REPO}"

if [[ -z "$PINNED_COMMIT" || -z "$REPO_URL" ]]; then
    echo "Revision file must define repo= and commit=: $REVISION_FILE" >&2
    exit 2
fi

echo "UniProp bootstrap mode: $MODE"
echo "Pinned nablaColors commit: $PINNED_COMMIT"
echo "Upstream directory: $UPSTREAM_DIR"
echo "Virtual environment: $VENV_DIR"

if [[ -d "$UPSTREAM_DIR" ]]; then
    if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
        echo "Existing upstream directory is not a Git checkout: $UPSTREAM_DIR" >&2
        write_json_report "failed" "existing upstream directory is not a Git checkout"
        exit 1
    fi
    CURRENT_COMMIT="$(git -C "$UPSTREAM_DIR" rev-parse HEAD)"
    if [[ "$CURRENT_COMMIT" != "$PINNED_COMMIT" ]]; then
        echo "Revision mismatch in $UPSTREAM_DIR" >&2
        echo "  expected: $PINNED_COMMIT" >&2
        echo "  actual:   $CURRENT_COMMIT" >&2
        write_json_report "failed" "revision mismatch"
        exit 1
    fi
    echo "Pinned upstream checkout already present."
else
    run_or_echo mkdir -p "$(dirname "$UPSTREAM_DIR")"
    run_or_echo git clone "$REPO_URL" "$UPSTREAM_DIR"
    run_or_echo git -C "$UPSTREAM_DIR" checkout --detach "$PINNED_COMMIT"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    CURRENT_COMMIT="$(git -C "$UPSTREAM_DIR" rev-parse HEAD)"
    if [[ "$CURRENT_COMMIT" != "$PINNED_COMMIT" ]]; then
        echo "Refusing to continue after clone; upstream revision does not match pin." >&2
        write_json_report "failed" "post-clone revision mismatch"
        exit 1
    fi
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    run_or_echo "$PYTHON_BIN" -m venv "$VENV_DIR"
    run_or_echo "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
    run_or_echo bash "$UPSTREAM_DIR/install_unicore.sh"
    run_or_echo "$VENV_DIR/bin/python" -m pip install -e "$UPSTREAM_DIR/Uni-Core"
    run_or_echo "$VENV_DIR/bin/python" -m pip install -e "$UPSTREAM_DIR/unimol_plus"
    write_json_report "dry-run" "planned bootstrap actions only"
    exit 0
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
"$VENV_PYTHON" - <<'PY'
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"UniProp bootstrap requires Python 3.10, got {sys.version.split()[0]}")
if sys.prefix == sys.base_prefix:
    raise SystemExit("Refusing to install outside an isolated virtual environment")
PY

"$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel

if [[ "$MODE" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"
fi

bash "$UPSTREAM_DIR/install_unicore.sh"

if [[ -d "$UPSTREAM_DIR/Uni-Core" ]]; then
    "$VENV_PYTHON" -m pip install -e "$UPSTREAM_DIR/Uni-Core"
else
    echo "WARNING: $UPSTREAM_DIR/Uni-Core not found after install_unicore.sh; verify Uni-Core manually." >&2
fi

if [[ ! -d "$UPSTREAM_DIR/unimol_plus" ]]; then
    echo "Uni-Mol+ directory not found: $UPSTREAM_DIR/unimol_plus" >&2
    write_json_report "failed" "unimol_plus directory missing"
    exit 1
fi
"$VENV_PYTHON" -m pip install -e "$UPSTREAM_DIR/unimol_plus"

write_json_report "ok" "bootstrap completed"
echo "UniProp bootstrap completed. Activate with: source $VENV_DIR/bin/activate"
