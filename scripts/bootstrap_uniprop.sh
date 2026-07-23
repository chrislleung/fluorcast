#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/bootstrap_uniprop.sh [options]

Create or verify the isolated FluorCast UniProp/nablaColors environment.

Options:
  --mode cpu|cuda             Bootstrap mode. Default: cpu.
  --python PATH               Python 3.10 executable. Default: python3.10.
  --venv PATH                 Isolated virtualenv path. Default: .venv-uniprop.
  --upstream-dir PATH         Clone path. Default: third_party/nablacolors.
  --revision-file PATH        Revision file. Default: third_party/nablacolors.REVISION.
  --repo-url URL              Override upstream Git URL from revision file.
  --torch-spec SPEC           PyTorch requirement. Default: torch==2.6.*.
  --enable-cuda-ext           Build optional Uni-Core fused CUDA extensions.
  --clean                     Remove and recreate the virtualenv before install.
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
TORCH_SPEC="${FLUORCAST_UNIPROP_TORCH_SPEC:-torch==2.6.*}"
NUMPY_SPEC="${FLUORCAST_UNIPROP_NUMPY_SPEC:-numpy==2.2.6}"
ENABLE_CUDA_EXT=0
CLEAN=0
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
        --torch-spec)
            TORCH_SPEC="${2:-}"
            shift 2
            ;;
        --enable-cuda-ext)
            ENABLE_CUDA_EXT=1
            shift
            ;;
        --clean)
            CLEAN=1
            shift
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
  "pinned_commit": "$(json_escape "${PINNED_COMMIT:-}")",
  "torch_spec": "$(json_escape "$TORCH_SPEC")",
  "numpy_spec": "$(json_escape "$NUMPY_SPEC")",
  "enable_cuda_ext": $ENABLE_CUDA_EXT
}
JSON
}

stage() {
    echo
    echo "==> $1"
}

fail() {
    local detail="$1"
    echo "ERROR: $detail" >&2
    write_json_report "failed" "$detail"
    exit 1
}

run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf 'DRY-RUN:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

python_here() {
    local script="$1"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf 'DRY-RUN: %q - <<'\''PY'\''\n%s\nPY\n' "$VENV_PYTHON" "$script"
    else
        "$VENV_PYTHON" - "$@" <<<"$script"
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

UNICORE_DIR="$UPSTREAM_DIR/Uni-Core"
UNIMOL_PLUS_DIR="$UPSTREAM_DIR/unimol_plus"
VENV_PYTHON="$VENV_DIR/bin/python"

echo "UniProp bootstrap mode: $MODE"
echo "Pinned nablaColors commit: $PINNED_COMMIT"
echo "Pinned nablaColors ref: ${PINNED_REF:-unknown}"
echo "Upstream directory: $UPSTREAM_DIR"
echo "Virtual environment: $VENV_DIR"
echo "PyTorch requirement: $TORCH_SPEC"
echo "NumPy requirement: $NUMPY_SPEC"
export FLUORCAST_UNIPROP_BOOTSTRAP_MODE="$MODE"
export FLUORCAST_UNIPROP_TORCH_SPEC="$TORCH_SPEC"
export FLUORCAST_UNIPROP_NUMPY_SPEC="$NUMPY_SPEC"

stage "Pinned upstream checkout"
if [[ -d "$UPSTREAM_DIR" ]]; then
    if [[ ! -d "$UPSTREAM_DIR/.git" ]]; then
        fail "Existing upstream directory is not a Git checkout: $UPSTREAM_DIR"
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
    run_cmd mkdir -p "$(dirname "$UPSTREAM_DIR")"
    run_cmd git clone "$REPO_URL" "$UPSTREAM_DIR"
    run_cmd git -C "$UPSTREAM_DIR" checkout --detach "$PINNED_COMMIT"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
    CURRENT_COMMIT="$(git -C "$UPSTREAM_DIR" rev-parse HEAD)"
    if [[ "$CURRENT_COMMIT" != "$PINNED_COMMIT" ]]; then
        fail "Refusing to continue after clone; upstream revision does not match pin."
    fi
fi

if [[ "$DRY_RUN" -eq 0 && ! -d "$UNICORE_DIR" ]]; then
    fail "Uni-Core directory is missing from pinned checkout: $UNICORE_DIR"
fi
if [[ "$DRY_RUN" -eq 0 && ! -d "$UNIMOL_PLUS_DIR" ]]; then
    fail "Uni-Mol+ directory is missing from pinned checkout: $UNIMOL_PLUS_DIR"
fi

stage "Python 3.10 virtual environment"
if [[ "$CLEAN" -eq 1 ]]; then
    if [[ "$VENV_DIR" == "/" || -z "$VENV_DIR" ]]; then
        fail "Refusing unsafe --clean target: $VENV_DIR"
    fi
    run_cmd rm -rf "$VENV_DIR"
fi

if [[ -e "$VENV_DIR" && ! -x "$VENV_PYTHON" ]]; then
    fail "Partial UniProp environment detected at $VENV_DIR; rerun with --clean to rebuild it."
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    run_cmd "$PYTHON_BIN" -m venv "$VENV_DIR"
else
    echo "Reusing existing virtual environment."
fi

python_here "$(cat <<'PY'
import os
import sys
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"UniProp bootstrap requires Python 3.10, got {sys.version.split()[0]}")
if sys.prefix == sys.base_prefix:
    raise SystemExit("Refusing to install outside an isolated virtual environment")
if os.environ.get("FLUORCAST_UNIPROP_BOOTSTRAP_MODE") == "cuda" and sys.platform.startswith("win"):
    raise SystemExit("CUDA UniProp bootstrap must run on Nibi/Linux, not native Windows.")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")
print(f"Environment path: {sys.prefix}")
PY
)" || fail "Python 3.10 virtual environment validation failed."

stage "Build tools and NumPy"
run_cmd "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
run_cmd "$VENV_PYTHON" -m pip install "$NUMPY_SPEC"

stage "PyTorch"
if [[ "$MODE" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:--1}"
fi
python_here "$(cat <<'PY'
import importlib.util
import os
import sys
if importlib.util.find_spec("torch") is not None:
    import torch
    print(f"PyTorch already installed: {torch.__version__}")
    print(f"Torch compiled CUDA version: {getattr(torch.version, 'cuda', None)}")
    print(f"CUDA available at runtime: {torch.cuda.is_available()}")
    if os.environ.get("FLUORCAST_UNIPROP_BOOTSTRAP_MODE") == "cuda" and getattr(torch.version, "cuda", None) is None:
        raise SystemExit("Installed torch is CPU-only; CUDA mode requires an Alliance CUDA-capable torch wheel.")
    raise SystemExit(0)
print(f"Selected PyTorch requirement: {os.environ['FLUORCAST_UNIPROP_TORCH_SPEC']}")
PY
)" || fail "Existing PyTorch installation is not compatible."

if [[ "$DRY_RUN" -eq 1 ]]; then
    run_cmd "$VENV_PYTHON" -m pip install "$TORCH_SPEC"
elif ! "$VENV_PYTHON" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") is not None else 1)
PY
then
    TORCH_REPORT="$(mktemp)"
    if ! "$VENV_PYTHON" -m pip install --dry-run --report "$TORCH_REPORT" "$TORCH_SPEC"; then
        rm -f "$TORCH_REPORT"
        fail "No compatible PyTorch wheel found for $TORCH_SPEC with this Python/pip configuration."
    fi
    "$VENV_PYTHON" - "$TORCH_REPORT" <<'PY'
import json
import sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
matches = [item for item in payload.get("install", []) if item.get("metadata", {}).get("name", "").lower() == "torch"]
if not matches:
    raise SystemExit("pip dry-run did not identify a torch wheel to install.")
torch_item = matches[0]
metadata = torch_item.get("metadata", {})
download = torch_item.get("download_info", {})
print(f"Selected PyTorch wheel/version: torch {metadata.get('version')} from {download.get('url', 'configured pip source')}")
PY
    run_cmd "$VENV_PYTHON" -m pip install "$TORCH_SPEC"
    rm -f "$TORCH_REPORT"
fi

python_here "$(cat <<'PY'
import os
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"Torch compiled CUDA version: {getattr(torch.version, 'cuda', None)}")
print(f"CUDA available at runtime: {torch.cuda.is_available()}")
if os.environ.get("FLUORCAST_UNIPROP_BOOTSTRAP_MODE") == "cuda" and getattr(torch.version, "cuda", None) is None:
    raise SystemExit("Installed torch is CPU-only; CUDA mode requires an Alliance CUDA-capable torch wheel.")
PY
)" || fail "PyTorch validation failed."

stage "Uni-Core build prerequisite diagnostic"
export FLUORCAST_UNIPROP_UNICORE_DIR="$UNICORE_DIR"
export FLUORCAST_UNIPROP_ENABLE_CUDA_EXT="$ENABLE_CUDA_EXT"
python_here "$(cat <<'PY'
import os
import sys
from importlib import metadata
from pathlib import Path

import numpy
import pip
import setuptools
import torch
import wheel

pip_executable = Path(sys.executable).with_name("pip")
print("Uni-Core build prerequisite diagnostic:")
print(f"  Python executable: {sys.executable}")
print(f"  pip executable: {pip_executable}")
print(f"  pip version: {pip.__version__}")
print(f"  setuptools version: {setuptools.__version__}")
print(f"  wheel version: {metadata.version('wheel')}")
print(f"  NumPy version: {numpy.__version__}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  PyTorch compiled CUDA version: {getattr(torch.version, 'cuda', None)}")
print(f"  Uni-Core source directory: {os.environ['FLUORCAST_UNIPROP_UNICORE_DIR']}")
print("  Build isolation disabled: yes")
print(f"  Optional CUDA extensions requested: {os.environ['FLUORCAST_UNIPROP_ENABLE_CUDA_EXT'] == '1'}")
if os.environ.get("FLUORCAST_UNIPROP_BOOTSTRAP_MODE") == "cuda" and getattr(torch.version, "cuda", None) is None:
    raise SystemExit("Installed torch is CPU-only; CUDA mode requires an Alliance CUDA-capable torch wheel.")
PY
)" || fail "Uni-Core build prerequisites are unavailable."

stage "Uni-Core direct install"
UNICORE_INSTALL_ARGS=("$UNICORE_DIR")
if [[ "$ENABLE_CUDA_EXT" -eq 1 ]]; then
    UNICORE_INSTALL_ARGS+=("--config-settings=--build-option=--enable-cuda-ext")
    echo "Optional Uni-Core fused CUDA extensions requested."
else
    echo "Optional Uni-Core fused CUDA extensions are not requested during bootstrap."
    echo "For a compute-node CUDA-extension build, rerun after loading CUDA/nvcc with --enable-cuda-ext."
fi
run_cmd "$VENV_PYTHON" -m pip install --no-build-isolation "${UNICORE_INSTALL_ARGS[@]}" || fail "Uni-Core installation failed."

stage "Uni-Core import validation"
python_here "$(cat <<'PY'
import importlib.util
import unicore
optional = [
    "unicore_fused_rounding",
    "unicore_fused_multi_tensor",
    "unicore_fused_adam",
    "unicore_fused_softmax_dropout",
    "unicore_fused_layernorm",
    "unicore_fused_layernorm_backward_gamma_beta",
]
print(f"Uni-Core import path: {getattr(unicore, '__file__', None)}")
for name in optional:
    status = "built" if importlib.util.find_spec(name) is not None else "not built"
    print(f"Optional Uni-Core extension {name}: {status}")
PY
)" || fail "Uni-Core required imports are unavailable."

stage "Uni-Mol+ direct install"
run_cmd "$VENV_PYTHON" -m pip install -e "$UNIMOL_PLUS_DIR" || fail "Uni-Mol+ installation failed."

stage "Final import diagnostic"
export FLUORCAST_UNIPROP_UPSTREAM_DIR="$UPSTREAM_DIR"
export FLUORCAST_UNIPROP_PINNED_COMMIT="$PINNED_COMMIT"
python_here "$(cat <<'PY'
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import unicore
import unimol_plus
from unimol_plus.models.uniprop import UniPropModel

upstream_dir = Path(os.environ["FLUORCAST_UNIPROP_UPSTREAM_DIR"])
pinned_commit = os.environ["FLUORCAST_UNIPROP_PINNED_COMMIT"]
actual_commit = subprocess.check_output(["git", "-C", str(upstream_dir), "rev-parse", "HEAD"], text=True).strip()
if actual_commit != pinned_commit:
    raise SystemExit(f"Pinned upstream commit mismatch: expected {pinned_commit}, got {actual_commit}")
schema_path = Path("configs/uniprop/feature_schema.json")
schema_hash = hashlib.sha256(schema_path.read_bytes()).hexdigest()
expected_schema_hash = "93e2a5aaf19617b7420a0020cea3c4d5a8550680fe4d2fd410b16d17081577f8"
if schema_hash != expected_schema_hash:
    raise SystemExit(f"Feature-schema hash mismatch: expected {expected_schema_hash}, got {schema_hash}")
model_module = importlib.import_module(UniPropModel.__module__)
print("Bootstrap diagnostic:")
print(f"  Python executable: {sys.executable}")
print(f"  Python version: {sys.version.split()[0]}")
print(f"  Environment path: {sys.prefix}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  Torch compiled CUDA version: {getattr(torch.version, 'cuda', None)}")
print(f"  CUDA available at runtime: {torch.cuda.is_available()}")
print(f"  Uni-Core import path: {getattr(unicore, '__file__', None)}")
print(f"  Uni-Mol+ import path: {getattr(unimol_plus, '__file__', None)}")
print(f"  Real UniProp model class: {UniPropModel.__module__}.{UniPropModel.__name__}")
print(f"  Real UniProp model source path: {getattr(model_module, '__file__', None)}")
print(f"  Pinned upstream Git commit: {actual_commit}")
print(f"  Feature-schema SHA-256 verified: {schema_hash}")
PY
)" || fail "Final UniProp import diagnostic failed."

if [[ "$DRY_RUN" -eq 1 ]]; then
    write_json_report "dry-run" "planned bootstrap actions only"
    echo
    echo "UniProp bootstrap dry-run completed."
else
    write_json_report "ok" "bootstrap completed"
    echo
    echo "UniProp bootstrap completed. Activate with: source $VENV_DIR/bin/activate"
fi
