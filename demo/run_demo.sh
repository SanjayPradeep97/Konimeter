#!/usr/bin/env bash
# Start the segment-and-classify demo (Linux / macOS).
#   1. activates the cnt-vfm conda environment if it is not already active
#   2. starts the page at http://127.0.0.1:7861; the first time, the page
#      offers to download the SAM ViT-B checkpoint (358 MB)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE

if [ "${CONDA_DEFAULT_ENV:-}" != "cnt-vfm" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate cnt-vfm || {
      echo "could not activate the conda environment cnt-vfm."
      echo "create it first:   conda env create -f ../environment-cpu.yml   (or environment.yml with an NVIDIA GPU)"
      exit 1
    }
  else
    echo "conda not found on PATH; activate an environment with the demo's dependencies first." >&2
  fi
fi

exec python demo_app.py "$@"
