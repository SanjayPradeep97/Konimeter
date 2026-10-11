#!/usr/bin/env bash
# Launch the click-to-classify demo (Linux / macOS).
#   1. activates the cnt-vfm conda env if it is not already active
#   2. downloads the SAM ViT-B checkpoint (358 MB) into ../sam_weights once
#   3. starts the Gradio page at http://127.0.0.1:7861
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export PYTHONIOENCODING=utf-8 KMP_DUPLICATE_LIB_OK=TRUE

if [ "${CONDA_DEFAULT_ENV:-}" != "cnt-vfm" ]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate cnt-vfm || {
      echo "could not activate the conda env cnt-vfm."
      echo "create it first:   conda env create -f ../environment.yml   (or environment-cpu.yml)"
      exit 1
    }
  else
    echo "conda not found on PATH; activate an environment with the demo's dependencies first." >&2
  fi
fi

if [ ! -f ../sam_weights/sam_vit_b_01ec64.pth ]; then
  echo "downloading the SAM ViT-B checkpoint (358 MB, once) ..."
  python ../download_sam_weights.py --model vit_b
fi

exec python demo_app.py "$@"
