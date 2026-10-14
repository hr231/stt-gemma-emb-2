#!/usr/bin/env bash
# One-time setup: a dedicated Python 3.12 venv (separate from ~/.venv-vllm-metal,
# whose transformers 5.9 is too old for EmbeddingGemma 2).
set -euo pipefail
VENV="${VENV:-$HOME/.venv-gemma-stt}"

python3.12 -m venv "$VENV"
"$VENV/bin/pip" install -U pip
"$VENV/bin/pip" install torch "transformers>=5.19.0" safetensors sentencepiece soundfile numpy pyarrow

"$VENV/bin/python" - << 'PY'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("MPS available:", torch.backends.mps.is_available())
PY
echo
echo "Activate with:  source $VENV/bin/activate"
