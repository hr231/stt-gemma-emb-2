#!/usr/bin/env bash
# Download + extract LibriSpeech splits in the OpenSLR layout:
#   <data_root>/LibriSpeech/<split>/<speaker>/<chapter>/*.flac (+ *.trans.txt)
#
# Usage:  bash download_librispeech.sh ~/gemma_stt/data train-clean-100 dev-clean
# Splits: train-clean-100 (6.3G) train-clean-360 (23G) train-other-500 (30G)
#         dev-clean dev-other test-clean test-other (~0.3G each)
# Other mirrors: MIRROR=https://www.openslr.org/resources/12  (or the EU mirror)
set -euo pipefail
ROOT="${1:?usage: download_librispeech.sh <data_root> <split> [split ...]}"; shift
MIRROR="${MIRROR:-https://us.openslr.org/resources/12}"
mkdir -p "$ROOT"

for s in "$@"; do
  if [ -f "$ROOT/.done_$s" ]; then echo "[skip] $s"; continue; fi
  echo "[get] $s from $MIRROR"
  curl -L -C - --retry 5 --fail -o "$ROOT/$s.tar.gz" "$MIRROR/$s.tar.gz"
  echo "[extract] $s"
  tar -xzf "$ROOT/$s.tar.gz" -C "$ROOT"
  rm "$ROOT/$s.tar.gz"
  touch "$ROOT/.done_$s"
done
echo "done -> $ROOT/LibriSpeech"
