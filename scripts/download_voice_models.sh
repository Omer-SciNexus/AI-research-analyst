#!/usr/bin/env bash
# Download the voice models (Piper TTS + faster-whisper STT) into data/models.
# Resumable and stall-proof: safe to re-run after a dropped connection.
#
# Run:  bash scripts/download_voice_models.sh
set -euo pipefail
cd "$(dirname "$0")/.."
models=data/models
hf=https://huggingface.co

# fetch URL OUTPUT EXPECTED_BYTES
fetch() {
  local url="$1" out="$2" expected="$3" size
  mkdir -p "$(dirname "$out")"
  for attempt in $(seq 1 200); do
    size=$(stat -c %s "$out" 2>/dev/null || echo 0)
    if [ "$size" -ge "$expected" ]; then echo "ok    $out"; return 0; fi
    echo "get   $out ($size / $expected bytes, attempt $attempt)"
    # Drop a stalled connection (<1 KB/s for 60 s) instead of hanging forever.
    curl -L -sS -C - --connect-timeout 30 --speed-limit 1024 --speed-time 60 -o "$out" "$url" || sleep 5
  done
  echo "failed $out" >&2; return 1
}

piper="$hf/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low"
fetch "$piper/en_US-lessac-low.onnx" "$models/en_US-lessac-low.onnx" 63201294
fetch "$piper/en_US-lessac-low.onnx.json" "$models/en_US-lessac-low.onnx.json" 4882

# Multilingual, so speech in other languages can be translated to English
whisper="$hf/Systran/faster-whisper-small/resolve/main"
fetch "$whisper/config.json" "$models/faster-whisper-small/config.json" 2370
fetch "$whisper/tokenizer.json" "$models/faster-whisper-small/tokenizer.json" 2203239
fetch "$whisper/vocabulary.txt" "$models/faster-whisper-small/vocabulary.txt" 459861
fetch "$whisper/model.bin" "$models/faster-whisper-small/model.bin" 483546902
