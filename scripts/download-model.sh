#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="app/src/main/assets"
MODEL_PATH="$MODEL_DIR/efficientdet-lite2.tflite"
MODEL_URL="https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/metadata/1?lite-format=tflite"

mkdir -p "$MODEL_DIR"
curl --fail --location --retry 3 --retry-delay 2 --connect-timeout 20 "$MODEL_URL" --output "$MODEL_PATH"

if [ ! -s "$MODEL_PATH" ]; then
  echo "XATOLIK: model fayli yuklanmadi yoki bo‘sh." >&2
  exit 1
fi

MODEL_SIZE=$(wc -c < "$MODEL_PATH")
if [ "$MODEL_SIZE" -lt 1000000 ]; then
  echo "XATOLIK: model fayli kutilgan hajmdan kichik ($MODEL_SIZE bayt)." >&2
  exit 1
fi

MODEL_MAGIC=$(dd if="$MODEL_PATH" bs=1 skip=4 count=4 2>/dev/null)
if [ "$MODEL_MAGIC" != "TFL3" ]; then
  echo "XATOLIK: yuklangan fayl TensorFlow Lite modeli emas (TFL3 imzosi topilmadi)." >&2
  exit 1
fi

sha256sum "$MODEL_PATH"
echo "Model tekshiruvdan o‘tdi: $MODEL_PATH ($MODEL_SIZE bayt)"
