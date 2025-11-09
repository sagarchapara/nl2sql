#!/usr/bin/env bash
set -euo pipefail

print_usage() {
  cat <<USAGE
Usage: $0 [output-directory]

Download and unpack the BIRD SQL dataset (train + evaluation splits).

Environment variables:
  BIRD_URLS   Space-separated list of mirrors to try before falling back to the built-in defaults.
              Each URL should point to a zip archive that expands to the official BIRD release.

If no output directory is supplied the data are placed in data/bird.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  print_usage
  exit 0
fi

OUTPUT_DIR=${1:-data/bird}

IFS=' ' read -r -a CUSTOM_URLS <<< "${BIRD_URLS:-}"
readonly DEFAULT_URLS=(
  "https://bird-bench.github.io/static/datasets/bird-v1.1.zip"
  "https://huggingface.co/datasets/bird-bench/bird/resolve/main/bird-v1_1.zip"
  "https://huggingface.co/datasets/bird-bench/bird/resolve/main/bird.zip"
)

URLS=("${CUSTOM_URLS[@]}" "${DEFAULT_URLS[@]}")

mkdir -p "$OUTPUT_DIR"
WORK_DIR=$(mktemp -d)
cleanup() {
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

ARCHIVE="$WORK_DIR/bird.zip"
download_completed=0
for url in "${URLS[@]}"; do
  [[ -z "$url" ]] && continue
  echo "Attempting to download BIRD dataset from: $url"
  if curl -fL "$url" -o "$ARCHIVE"; then
    download_completed=1
    break
  else
    echo "Download failed for $url" >&2
  fi
  rm -f "$ARCHIVE"
done

if [[ $download_completed -eq 0 ]]; then
  echo "ERROR: Unable to download the BIRD dataset. Please set BIRD_URLS to a working mirror." >&2
  exit 1
fi

echo "Unpacking archive into $OUTPUT_DIR"
unzip -q -o "$ARCHIVE" -d "$OUTPUT_DIR"

echo "BIRD dataset downloaded to $OUTPUT_DIR"
