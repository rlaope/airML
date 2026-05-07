#!/usr/bin/env bash
# refresh-registry.sh
#
# Downloads each model in the airml-hub registry, computes sha256 + size,
# then prints a Rust snippet to paste back into crates/airml-hub/src/registry.rs.
#
# Usage:
#   ./scripts/refresh-registry.sh           # download and hash all entries
#   ./scripts/refresh-registry.sh --dry-run # list URLs only, no downloads

set -uo pipefail

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Registry entries — keep in sync with crates/airml-hub/src/registry.rs
# Format: "id|hf_repo|file|description"
# ---------------------------------------------------------------------------
declare -a ENTRIES=(
    "bge-small-en|BAAI/bge-small-en-v1.5|onnx/model.onnx|BGE Small English v1.5 — compact general-purpose text embeddings (~133 MB)"
    "all-minilm-l6-v2|sentence-transformers/all-MiniLM-L6-v2|onnx/model.onnx|all-MiniLM-L6-v2 — fast sentence embeddings (~90 MB)"
    "clip-vit-b32|Xenova/clip-vit-base-patch32|onnx/model.onnx|CLIP ViT-B/32 — joint image + text embeddings (~605 MB)"
    "mobilenetv3-small|onnx/models|validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx|MobileNetV2 image classification (~14 MB)"
    "whisper-tiny-encoder|Xenova/whisper-tiny|onnx/encoder_model.onnx|Whisper Tiny encoder — speech feature extraction (~80 MB)"
)

TMP_PREFIX="/tmp/airml-refresh"
TMPFILES=()

# Cleanup on exit.
cleanup() {
    for f in "${TMPFILES[@]+"${TMPFILES[@]}"}"; do
        rm -f "$f"
    done
}
trap cleanup EXIT

# Detect sha256 command.
sha256_cmd() {
    local file="$1"
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file" | awk '{print $1}'
    elif command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" | awk '{print $1}'
    else
        echo "ERROR: neither shasum nor sha256sum found" >&2
        return 1
    fi
}

# Detect file size command.
file_size() {
    local file="$1"
    if command -v stat >/dev/null 2>&1; then
        # macOS stat uses -f %z; GNU stat uses -c %s
        stat -f %z "$file" 2>/dev/null || stat -c %s "$file" 2>/dev/null
    else
        wc -c < "$file" | tr -d ' '
    fi
}

if $DRY_RUN; then
    echo "=== Dry run — URLs that would be downloaded ==="
    for entry in "${ENTRIES[@]}"; do
        IFS='|' read -r id hf_repo file desc <<< "$entry"
        url="https://huggingface.co/${hf_repo}/resolve/main/${file}"
        echo "  [$id]  $url"
    done
    echo ""
    echo "Run without --dry-run to download and compute hashes."
    exit 0
fi

echo "=== Downloading and hashing registry entries ==="
echo ""

declare -a RESULTS=()

for entry in "${ENTRIES[@]}"; do
    IFS='|' read -r id hf_repo file desc <<< "$entry"
    url="https://huggingface.co/${hf_repo}/resolve/main/${file}"
    tmpfile="${TMP_PREFIX}-${id}.bin"
    TMPFILES+=("$tmpfile")

    echo "Downloading [$id]..."
    echo "  URL: $url"

    if ! curl -L --fail --silent --show-error -o "$tmpfile" "$url"; then
        echo "  FAILED: $url" >&2
        RESULTS+=("FAILED|${id}|${hf_repo}|${file}|${desc}")
        continue
    fi

    sha=$(sha256_cmd "$tmpfile") || { echo "  sha256 failed for $id" >&2; continue; }
    size=$(file_size "$tmpfile") || { echo "  stat failed for $id" >&2; continue; }

    echo "  sha256: $sha"
    echo "  size:   $size bytes"
    echo ""

    RESULTS+=("OK|${id}|${hf_repo}|${file}|${desc}|${sha}|${size}")
done

# ---------------------------------------------------------------------------
# Emit the Rust snippet.
# ---------------------------------------------------------------------------
echo "=== Rust snippet — paste into crates/airml-hub/src/registry.rs ==="
echo ""
echo "pub const REGISTRY: &[ModelEntry] = &["

for result in "${RESULTS[@]}"; do
    IFS='|' read -r status id hf_repo file desc rest <<< "$result"

    if [[ "$status" == "FAILED" ]]; then
        sha="PLACEHOLDER_TO_VERIFY"
        size=0
        echo "    // WARNING: download failed for $id"
    else
        IFS='|' read -r sha size <<< "$rest"
    fi

    cat <<RUST
    ModelEntry {
        id: "${id}",
        hf_repo: "${hf_repo}",
        file: "${file}",
        sha256: "${sha}",
        size_bytes: ${size},
        description: "${desc}",
    },
RUST
done

echo "];"
