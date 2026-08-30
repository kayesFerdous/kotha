#!/usr/bin/env bash
#
# Kotha — one-time machine setup.
#
# Installs the Rust toolchain and cmake, then downloads the model.
# Every step is idempotent: safe to re-run.
#
# The heavy steps refuse to run on battery. That is deliberate — compiling
# CTranslate2 and pulling 778 MB will drain a laptop. Plug in, or pass --force
# if you know what you are doing.
#
#   ./setup.sh                 # everything
#   ./setup.sh --skip-model    # toolchain only
#   ./setup.sh --force         # ignore the battery guard

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$ROOT/models/whisper-medium-bn-en-cs-faster"
HF_REPO="kayees/whisper-medium-bn-en-cs-faster"
HF_BASE="https://huggingface.co/$HF_REPO/resolve/main"

# model.bin is 775 MB; everything else is small.
MODEL_FILES=(model.bin tokenizer.json vocabulary.json config.json preprocessor_config.json)

FORCE=0
SKIP_MODEL=0
for arg in "$@"; do
  case "$arg" in
    --force)      FORCE=1 ;;
    --skip-model) SKIP_MODEL=1 ;;
    -h|--help)    sed -n '3,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m▸ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

on_battery() {
  [[ "$(uname)" == "Darwin" ]] || return 1
  pmset -g batt 2>/dev/null | grep -q "'Battery Power'"
}

require_power() {
  if on_battery && [[ $FORCE -eq 0 ]]; then
    local pct
    pct=$(pmset -g batt | grep -Eo '[0-9]+%' | head -1)
    die "On battery (${pct:-?}) — refusing to $1.

    This step is a long compile or a large download and will drain the laptop.
    Plug in and re-run, or pass --force to override."
  fi
}

# ---------------------------------------------------------------- toolchain

say "Rust toolchain"
if command -v cargo >/dev/null 2>&1; then
  ok "already installed — $(rustc --version)"
else
  require_power "install Rust"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
  ok "installed $(rustc --version)"
  warn "add this to your shell profile:  source \$HOME/.cargo/env"
fi

say "cmake  (ct2rs builds CTranslate2 from source and needs it)"
if command -v cmake >/dev/null 2>&1; then
  ok "already installed — $(cmake --version | head -1)"
elif command -v brew >/dev/null 2>&1; then
  brew install cmake            # bottled, no compile, safe on battery
  ok "installed $(cmake --version | head -1)"
else
  die "cmake missing and no Homebrew found. Install cmake, then re-run."
fi

# -------------------------------------------------------------------- model

if [[ $SKIP_MODEL -eq 1 ]]; then
  say "Model — skipped (--skip-model)"
  exit 0
fi

say "Model  ($HF_REPO, 778 MB)"
mkdir -p "$MODEL_DIR" "$ROOT/samples"

missing=()
for f in "${MODEL_FILES[@]}"; do
  [[ -s "$MODEL_DIR/$f" ]] || missing+=("$f")
done

if [[ ${#missing[@]} -eq 0 ]]; then
  ok "already present in $MODEL_DIR"
else
  require_power "download ${#missing[@]} model file(s)"
  for f in "${missing[@]}"; do
    printf '  ↓ %s\n' "$f"
    # -C - resumes a partial file, so an interrupted download is not wasted.
    curl -fL -C - --retry 3 --progress-bar -o "$MODEL_DIR/$f" "$HF_BASE/$f"
  done
  ok "downloaded to $MODEL_DIR"
fi

# Cheap sanity check: model.bin should be ~775 MB. A truncated download is the
# most likely failure here and it fails confusingly later.
size=$(wc -c < "$MODEL_DIR/model.bin" | tr -d ' ')
if [[ "$size" -lt 700000000 ]]; then
  die "model.bin is only $((size/1024/1024)) MB — expected ~775 MB.
    The download was truncated. Delete it and re-run:
      rm '$MODEL_DIR/model.bin' && ./setup.sh"
fi
ok "model.bin $((size/1024/1024)) MB"

say "Done"
cat <<EOF

  Model:  $MODEL_DIR

  Next — the Phase 0 gate. Put a few 16 kHz mono WAVs in ./samples/, then:

    cd spike
    cargo run --release -- "$MODEL_DIR" ../samples/*.wav

  The first build compiles CTranslate2 from source. Expect 15–30 minutes,
  and stay plugged in.

EOF
