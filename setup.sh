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
#   ./setup.sh --bench         # sweep decode thread counts, then stop
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
BENCH=0
for arg in "$@"; do
  case "$arg" in
    --force)      FORCE=1 ;;
    --skip-model) SKIP_MODEL=1 ;;
    --bench)      BENCH=1 ;;
    -h|--help)    sed -n '3,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
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

# -------------------------------------------------------------------- bench
#
# What this answers: how many threads CTranslate2 should get on this machine.
#
# It exists because the honest answer is not derivable. On the Ryzen 5600G six
# threads beat twelve, so SMT hurt. An M-series chip has no SMT and a different
# problem instead — four performance cores and four efficiency cores, where the
# efficiency ones are roughly a third of the speed and CTranslate2 joins at
# every layer, so the fast threads finish and wait. `decode_threads()` now
# defaults to the performance-core count for that reason. This is the sweep
# that says whether that was right.
#
# Peak RSS comes along for free and is the other number worth having: the
# Ryzen's 1.4 GB is the figure to beat on a 16 GB laptop.

bench() {
  require_power "run the benchmark — it builds CTranslate2 and decodes audio"

  [[ -s "$MODEL_DIR/model.bin" ]] ||
    die "No model at $MODEL_DIR.
    Run ./setup.sh first."

  shopt -s nullglob
  local wavs=("$ROOT"/samples/*.wav)
  shopt -u nullglob
  [[ ${#wavs[@]} -gt 0 ]] ||
    die "No WAVs in $ROOT/samples/.
    The sweep needs audio to decode — 16 kHz mono, and the same files every
    time or the numbers are not comparable across runs. A few minutes of
    speech is plenty."

  # The sweep points. 1 and 2 show how the curve starts, then the two that
  # matter: the performance-core count and every physical core. On a machine
  # with one performance level these collapse and the duplicates drop out.
  local p_cores t_cores
  if [[ "$(uname)" == "Darwin" ]]; then
    p_cores=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu)
    t_cores=$(sysctl -n hw.physicalcpu)
    say "Machine"
    ok "$(sysctl -n machdep.cpu.brand_string) — ${p_cores} performance core(s), \
$(( t_cores - p_cores )) efficiency core(s)"
  else
    p_cores=$(nproc --all)
    t_cores=$p_cores
  fi

  local counts
  counts=$(printf '%s\n' 1 2 "$p_cores" $(( p_cores + 2 )) "$t_cores" |
           awk -v max="$t_cores" '$1 >= 1 && $1 <= max' | sort -n -u)

  # Honour CARGO_TARGET_DIR, because the Accelerate comparison below depends on
  # it: building into a second directory and then running the binary from the
  # first would benchmark the old build and say nothing about it.
  local target_dir bin
  target_dir="${CARGO_TARGET_DIR:-$ROOT/target}"
  [[ "$target_dir" = /* ]] || target_dir="$ROOT/$target_dir"
  bin="$target_dir/release/kotha-spike"

  say "Building (CTranslate2 from source the first time — 15–30 minutes)"
  ( cd "$ROOT" && cargo build --release -p kotha-spike ) || die "build failed"
  [[ -x "$bin" ]] || die "built, but no binary at $bin"
  ok "built  $bin"

  say "Sweeping ${#wavs[@]} file(s) at: $(echo $counts | tr '\n' ' ')"
  printf '\n  %-9s %-9s %-12s %s\n' threads RTF realtime "peak RSS"
  printf '  %-9s %-9s %-12s %s\n' ------- ------- -------- --------

  # The transcripts are kept, not thrown away. A sweep that reports RTF and
  # deletes the text it decoded cannot answer the question that actually
  # matters — whether the fast run was also a correct one — and the
  # suppress_tokens fusion warning the gate prints would go with it. They land
  # under the target directory, so the Accelerate comparison keeps its own set.
  local logdir log rtf rss n
  logdir="$target_dir/bench-logs"
  mkdir -p "$logdir"

  for n in $counts; do
    log="$logdir/threads-$n.log"
    # /usr/bin/time -l reports peak RSS in bytes on macOS; GNU time uses -v and
    # kilobytes, so this is read back defensively rather than assumed.
    KOTHA_THREADS="$n" /usr/bin/time -l \
      "$bin" "$MODEL_DIR" "${wavs[@]}" \
      >"$log" 2>&1 || { warn "$n threads: run failed — see $log"; continue; }

    rtf=$(grep -Eo '^RTF [0-9.]+' "$log" | awk '{print $2}')
    rss=$(grep -Eo '[0-9]+ +maximum resident set size' "$log" | awk '{print $1}')
    printf '  %-9s %-9s %-12s %s\n' \
      "$n" "${rtf:-?}" \
      "$(awk -v r="${rtf:-0}" 'BEGIN{ if (r>0) printf "%.2fx", 1/r; else print "?" }')" \
      "$(awk -v b="${rss:-0}" 'BEGIN{ if (b>0) printf "%.2f GB", b/1073741824; else print "?" }')"
  done
  ok "transcripts kept in $logdir"

  say "Reading it"
  cat <<'EOF'

  Lowest RTF wins; it is decode seconds per audio second, so under 1.0 is
  faster than real time. Put the winner in decode_threads() — and if it is not
  the performance-core count, say so in the commit, because that is the
  assumption the default was built on.

  The other half of the M-series question is the GEMM backend, and it needs a
  second build rather than a second run. Apple Silicon currently compiles ruy
  *and* Accelerate: ruy does int8, Accelerate does float32. To measure what
  Accelerate is worth, comment out the aarch64-apple-darwin block in
  spike/Cargo.toml, build into a separate directory so neither result clobbers
  the other, and sweep again:

    CARGO_TARGET_DIR=target/no-accel ./setup.sh --bench

  Do not instead swap Accelerate in for ruy. Accelerate serves float32 only,
  so without ruy there is no int8 backend at all and the model silently
  resolves to float32. Measured on the Ryzen, that is 1.9x slower and 2.6x the
  memory, with nothing printed to say so.

EOF
}

if [[ $BENCH -eq 1 ]]; then
  bench
  exit 0
fi

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
    # HuggingFace resets long connections often enough that a single curl is
    # not reliable for a 775 MB file — we lost one at 226 MB. -C - resumes from
    # whatever is already on disk, so each attempt picks up where the last one
    # died rather than starting over.
    for attempt in 1 2 3 4 5 6; do
      if curl -fL -C - --retry 5 --retry-delay 3 --retry-all-errors \
              --progress-bar -o "$MODEL_DIR/$f" "$HF_BASE/$f"; then
        break
      fi
      # curl exits 33 when the server will not honour a range request and 22
      # when the file is already complete; both mean "stop retrying".
      rc=$?
      [[ $rc -eq 33 || $rc -eq 22 ]] && break
      warn "attempt $attempt failed (curl $rc) — resuming"
      [[ $attempt -eq 6 ]] && die "gave up on $f after 6 attempts"
      sleep 3
    done
  done
  ok "downloaded to $MODEL_DIR"
fi

# Verify against HuggingFace's own manifest rather than a hard-coded guess.
# A truncated or badly-resumed model.bin fails much later and very confusingly,
# and eyeballing the size does not catch a corrupt resume. Note HF reports
# decimal MB, so the "775 MB" on the web page is 774,731,149 bytes — do not
# compare it against a MiB figure.
say "Verifying"
manifest=$(curl -sS "https://huggingface.co/api/models/$HF_REPO/tree/main")

expected_size=$(printf '%s' "$manifest" | python3 -c "
import sys, json
print(next(e['size'] for e in json.load(sys.stdin) if e['path'] == 'model.bin'))")
actual_size=$(wc -c < "$MODEL_DIR/model.bin" | tr -d ' ')

if [[ "$actual_size" != "$expected_size" ]]; then
  die "model.bin is $actual_size bytes, expected $expected_size.
    The download is incomplete. Re-run ./setup.sh — curl resumes from where it
    stopped, so nothing already fetched is wasted."
fi
ok "model.bin $actual_size bytes — size matches"

expected_sha=$(printf '%s' "$manifest" | python3 -c "
import sys, json
print(next((e.get('lfs') or {}).get('oid', '') for e in json.load(sys.stdin) if e['path'] == 'model.bin'))")

if [[ -n "$expected_sha" ]]; then
  printf '  … hashing 774 MB, this takes a few seconds\n'
  actual_sha=$(shasum -a 256 "$MODEL_DIR/model.bin" | awk '{print $1}')
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    die "model.bin sha256 does not match HuggingFace.
      expected $expected_sha
      actual   $actual_sha
    A resume corrupted it. Delete and re-fetch:
      rm '$MODEL_DIR/model.bin' && ./setup.sh"
  fi
  ok "sha256 verified"
fi

say "Done"
cat <<EOF

  Model:  $MODEL_DIR

  Next — the Phase 0 gate. Put a few 16 kHz mono WAVs in ./samples/, then:

    cd spike
    cargo run --release -- "$MODEL_DIR" ../samples/*.wav

  The first build compiles CTranslate2 from source. Expect 15–30 minutes,
  and stay plugged in.

EOF
