# Kotha — plan and live state

Read `CLAUDE.md` first for the durable rules. The same plan with the reasoning
laid out visually is at
<https://claude.ai/code/artifact/3107084e-3472-4798-918f-e47e3410785b>. This file is the working state:
update it as phases complete. Last touched 2026-08-30.

**Status: Phase 0 run on the Arch/Ryzen machine, 2026-08-30. The engine works.**
`ct2rs` loads the published model and decodes at faster-whisper's speed. Three
defects were found and two are fixed; the third (§ *The mel bug*) is diagnosed
and not yet fixed. Phase 0 is not signed off until it is.

---

## How the app works

```
hotkey  →  mic 16 kHz  →  split on silence  →  transcribe  →  fix English  →  paste
 rdev        cpal            earshot           ct2rs        symspell      arboard
                                                                          + enigo
```

Press the hotkey. A small pill fades in with a live waveform. Talk. Each time
you pause, that chunk is transcribed and the text lands at the cursor — so words
appear while you are still speaking. Press again, or stay silent for two
seconds, and it fades out.

Chunking on pauses is not cosmetic. At ~1.5× real time, waiting for the whole
utterance would leave seconds of dead air. Sentence-sized chunks also match what
the model was trained on.

---

## Phases

Each phase has one acceptance test. Do not start the next phase until it passes.

### Phase 0 — Prove the engine  ⟵ GATE, nearly closed

Does `ct2rs` load this model and produce the same text faster-whisper does?

- [x] Toolchain — Arch ships it: `pacman -S rust cmake`. No rustup needed
- [x] Model verified — 774,731,149 bytes, sha256 `9c0e38dc…ea74`
- [x] `cargo build --release` — compiles clean, first try, no API fixes
- [x] Test audio — all 393 WAVs already on this machine
- [x] Diff Rust output against faster-whisper on 50 utterances
- [x] Record real RTF for the Ryzen (below)
- [ ] **Fix the mel normalisation bug, then re-diff** — the one thing left
- [ ] Repeat on the M2

**Accept when:** the strings match faster-whisper's, and English words come out
in Latin script with spaces intact.

**Where it stands:** the second half passes outright — English is in Latin
script, spaces intact, zero fusion warnings across 50 utterances. The
`suppress_tokens` catastrophe did not occur. Speed matches Python exactly. The
strings do *not* yet match, and the reason is understood and fixable.

**If it fails:** stop and reconsider. Fallback is a bundled Python sidecar
running faster-whisper — roughly +300 MB and worse packaging, but it works. This
gate exists so that decision costs two days, not six weeks.

**Why first:** everything after this is ordinary application work. This is the
only genuine unknown in the project.

#### What the gate found — 2026-08-30, Ryzen 5600G

Baseline for the diff is `~/Documents/ASR/results/cpu_bench.json`, 50 utterances
decoded by faster-whisper **on this same machine** with the identical params
(`language="bn", beam_size=1, suppress_tokens=[]`). Same files, same hardware,
so the comparison is clean.

**1. `processor_class` is missing from the published model.** `ct2rs`
deserialises `preprocessor_config.json` into a struct that requires a
`processor_class` field. The published file has no such field, so
`Whisper::new()` fails with `missing field processor_class`. faster-whisper
never needed it.

Worked around locally by writing a patched copy into
`models/whisper-medium-bn-en-cs-faster/` (the large files are symlinked to
`~/Documents/ASR/fine_tuned/whisper-medium-bn-v1.3-ct2-int8/`, which is *not*
edited — that tree belongs to the paper project).

**This will hit every user**, because the file on HuggingFace is the one that
lacks the field. Two ways out, and it is Kayes's call which:
  - add `"processor_class": "WhisperProcessor"` to the HF repo, or
  - have the app patch the file after download.

**2. The mel bug — `ct2rs` normalises per frame.**  ⟵ *unfixed, blocks the gate*

`ct2rs-0.9.22/src/whisper.rs:104` calls `norm_mel()` on one 80×1 frame at a
time. `norm_mel` clamps against `max - 8.0` computed over whatever array it is
handed — so each frame is normalised against *its own* peak. Whisper takes that
max over the entire 30-second spectrogram. The encoder is therefore fed
subtly wrong features everywhere.

This is why only 14 of 50 outputs match faster-whisper byte-for-byte. The
divergence clusters at utterance starts, where these chunks begin mid-word and
the decoder is least certain — faster-whisper emits `0`, `ntermedit`, `jara`
where Rust emits `করতে পারেন`, `intermediat`, `যারা`.

**The fix:** `pub mod sys` is public, and `sys::Whisper::generate()` takes a
features `StorageView` directly. So bypass the high-level wrapper: compute the
log-mel ourselves, apply the `max - 8.0` clamp once globally, hand over the
StorageView. `mel_spec` and `ndarray` are already in the tree — the filterbank
is `mel_spec::mel::mel(16000.0, 400, 80, None, None, false, true)`, which is
exactly what `ct2rs` builds internally. Roughly 60 lines, no new dependencies.

Worth reporting upstream; the accumulate-then-normalise change is small.

**3. Wrong GEMM backend cost 1.8×.**  ⟵ *fixed*

`ct2rs`'s default features are `["all-tokenizers", "ruy", "cuda-small-binary"]`.
`ruy` is Google's **ARM**-tuned int8 kernel — right on Apple Silicon, wrong on
x86-64, where the fast path is oneDNN. faster-whisper's PyPI wheel ships oneDNN
plus MKL, which is the whole of the gap:

| Build | RTF | vs realtime |
|---|---|---|
| Rust, `ruy` (default) | 1.222 | 0.82× |
| Rust, `dnnl` (oneDNN) | **0.660** | **1.52×** |
| faster-whisper (Python) | 0.669 | 1.50× |

`spike/Cargo.toml` now selects the backend per target — `dnnl` on x86-64, `ruy`
elsewhere. Costs 8m48s of build time and grows the binary 8.3 MB → 75.7 MB
(oneDNN is linked statically). That matters against the "~20 MB installers"
line in CLAUDE.md §3 and should be revisited at Phase 6; next to a 775 MB model
it is not the thing to optimise first.

**On accuracy — do not bank this yet.** Against the references, Rust scores
CER 0.0415 and faster-whisper 0.0500 on these 50 utterances. That is not a
claim of a better engine: 50 utterances is a small sample, and the most likely
explanation is that per-frame normalisation is acting as accidental AGC on
noisy YouTube audio. Expect it to converge toward the Python number once the
mel bug is fixed. Re-measure then.

---

### Phase 1 — The spelling corrector

Independent of the Rust work. Prototype in Python, port to Rust after.

- [ ] Re-decode the 393 test utterances to get hypothesis text
- [ ] Align English tokens against references; **split the 13.6-point
      strict/tolerant gap into *misspelled in Latin* vs *written in Bengali***
- [ ] Build the dictionary: `en_50k.txt` (OpenSubtitles — casual spoken English,
      the right register) + the English side of the corpus vocabulary
- [ ] SymSpell index (edit distance ≤ 2) + Double Metaphone fallback
- [ ] Tune the abstain threshold **on training-side data only**
- [ ] Port to Rust (~200 lines)

**Accept when:** strict English-F1 improves measurably on held-out data, and
zero non-Latin tokens are modified across the whole test set.

**Rule:** do not tune on the 393 test utterances. Hold them out, evaluate once.

---

### Phase 2 — The loop, no interface

- [ ] Hotkey → record → VAD → transcribe → clipboard
- [ ] Model stays warm in memory between presses
- [ ] Threads pinned to the measured-best count

**Accept when:** a terminal app prints what you said, twice in a row, with no
model reload between them.

---

### Phase 3 — Text at the cursor

The OS-specific part. Three mechanisms underneath.

- [ ] macOS: Accessibility permission, synthetic ⌘V
- [ ] Windows: SendInput
- [ ] Linux: X11 vs Wayland split
- [ ] **Copy-only mode that needs no permissions**, so the app is useful before
      anything is granted
- [ ] Restore the previous clipboard contents afterwards

**Accept when:** dictated text appears in TextEdit, a browser field, and a
terminal, without the focused window changing.

**Why clipboard and not keystrokes:** Bengali conjuncts and combining marks
break character-by-character injection in many apps. A paste is atomic.

---

### Phase 4 — The pill

- [ ] Frameless, transparent, always-on-top, **never takes focus**
- [ ] macOS needs an **NSPanel**, not a normal window — a normal one steals the
      cursor and the whole feature collapses (`tauri-nspanel`)
- [ ] Waveform driven by real mic levels streamed from Rust at 30 fps
- [ ] Four states: appearing, listening, thinking, done. CSS only
- [ ] Tray icon: Start, Settings, Quit

**Accept when:** the pill animates over a focused text field and the caret keeps
blinking in that field.

---

### Phase 5 — First run

Where "easy to start" is won or lost.

- [ ] Model download: 778 MB, resumable, progress bar, checksum
- [ ] Permission walkthrough with buttons that open the right settings panel
- [ ] A short "press this key and talk" demo
- [ ] Settings: hotkey, microphone, paste mode

**Accept when:** a fresh user account gets from downloaded installer to first
dictated sentence without reading anything.

---

### Phase 6 — Ship

- [ ] GitHub Actions matrix: `.dmg` (Intel + Apple Silicon), `.msi`,
      `.AppImage`, `.deb`
- [ ] Tauri updater (free, no certificate needed)
- [ ] One-page download site
- [ ] AUR entry

---

### Phase 7 — Polish

- [ ] Push-to-talk as an alternative to toggle
- [ ] History window
- [ ] Start/stop sounds
- [ ] Bigram context for the corrector (SymSpell supports it; still sub-ms)
- [ ] Per-app "never dictate here" list

---

## The two machines

Phase 0 was run on the **Arch / Ryzen 5600G desktop**, which turned out to have
everything already: `pacman -S rust cmake` (no rustup — Arch's `rust` package
ships cargo), the model at
`~/Documents/ASR/fine_tuned/whisper-medium-bn-v1.3-ct2-int8` verified byte-exact
against the release, all 393 test WAVs under `bangla-asr-test/chunks/test/`, and
a faster-whisper baseline in `results/cpu_bench.json`. Nothing needed
downloading and the "copy 20 WAVs over" step never applied.

`models/whisper-medium-bn-en-cs-faster/` here is a directory of symlinks into
that paper-project tree, plus one real local file — the patched
`preprocessor_config.json`. Nothing under `~/Documents/ASR` is modified.

**Still to do on the M2:** install the toolchain by hand (below), fetch the
model with `./setup.sh`, and re-measure. The numbers do not transfer — the
Ryzen is 6 cores / 12 SMT threads where 6 beat 12; the M2 is 4P + 4E with no
SMT, so sweep 4 against 8. The backend also differs: `ruy` is correct there and
`dnnl` is correct here, which `spike/Cargo.toml` now handles per target. Also
worth benchmarking `accelerate` against `ruy` on Apple Silicon.

The battery guard in `setup.sh` is macOS-only and a no-op on Linux, which is
right — the Arch box is a desktop.

**M2 toolchain**, run by hand (auto mode blocks package installs):

```bash
brew install cmake rustup
echo 'export PATH="/opt/homebrew/opt/rustup/bin:$PATH"' >> ~/.zshrc
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
rustup default stable
```

Two traps, both hit on 2026-08-30: Homebrew's `rustup` formula no longer ships
`rustup-init` (use `rustup default stable`), and it is keg-only, so without the
PATH line `rustup` is "command not found" even though it installed fine.
`rustup` beats `brew install rust` because Tauri needs per-target toolchains
later, and the two formulae conflict.

---

## Open decisions

| Question | State |
|---|---|
| Name | **Kotha (কথা)** — decided |
| Hotkey | Proposed `Ctrl+Alt+Space` on all platforms. Not final |
| Code signing | Undecided. $0 / $99 a year for macOS / ~$400 for both |
| Public repo | Undecided |

Signing is the one that changes how the first thirty seconds feel for every user.
macOS alone is the one worth buying if only one is bought.

---

## Measurements

Fill these in as they are taken. Do not carry over numbers from the paper
project — that hardware is a Ryzen 5600G, this is an M2.

| What | Value | Taken on |
|---|---|---|
| RTF, int8, 6 threads — Ryzen 5600G, oneDNN | **0.660** (1.52× realtime) | 2026-08-30 |
| RTF, int8, 6 threads — Ryzen 5600G, ruy | 1.222 (0.82× realtime) | 2026-08-30 |
| RTF, int8, 6 threads — faster-whisper, same 50 files | 0.669 (1.50× realtime) | earlier, `cpu_bench.json` |
| Model load time — Ryzen | 0.3 s | 2026-08-30 |
| CER vs reference, 50 utts — Rust, mel bug present | 0.0415 | 2026-08-30 |
| CER vs reference, 50 utts — faster-whisper | 0.0500 | 2026-08-30 |
| Strings identical to faster-whisper, 50 utts | 14 / 50 | 2026-08-30 |
| RTF, int8, 4 threads — M2 | — | |
| RTF, int8, 8 threads — M2 | — | |
| Peak RSS | — | |
| Corrector latency per sentence | — | |
| Strict English-F1, before corrector | — | |
| Strict English-F1, after corrector | — | |
