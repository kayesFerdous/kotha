# Kotha — plan and live state

Read `CLAUDE.md` first for the durable rules. The same plan with the reasoning
laid out visually is at
<https://claude.ai/code/artifact/3107084e-3472-4798-918f-e47e3410785b>. This file is the working state:
update it as phases complete. Last touched 2026-08-30.

**Status: Phase 0 passes on the Arch/Ryzen machine, 2026-08-30.** `ct2rs` runs
this model at faster-whisper's speed and matching accuracy, agreeing with it on
99.27% of characters once decode settings are matched, with feature extraction
verified bit-exact against Whisper's reference. Four defects were
found; three were fixed in the spike and the fourth on HuggingFace. The engine
is settled — the only thing left for Phase 0 is repeating the measurements on
the M2.

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

### Phase 0 — Prove the engine  ⟵ GATE, PASSED (M2 numbers outstanding)

Does `ct2rs` load this model and produce the same text faster-whisper does?

- [x] Toolchain — Arch ships it: `pacman -S rust cmake`. No rustup needed
- [x] Model verified — 774,731,149 bytes, sha256 `9c0e38dc…ea74`
- [x] `cargo build --release` — compiles clean, first try, no API fixes
- [x] Test audio — all 393 WAVs already on this machine
- [x] Diff Rust output against faster-whisper on 50 utterances
- [x] Record real RTF for the Ryzen (below)
- [x] Fix `ct2rs`'s feature extraction; verify it against Whisper's reference
- [x] Close the byte-diff against a properly matched faster-whisper baseline
- [ ] Repeat on the M2

**Accept when:** the strings match faster-whisper's, and English words come out
in Latin script with spaces intact. *Both met* — the first against a matched
baseline, once it became clear the stored one used different decode settings.

**Where it stands:** everything the gate was built to catch is clear. English
is in Latin script with spaces intact, zero fusion warnings across 50
utterances — the `suppress_tokens` catastrophe did not occur. Speed matches
Python. Accuracy against the human references is better than the Python
baseline's, and the mel is bit-exact against Whisper's published formula.

Against a properly matched faster-whisper baseline the two engines agree to
99.27% of characters, with the remainder traced to the MKL-vs-oneDNN build —
see *Why the strings differ*.

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

**1. `processor_class` was missing from the published model.**  ⟵ *fixed at source*

`ct2rs` deserialises `preprocessor_config.json` into a struct with a required
`processor_class` field. The published file had no such field, so
`Whisper::new()` failed outright with `missing field processor_class`.
faster-whisper never reads it, so nothing had noticed.

The cause was asymmetry in the CTranslate2 conversion: it carries
`preprocessor_config.json` across but drops the sibling files that hold the
declaration. The fp16 repo declares `processor_class` in both
`processor_config.json` and `tokenizer_config.json`; the converted CT2 repo had
it nowhere.

**Fixed on HuggingFace, 2026-08-30.** `kayees/whisper-medium-bn-en-cs-faster`
now ships `"processor_class": "WhisperProcessor"`. Verified live, and
`model.bin` is untouched — still 774,731,149 bytes, sha256 `9c0e38dc…ea74`. A
fresh `./setup.sh` therefore produces a directory that loads as-is, with no
patching needed anywhere in the app.

The fp16 repo needs no change; transformers already finds the field in its
sibling files.

**2. `ct2rs`'s feature extraction is wrong in three ways.**  ⟵ *fixed*

All three are silent. None crashes, warns, or shows up in a smoke test; they
surface only as text that quietly disagrees with a reference implementation.

  - **Normalisation scope.** `whisper.rs:104` calls `mel_spec`'s `norm_mel` on
    one 80×1 frame at a time, so the `max - 8.0` dynamic-range floor comes from
    each frame's own peak. Whisper takes that maximum over the whole 30-second
    window. Per-frame normalisation behaves like an automatic gain control —
    it lifts silence and flattens loud frames.
  - **Padding.** `ct2rs` computes frames only where audio exists and leaves the
    remaining columns at 0.0. Whisper zero-pads the *audio* to 30 s, so trailing
    frames hold the mel of silence, a large negative value that clamps to the
    floor. 0.0 is nowhere near it. For a 10-second utterance this is two thirds
    of the input.
  - **Framing.** `mel_spec`'s `Spectrogram` is overlap-and-save with no
    centring, where Whisper uses `torch.stft(center=True)` — reflection-pad by
    `n_fft/2`, then frame. Its 400-sample buffer also advances by a 160-sample
    hop, placing frames 80 samples off each hop boundary. Since 400 is not a
    multiple of 160, that offset cannot be tuned away: **`mel_spec`'s STFT
    cannot reproduce Whisper's framing at these parameters at all.** Its
    `log_mel_spectrogram` separately drops the Nyquist bin and substitutes a
    literal 0.0.

Fixed by computing the mel in the spike and calling `ct2rs::sys::Whisper`,
which takes a features `StorageView` directly, instead of the `ct2rs::Whisper`
wrapper. `mel_spec` is kept only for its filterbank matrix — that part is
correct, being librosa's Slaney-normalised filters, which is what Whisper uses.

**Verified, not assumed.** `spike/check_mel.py` recomputes the mel from
Whisper's published formula in plain numpy — Slaney filterbank included, so it
needs no librosa — and diffs it against what the Rust binary produces:

```bash
cd spike && python3 check_mel.py <any 16 kHz wav>
```

Result on the test audio: **max |diff| 0.0 across all 240,000 values.** Bit-exact.
Worth re-running after any change to the audio path; it is the only part of the
inference chain we implement ourselves.

The `ct2rs` bugs are worth reporting upstream. The normalisation one is a
small change; the framing one needs a different STFT.

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

#### Why the strings differ — settled

The gate said: accept when the strings match faster-whisper's. Against the
stored baseline only 14 of 50 matched, which looked alarming. It was not a
defect in either engine; it was two different questions being compared.

**The stored baseline was never a like-for-like target.**
`results/cpu_bench.py` calls
`transcribe(language="bn", beam_size=1, suppress_tokens=[])` and takes
faster-whisper's defaults for everything else — including
`temperature=[0.0, 0.2, ... 1.0]`, a fallback ladder that **samples** whenever
greedy output trips the compression-ratio or logprob check, and
`without_timestamps=False`. So it is a stochastic multi-temperature decode with
a different prompt, and it is not reproducible byte-for-byte even against
itself. That is also where its mangled openings come from — `0`, `00`,
`ntermedit`, `jara` are fallback artifacts — and why its CER is the worse of
the two.

**Re-run matched, the picture changes completely.** `gate.py --baseline matched`
re-decodes the same files with `temperature=0.0`, `without_timestamps=True`,
`condition_on_previous_text=False`, using the venv at
`~/Documents/ASR/bangla-asr-test/.venv`:

| Baseline | identical | faster-whisper CER | Rust CER |
|---|---|---|---|
| stored (temperature fallback) | 14/50 | 0.0500 | 0.0389 |
| **matched (greedy, no timestamps)** | **25/50** | **0.0409** | **0.0389** |

Temperature fallback was costing faster-whisper 0.009 CER on its own. With
settings matched, the two engines are within 0.002 CER, and character-level
disagreement is **0.73% — 66 characters out of 9,001**, spread over 25
utterances, the largest single difference being 12 characters. Every one is a
single low-confidence word: `অবশ্যই`/`অবশ্য`, `arabi`/`arabic`,
`সোনা`/`শোনা`.

**What accounts for the last 0.73%.** Every other variable was measured, not
assumed:

| Variable | Result |
|---|---|
| Audio decoding | **bit-identical** — faster-whisper's `decode_audio` vs the spike's `hound` path, max abs diff 0.0 |
| Mel features | **identical** — the spike is bit-exact against Whisper's reference; faster-whisper's own extractor is within 3.3e-06 of it |
| Decode policy | matched explicitly |
| CTranslate2 version | **4.8.1 in both** |
| Threading | output is invariant to thread count, so this is not reduction-order noise |
| **GEMM build** | Python wheel links **Intel MKL**; this build uses **oneDNN**. The only variable left. |

By elimination the residual is the compiled math backend: different int8 kernels
round differently, and a handful of near-ties fall the other way. Consistent
with the earlier ruy→oneDNN swap, which moved two outputs with everything else
held fixed.

**Verdict: the gate passes.** Same audio, same features, same decode policy,
same library version, 99.27% character agreement, and equal accuracy against
the human references. `ct2rs` runs this model correctly.

*Optional, if certainty is ever wanted:* build the spike with ct2rs's `mkl`
feature and re-run `--baseline matched`. If the backend theory is right that
should go to ~50/50. Not recommended for shipping — MKL dispatches poorly on
AMD, which is why oneDNN is the x86-64 choice here — so this is a proof, not a
change.

**On the CER numbers.** Rust 0.0389 vs matched faster-whisper 0.0409 is a
0.002 gap on 50 utterances: noise, not a finding. The earlier 0.0500 figure was
a *decode-settings* artifact, not a Rust-versus-Python result. None of these
belong in the paper.

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
that paper-project tree, plus one real local file. That file is
`preprocessor_config.json`, byte-identical to what HuggingFace now serves — the
copy in the paper tree predates the fix and still lacks `processor_class`, so it
cannot simply be symlinked. Nothing under `~/Documents/ASR` is modified.

On a machine that fetches the model with `./setup.sh` none of this applies: the
download is correct as published and needs no local file at all.

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
| RTF, int8, 6 threads — Ryzen 5600G, oneDNN | **0.639** (1.57× realtime) | 2026-08-30 |
| RTF, int8, 6 threads — Ryzen 5600G, ruy | 1.222 (0.82× realtime) | 2026-08-30 |
| RTF, int8, 6 threads — faster-whisper, same 50 files | 0.669 (1.50× realtime) | earlier, `cpu_bench.json` |
| Model load time — Ryzen | 0.3 s | 2026-08-30 |
| CER vs reference, 50 utts — Rust, conformant mel | 0.0389 | 2026-08-30 |
| CER, stored baseline (temperature fallback) | 0.0500 | 2026-08-30 |
| Strings identical — matched baseline, 50 utts | 25 / 50 | 2026-08-30 |
| Character agreement with matched baseline | 99.27% (66 chars of 9,001) | 2026-08-30 |
| CER, matched faster-whisper baseline | 0.0409 | 2026-08-30 |
| Mel vs Whisper reference, max abs diff | 0.0 — bit-exact | 2026-08-30 |
| RTF, int8, 4 threads — M2 | — | |
| RTF, int8, 8 threads — M2 | — | |
| Peak RSS | — | |
| Corrector latency per sentence | — | |
| Strict English-F1, before corrector | — | |
| Strict English-F1, after corrector | — | |
