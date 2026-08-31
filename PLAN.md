# Kotha — plan and live state

Read `CLAUDE.md` first for the durable rules. The same plan with the reasoning
laid out visually is at
<https://claude.ai/code/artifact/3107084e-3472-4798-918f-e47e3410785b>. This file is the working state:
update it as phases complete. Last touched 2026-08-30.

**Status: Phase 0 passes on the Arch/Ryzen machine, 2026-08-30.** `ct2rs` runs
this model at faster-whisper's speed and better accuracy, with feature
extraction verified bit-exact against Whisper's reference. Four defects were
found in `ct2rs`; all four are fixed here. One decision is open — see
*Why the strings still differ*. Phase 0 has not been repeated on the M2.

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

### Phase 0 — Prove the engine  ⟵ GATE, passed on evidence

Does `ct2rs` load this model and produce the same text faster-whisper does?

- [x] Toolchain — Arch ships it: `pacman -S rust cmake`. No rustup needed
- [x] Model verified — 774,731,149 bytes, sha256 `9c0e38dc…ea74`
- [x] `cargo build --release` — compiles clean, first try, no API fixes
- [x] Test audio — all 393 WAVs already on this machine
- [x] Diff Rust output against faster-whisper on 50 utterances
- [x] Record real RTF for the Ryzen (below)
- [x] Fix `ct2rs`'s feature extraction; verify it against Whisper's reference
- [ ] **Decide how to close the byte-diff** — see *Why the strings still differ*
- [ ] Repeat on the M2

**Accept when:** the strings match faster-whisper's, and English words come out
in Latin script with spaces intact.

**Where it stands:** everything the gate was built to catch is clear. English
is in Latin script with spaces intact, zero fusion warnings across 50
utterances — the `suppress_tokens` catastrophe did not occur. Speed matches
Python. Accuracy against the human references is better than the Python
baseline's, and the mel is bit-exact against Whisper's published formula.

The strings still do not match the stored baseline byte-for-byte, but that
baseline turned out not to be a like-for-like target — see below. The engine is
not in doubt; what to compare against is.

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

#### Why the strings still differ  ⟵ *needs a decision*

The gate said: accept when the strings match faster-whisper's. They do not —
14 of 50 match byte-for-byte. The reason is not a defect, and chasing it
further would be chasing the wrong thing.

Four different feature pipelines were measured against the same baseline:

| Feature path | identical | Rust CER |
|---|---|---|
| `ct2rs` stock, ruy | 16/50 | 0.0411 |
| `ct2rs` stock, oneDNN | 14/50 | 0.0415 |
| global normalisation, `mel_spec` framing | 9/50 | 0.0383 |
| **fully conformant STFT** | **14/50** | **0.0389** |
| faster-whisper baseline | — | 0.0500 |

The agreement count barely moves while the CER gap stays put. Preprocessing is
not what separates the two engines — and once the mel was verified bit-exact,
it could not be.

The baseline is what differs. `results/cpu_bench.py` calls
`m.transcribe(p, language="bn", beam_size=1, suppress_tokens=[])` and takes
**faster-whisper's defaults for everything else**, which include:

  - `temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` — a fallback ladder that
    **samples** whenever greedy output trips the compression-ratio or logprob
    check. The baseline is therefore not deterministic, and not reproducible
    byte-for-byte by anything, including faster-whisper itself.
  - `without_timestamps=False` — timestamp tokens are generated and used for
    segmentation. The spike sends `<|notimestamps|>`, a different prompt and so
    a different decode path.
  - `condition_on_previous_text=True`.

So the comparison has been one deterministic greedy pass against a stochastic
multi-temperature decode using a different prompt. Those cannot match in
general. It also explains the mangled openings in the baseline — `0`, `00`,
`ntermedit`, `jara`, `oup` — which are fallback artifacts, and why the
baseline's CER is the worse of the two.

**Kotha should keep the single greedy pass.** Temperature fallback is
non-deterministic, costs extra decode passes, and for live dictation an
occasional visible error beats an invisible resample. The divergence is a
deliberate difference in decode policy, not a bug to fix.

**The open decision — how to close this out. Kayes's call:**

  1. **Re-run the baseline with matched settings** — `temperature=0`,
     `without_timestamps=True`, `condition_on_previous_text=False` — and diff
     against that. This is the only way to get a true byte-level comparison.
     It needs `pip install faster-whisper` in a venv; agent auto mode blocks
     package installation, so it has to be run by hand.
  2. **Accept the gate on the evidence already in hand:** mel bit-exact against
     Whisper's reference, better CER than the Python baseline, correct
     code-switched script, no token fusion, and speed parity.

Recommendation: (1) if it is worth an hour, because a clean byte-diff is the
strongest possible evidence and it retires the question permanently. (2) is
defensible on its own, and nothing downstream is blocked meanwhile.

**On the CER gap — state it carefully.** Rust scores 0.0389 against the
references where the baseline scores 0.0500, and that gap held across every
preprocessing variant. The likeliest reading is that temperature fallback hurts
on this audio, since these chunks begin mid-word and trip the
compression-ratio check often. That is a claim about *decode settings*, not
about Rust versus Python — the same settings in either language should give the
same result. It is 50 utterances. Do not repeat it as a headline number, and
do not let it near the paper.

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
| RTF, int8, 6 threads — Ryzen 5600G, oneDNN | **0.639** (1.57× realtime) | 2026-08-30 |
| RTF, int8, 6 threads — Ryzen 5600G, ruy | 1.222 (0.82× realtime) | 2026-08-30 |
| RTF, int8, 6 threads — faster-whisper, same 50 files | 0.669 (1.50× realtime) | earlier, `cpu_bench.json` |
| Model load time — Ryzen | 0.3 s | 2026-08-30 |
| CER vs reference, 50 utts — Rust, conformant mel | 0.0389 | 2026-08-30 |
| CER vs reference, 50 utts — faster-whisper baseline | 0.0500 | 2026-08-30 |
| Strings identical to baseline, 50 utts | 14 / 50 (see *Why the strings still differ*) | 2026-08-30 |
| Mel vs Whisper reference, max abs diff | 0.0 — bit-exact | 2026-08-30 |
| RTF, int8, 4 threads — M2 | — | |
| RTF, int8, 8 threads — M2 | — | |
| Peak RSS | — | |
| Corrector latency per sentence | — | |
| Strict English-F1, before corrector | — | |
| Strict English-F1, after corrector | — | |
