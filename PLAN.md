# Kotha — plan and live state

Read `CLAUDE.md` first for the durable rules. The same plan with the reasoning
laid out visually is at
<https://claude.ai/code/artifact/3107084e-3472-4798-918f-e47e3410785b>. This file is the working state:
update it as phases complete. Last touched 2026-08-30.

**Status: Phase 0 written, not yet run.** Nothing has been compiled and the
model has not been downloaded — the laptop was on battery when the scaffold was
written. `./setup.sh` is the next command, plugged in.

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

### Phase 0 — Prove the engine  ⟵ GATE, next up

Does `ct2rs` load this model and produce the same text faster-whisper does?

- [ ] Toolchain — **run this yourself, an agent cannot** (see note below)
- [x] Model downloaded to `models/whisper-medium-bn-en-cs-faster/` —
      774,731,149 bytes, sha256 `9c0e38dc…ea74`, verified against the HF manifest
- [ ] `cd spike && cargo run --release -- <model-dir> <wav>...`
- [ ] Get 20 test WAVs onto this machine (see *Getting audio* below)
- [ ] Diff Rust output against faster-whisper on the same files
- [ ] Record the M2's real RTF and best thread count in this file

**Accept when:** the strings match faster-whisper's, and English words come out
in Latin script with spaces intact.

**If it fails:** stop and reconsider. Fallback is a bundled Python sidecar
running faster-whisper — roughly +300 MB and worse packaging, but it works. This
gate exists so that decision costs two days, not six weeks.

**Why first:** everything after this is ordinary application work. This is the
only genuine unknown in the project.

**Toolchain note.** Claude Code's auto mode blocks package installation and
`curl | sh`, so `setup.sh`'s Rust and cmake steps cannot be run by an agent.
Kayes runs these once, by hand:

```bash
brew install cmake rustup
echo 'export PATH="/opt/homebrew/opt/rustup/bin:$PATH"' >> ~/.zshrc
export PATH="/opt/homebrew/opt/rustup/bin:$PATH"
rustup default stable
```

Two traps, both hit on 2026-08-30:

- Homebrew's `rustup` formula **no longer ships `rustup-init`**. Use
  `rustup default stable` to install the toolchain instead.
- It is **keg-only**, so it is not symlinked into `/opt/homebrew/bin`. Without
  the PATH line above, `rustup` is "command not found" even though it installed
  fine.

`rustup` is preferred over `brew install rust` because Tauri needs per-target
toolchains later, and the two formulae conflict.

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

## Continuing on the Arch / Ryzen machine

The scaffold was written on the M2. To pick up there:

```bash
sudo pacman -S cmake rustup && rustup default stable
```

Two shortcuts that machine has and the Mac did not:

- **The model is probably already local.** `cpu_bench.py` in the paper repo
  points at `~/Documents/ASR/fine_tuned/whisper-medium-bn-v1.3-ct2-int8`. If
  that is the shipped int8 release, pass it to the spike directly instead of
  re-downloading 774 MB. Confirm first — `model.bin` should be
  774,731,149 bytes with sha256 `9c0e38dc…ea74`. If it differs, run
  `./setup.sh` and use the fresh copy; the gate has to test what ships.
- **The test WAVs are there**, under `bangla-asr-test/chunks/test/`. That is the
  audio the gate actually needs, because it comes with reference text.

Note the battery guard in `setup.sh` is macOS-only and is a no-op on Linux,
which is correct — that machine is a desktop.

Hardware differs and the numbers do not transfer: Ryzen 5 5600G is 6 cores / 12
SMT threads, where 6 threads beat 12. The M2 is 4P + 4E with no SMT. Record both
separately in the measurements table.

## Getting audio for Phase 0

The 393 test WAVs are **not on this Mac**. They live on the Arch machine under
`bangla-asr-test/chunks/test/`. Two ways forward:

1. **Copy ~20 chunks over.** Best for the gate — they come with known reference
   text, so the comparison is meaningful. A few MB.
2. **Record on the Mac.** Fine for a smoke test, and it is the real use case,
   but there is no reference to diff against.

Do (1) for the gate, (2) for everything after.

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
| RTF, int8, 4 threads | — | |
| RTF, int8, 8 threads | — | |
| Peak RSS | — | |
| Model load time | — | |
| Corrector latency per sentence | — | |
| Strict English-F1, before corrector | — | |
| Strict English-F1, after corrector | — | |
