# Kotha — project instructions

**Read `PLAN.md` next.** It holds the live state: which phase is done, what is
in progress, what is blocked. This file holds only the durable rules.

---

## 1. What this is

**Kotha (কথা)** is an offline desktop dictation app for Bengali–English
code-switched speech. Press a hotkey anywhere, talk, and the text lands at your
cursor — the way macOS dictation works, but for the way Bangladeshis actually
speak: Bangla matrix speech with English words written in **Latin script**.

It runs Kayes's fine-tuned model, `kayees/whisper-medium-bn-en-cs-faster`,
entirely on the user's CPU. No network after first run. No account, no API key.

## 2. This is NOT the paper project

`~/Documents/asr` is a separate repository containing an academic paper about
the model. **Kotha is a product, not an experiment.**

- Do not edit anything under `~/Documents/asr` from this project.
- Do not treat evaluation numbers here as paper results.
- The one legitimate borrow is **read-only reference**: the model's inference
  parameters, and `notebooks/eval/files/bnasr_eval.py` for its `script_of()`
  token classifier, which the spelling corrector reuses so that "what counts as
  an English token" means the same thing in both projects.

If work here produces something the paper would want, say so and stop. Kayes
decides whether it crosses over.

## 3. The stack, and why

| Layer | Choice |
|---|---|
| Language | **Rust** |
| App shell | **Tauri 2** (system webview, ~20 MB installers) |
| Inference | **`ct2rs`** → CTranslate2 C++ |
| UI | Plain HTML + CSS. No React, no framework |
| Audio in | `cpal` → 16 kHz mono, `rubato` to resample |
| Segmenting | `earshot` (pure-Rust VAD, ~110 KB) |
| Spelling repair | `symspell` + `rphonetic` (Double Metaphone) |
| Text out | `arboard` clipboard + `enigo` synthetic paste |
| Hotkey | `rdev` / `tauri-plugin-global-shortcut` |

**Why Rust and not Python:** `ct2rs` loads a faster-whisper model directory
*exactly as published* — `model.bin`, `tokenizer.json`,
`preprocessor_config.json`. No conversion, no re-quantisation, no revalidation.
The app runs the identical int8 weights that were benchmarked. A Python build
would add ~350 MB and PyInstaller packaging pain on three platforms.

**Why not whisper.cpp:** the model's vocabulary is 50,364, not Whisper's
standard 51,865. whisper.cpp hard-codes vocab size and special-token IDs;
custom-vocab models are a documented breakage.

## 4. The model — exact facts

**Repo:** `kayees/whisper-medium-bn-en-cs-faster` (CTranslate2, int8)

| File | Size |
|---|---|
| `model.bin` | 775 MB |
| `tokenizer.json` | 2.1 MB |
| `vocabulary.json` | 1.6 MB |
| `config.json` | 1.4 KB |
| `preprocessor_config.json` | 315 B |

`ct2rs::Whisper::new()` wants that directory as-is. Do not rename or prune it.

**Inference parameters — treat as fixed until measured otherwise:**

```rust
language:        Some("bn")
beam_size:       1              // beam 5 is ~3x slower; unusable for dictation
suppress_tokens: vec![]         // see §5
```

Threads: `Config { num_threads_per_replica: <physical cores> }`. On the Ryzen
5600G, 6 threads beat 12 — SMT hurt. The M2 has 4 performance + 4 efficiency
cores, so 8 may lose to 4. **Measure before assuming.**

Reference performance (Ryzen 5 5600G, 6 threads, int8): 1.50× real time,
1.4 GB peak RSS, ~4 s model load. The M2 should beat this.

## 5. Non-negotiables

**`suppress_tokens: vec![]` at every call.** Under this model's Bengali BPE,
token 220 is the space. faster-whisper's Python layer computes a suppression
list against OpenAI's token IDs and bans it, fusing English words to their
neighbours: English-F1 collapses ~70 → ~10 while CER barely moves.

The Rust path is *probably* immune — the model's `config.json` carries
`"suppress_ids": []`, and the C++ engine reads its default from there rather
than recomputing. Pass the empty vector explicitly anyway. This bug is silent,
catastrophic, and invisible in CER.

**The spelling corrector never touches non-Latin tokens.** It runs only on
tokens that `script_of()` classifies as `latin`. Bengali output must be
unmodifiable by construction — that guarantee is what makes it safe to correct
English aggressively.

**Correct-or-abstain, never guess.** If no candidate clears the confidence
margin, leave the word alone. A confidently wrong real word is worse than a
visible misspelling: the user can fix what they can see.

**Nothing heavy on battery.** See §7.

## 6. Why there is a spelling corrector

The model hears English well and writes it in Latin script, but misspells it.
Measured on the paper's 393-utterance test set:

| English-token metric | P | R | F1 |
|---|---|---|---|
| Strict — Latin *and* spelled right | 75.3 | 69.4 | 72.2 |
| Tolerant — heard it, near-miss ok | 89.4 | 82.4 | 85.8 |

The **13.6-point gap** is the corrector's target. Caveat: tolerant matching also
forgives Bengali-script renderings, so part of that gap is not spelling.
Phase 1 splits the two before anything is built.

## 7. This machine

Apple M2, 8 cores (4P + 4E, no SMT), 16 GB RAM, ~130 GB free. macOS. Homebrew
present. Node 26 and Python 3.14 present. **Rust and cmake are not installed.**

**Battery rule.** Kayes works on this laptop unplugged. These are plugged-in
work only:

- `cargo build` of anything touching CTranslate2 (a long C++ compile)
- downloading the 778 MB model
- running transcription

`setup.sh` refuses those steps on battery unless given `--force`. Do not work
around the guard; tell Kayes to plug in. Writing code, docs and tests is always
fine.

## 8. How Kayes works

- Wants **short, plain-language, direct** answers. A recommendation, not a
  survey of options.
- **One change at a time.** Dislikes large refactors delivered in one go.
- Wants to **understand what a component is for**, not just have it appear.
- Prefers **editing existing files** over adding new ones; consolidated scripts
  over many small ones.
- Catches bugs by noticing something feels wrong, then asks for diagnosis from
  logs. So: **log generously**, never drop data silently.
- Reserves final say on linguistic and naming calls. Grants decision authority
  on technical and configuration choices.
