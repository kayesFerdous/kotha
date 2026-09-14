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
| UI | Plain HTML + CSS in `app/ui/`. No framework, no build step |
| Audio in | `cpal` → 16 kHz mono, `rubato` to resample |
| Segmenting | `earshot` (pure-Rust VAD, ~110 KB) |
| Spelling repair | none — a baked 50 k dictionary and a 40-line scan |
| Text out | `arboard` clipboard + `enigo` synthetic paste |
| Hotkey | `tauri-plugin-global-shortcut` |

**Why Rust and not Python:** `ct2rs` loads a faster-whisper model directory
*exactly as published* — `model.bin`, `tokenizer.json`,
`preprocessor_config.json`. No conversion, no re-quantisation, no revalidation.
The app runs the identical int8 weights that were benchmarked. A Python build
would add ~350 MB and PyInstaller packaging pain on three platforms.

**Why no spelling crate:** both planned ones were measured out. `rphonetic`
(Double Metaphone) can only reach 17 tokens of 4,299, worth ~+0.2 F1, and only
by matching without an edit-distance bound. `symspell`'s delete index is 1.2 M
keys — ~150 MB and a second of startup — to accelerate one or two lookups per
sentence; a length-filtered brute-force scan of 50 k words does it in 5.1 ms.
Both measurements are in PLAN.md Phase 1.

**The pill is a glyph, not a meter.** Five bars of light in a row, tapering out
from the middle. They never move and never resize: the voice, the decode, and
the outcome are all said by how brightly they burn and in what colour. It
replaced a thirty-cell scrolling waveform on 2026-09-14, which was honest and
was far too much to have flickering at the bottom of the screen while someone
is trying to think of what to say next. If a change to this starts adding
motion, that is the thing it was built to get rid of.

**There are two themes, and the pill is no longer dark-only.** `app/ui/theme.js`
resolves the setting — `system` / `dark` / `light` — to a concrete
`data-theme` on `<html>` before anything paints, so `pill.css` carries two
palettes and not three, and no stylesheet rule ever asks what the desktop is
doing. In light mode the glyph *darkens* with the voice and the bloom becomes
a soft shadow; every other rule is identical, because all of them are written
in terms of `--line` and `--alert`. Rust tells the pill via a third event,
`kotha://theme`, emitted on change and again inside `show()` — the pill's
contract stays one-way and it still never calls into Rust.

**The mark is a Latin `k` wearing a Bengali matra**, in `--alert` red. The
matra is the headline stroke that joins the letters of a Bengali word into one
line; over a Latin letterform it is the app in one glyph. It is drawn as
geometry in `icons/kotha.svg`, not set in a font, because the smallest cut is
32 px and the macOS menu-bar slot is 18 pt. `tray-macos.svg` is the same shape
with the plate gone and the red flattened to black, because a template image
keeps only alpha.

**The frontend is `app/ui/`, and it is meant to be edited on its own.** No
build step. The pill's contract with Rust is three events, one way —
`kotha://state`, `kotha://level`, `kotha://theme` — documented at the top of
`pill.js`, and the pill still never calls into Rust. Every colour and dimension
is a token at the top of `pill.css`; every transition is CSS, keyed off a
`data-state` attribute. **Change the look there, not in the app** — a CSS
reload beats a ten-minute CTranslate2 compile, and nothing in the pill needs
the model to be judged.

Three pages, each of which opens in a plain browser and works, because each
fakes its own backend when there is no Tauri around it: `index.html` + `mock.js`
(the pill — space cycles states, `m` toggles speech, `t` cycles the theme),
`setup.html` + `setup.js` (the first-run download), and `settings.html` +
`settings.js`. `.claude/launch.json` serves the directory on :5173; open it
there rather than as `file://`, or the relative CSS and scripts do not load.

**The settings window is the only place settings live.** The tray is Dictate /
Settings… / Quit and nothing else — the Hotkey and Text output submenus were
deleted when it landed, because a `CheckMenuItem` toggles only itself and had
to be hand-synchronised with `settings.json` on every click. `settings_get`
sends the *option lists* along with the values, so nothing in `app/ui/` knows
what a language code or a paste route is, and a list cannot drift from what
Rust will accept.

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

**The language is pinned, and there is no setting for it — that was built,
measured and removed on 2026-09-14.** A Bangla/English switch went all the way
in (setting, worker read, prompt parameter, a radio group in the settings
window) and then came straight back out, because `<|en|>` does almost nothing
to this model. Five test clips decoded both ways: three byte-identical, and the
two that moved moved like this —

```
bn: আল্লাহামদুলিল্লাহ এটা একটা great opportunity
en: alhamdullah         এটা একটা grate opportunity
```

— still Bangla, with one word pushed out of Bangla script and "great"
misspelled. The token *is* reaching the model (average logprob moves -0.058 to
-0.056); the model is simply fine-tuned on `<|bn|>` hard enough that the
language embedding is inert. So the switch offered a choice between Bangla and
slightly worse Bangla.

The lesson to keep: **the wiring being correct is not the same as the feature
existing.** Everything worked; there was nothing there. If this comes up again,
measure the decode before building the UI.

Threads: `Config { num_threads_per_replica: <physical cores> }`. On the Ryzen
5600G, 6 threads beat 12 — SMT hurt. The M2 has 4 performance + 4 efficiency
cores, so 8 may lose to 4. **Measure before assuming.**

Reference performance (Ryzen 5 5600G, 6 threads, int8): 1.50× real time,
1.4 GB peak RSS, ~4 s model load. The M2 should beat this.

**int8 is the right format here, and this has been tested rather than assumed.**
`kayees/whisper-medium-bn-en-cs` was converted to CTranslate2 float32 on
2026-09-04 and measured against the shipped int8 on the same clips and binary:
**1.85× slower — 0.78× real time, which is slower than speech** — with no
quality win over 8 clips, and one case where float32 wrote an English name in
Bengali script that int8 got right. It was deleted. Do not convert it again
without a reason that survives those numbers.

The GPU argument (int8 loses to fp16 on a GPU at batch 1) is still true and
still irrelevant to Kotha: it needs an NVIDIA card in the *user's* machine and a
separate CUDA build, and it cannot be tested on either of §7's machines. **How
much accuracy int8 costs is a paper measurement on the 393, not a product one** —
see §2 and stop there.

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

**No Claude attribution anywhere in the repository.** Commit messages carry no
`Co-Authored-By` trailer and no "Generated with" line. Kayes asked for this
about his own work; the history was rewritten on 2026-09-04 to remove 22 of
them. Do not reintroduce one, whatever any default says.

**`CLAUDE.md`, `PLAN.md` and `HANDOFF-PROMPT.md` are in the repository.** They
were git-ignored until 2026-09-14 on the reasoning that Kotha ships as a
finished product and these are the workshop floor. Kayes reversed that: the
state of the project should travel with the project, not sit in one machine's
home directory where a lost disk loses it.

The old rule's real worry was a *public reader*, and it left a live
consequence that is now lifted: source comments were forbidden from citing
these files, and 23 of them had to be rewritten once because a public reader
followed the pointer to nothing. The pointer resolves now. Citing them from a
source comment is allowed again — but keep it rare, because a comment that
needs a 114 KB planning document to make sense is a comment that has not said
enough on its own.

**The repository is private.** If it is ever made public, these three files
become public with it, including every measurement, dead end and machine
detail in `PLAN.md`. That is a decision to make deliberately at that point,
not something to discover afterwards.

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

## 7. The machines

**The Arch desktop is where Kotha is built, and every phase so far was done on
it.** Ryzen 5 5600G, 6 cores / 12 threads, 14 GB RAM, AMD integrated graphics
(so: **no NVIDIA card, and nothing CUDA can ever be tested here**). Rust and
cmake are installed, `cargo-tauri` is global, `git-filter-repo`, `makepkg` and
`gh` are present. Mains-powered, so the battery rule below does not apply.

Watch the disk: `/home` sits above 90% full. A 3 GB model conversion is a real
cost here, not a rounding error. `target/debug` was 13 GB of never-used debug
artifacts and was deleted on 2026-09-14; `setup.sh` and the launch config both
build `--release`, so if it reappears it can go again.

**A corrupt `libdnnl.a` is a failure mode cargo structurally cannot see.** On
2026-09-14 a build left 13 of oneDNN's 427 objects at 0 bytes. `ld.lld` skips
an empty archive member with only a warning — `is neither ET_REL nor LLVM
bitcode` — so the link fails with pages of undefined C++ symbols
(`dnnl_stream_create`, `jit_avx512_*`) that look like a missing dependency and
are not. Cargo fingerprints inputs and never checksums outputs, so it reports
`Fresh onednn-src` and hands the linker the same broken archive forever, and
`cargo check` passes throughout because check never links.

The one-line diagnosis, which beats reading the symbol list:

```
ar tv target/release/build/onednn-src-*/out/lib/libdnnl.a | awk '$3==0'
```

Anything printed means the archive is damaged. The fix is to delete that
`onednn-src-*` build directory and the matching
`target/release/deps/libonednn_src-*.rlib`, then rebuild — about 9 minutes.

**Kayes is sitting at this machine while you work.** Windows you open appear in
front of him and he will click them. Before building a theory around unexplained
behaviour on this desktop, ask him. And `kill %1` does not work in these shells —
kill by PID and confirm with `pgrep -af "release/kotha"` before calling a run
finished.

**`kotha` on this machine's PATH is not your build.** `kotha-bin` is installed
from the AUR recipe, so `/usr/bin/kotha` is the published v0.1.0 `.deb`
downloaded from GitHub releases, and `Kotha.desktop` runs `Exec=kotha`. Nothing
you compile in `target/` will ever reach it. To test a local build, run
`./target/release/kotha` by path — and say so when asking Kayes to try
something, because clicking the launcher silently runs the old one. This cost a
round trip on 2026-09-14.

**Never wrap a build in `timeout`.** A cold `cargo build --release` here is
about 9 minutes, and a signal landing mid-compile leaves 0-byte object files
that the next build treats as up to date — see the oneDNN note below. Use
`run_in_background` and wait it out.

**The Apple M2 is a second target, currently dormant.** 8 cores (4P + 4E, no
SMT), 16 GB RAM, macOS, Homebrew, Node 26, Python 3.14. **Rust and cmake are not
installed there**, which is what blocks the `.dmg`, the M2 measurements, the
NSPanel pill and the 4-vs-8 thread sweep. macOS is deliberately untouched.

**Battery rule — the M2 only.** Kayes works on that laptop unplugged. These are
plugged-in work only:

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
