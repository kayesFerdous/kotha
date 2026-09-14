# Kotha — plan and live state

Read `CLAUDE.md` first for the durable rules. The same plan with the reasoning
laid out visually is at
<https://claude.ai/code/artifact/3107084e-3472-4798-918f-e47e3410785b>. This file is the working state:
update it as phases complete. Last touched 2026-09-02.

**2026-09-14 — the pill was redesigned and the glyph latency fixed. See *The
glyph* at the end of this file.**

**Status: Phases 0-4 pass on the Arch/Ryzen machine. Phase 5 is in progress.** The loop runs end to end — microphone to VAD to model to
spelling corrector to clipboard — with the model held warm between utterances,
verified against live speech. Phase 1's Rust port landed 2026-08-31 and matches
the Python prototype on 392 of 393 held-out lines — the one difference being an
exact tie the prototype resolves at random — so the corrector is now *in* the
pipeline rather than beside it.

Phase 4, 2026-08-31: the pill is built and the Tauri shell around it runs —
frameless, transparent, always on top, click-through, bottom centre, and
`focused = false` every time it appears. The chain behind it is still a stub;
wiring the real one is the next commit.

One known defect carried forward: Whisper repetition loops on roughly a quarter
of live utterances. The symptom is now repaired in the text (Phase 4,
*repetition loops*); the cause is still open and needs a decode-side sweep.

**Phase 0, 2026-08-30.** `ct2rs` runs
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
 tauri       cpal            earshot           ct2rs        correct.rs    arboard
 plugin                                                                   + enigo
```

`rdev` and `symspell` are off this list. The hotkey comes from
`tauri-plugin-global-shortcut`, which Phase 4 confirmed registers on KDE
Wayland; the corrector is a baked dictionary and a 40-line scan (Phase 1).

Press the hotkey. A small pill grows out of a line, showing a five-bar glyph
that brightens with your voice. Talk. Each time
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

- [x] Re-decode the 393 test utterances to get hypothesis text
- [x] Align English tokens against references; **split the 13.6-point
      strict/tolerant gap into *misspelled in Latin* vs *written in Bengali***
- [x] Build the dictionary — **`wordfreq`, already installed, no download.** It
      is the same OpenSubtitles-blend register `en_50k.txt` comes from, plus
      the English side of the **training** corpus (7,320 types) for the domain
      words a general list lacks. 50,264 entries
- [x] SymSpell index, edit distance ≤ 2 (1.2 M delete keys)
- [x] Double Metaphone fallback — **measured out, not built.** Its whole
      addressable market is 17 tokens of 4,299. See below
- [x] Tune the abstain threshold **on training-side data only**
- [x] Port to Rust — `spike/src/correct.rs`, byte-identical to the prototype

**Accept when:** strict English-F1 improves measurably on held-out data, and
zero non-Latin tokens are modified across the whole test set.

**Rule:** do not tune on the 393 test utterances. Hold them out, evaluate once.

#### Result — 2026-08-31. **Phase 1 passes.**

`spike/correct.py`. Strict English-F1 **71.81 → 77.02, +5.21 points** on the
held-out 393, and **zero non-Latin tokens modified** — asserted in the
evaluation, not trusted. Both acceptance conditions met.

| | before | after |
|---|---|---|
| written in Latin, spelled right | 2952 | **3169** |
| written in Latin, misspelled | 545 | **309** |
| written in Latin, unrecognisably wrong | 194 | 216 |
| strict / tolerant gap | 13.54 pts | **8.05 pts** |
| strict P / R / F1 | 74.54 / 69.27 / 71.81 | **79.95 / 74.30 / 77.02** |

236 misspellings corrected: 217 became exact, ~19 got worse. **About 91%
precision on the tokens it chose to touch.**

**The gate is a frequency margin under a floor, not "is it a word."** The
obvious design — correct only tokens missing from the dictionary — fails twice
here: 26.4% of the model's misspellings *are* real English words (`mill` for
`meal`, `throw` for `through`), and the dictionary itself contains common
misspellings (`grammer` is in wordfreq's top 50k). So instead:

- `KNOWN_FLOOR = 2.5` — a token above this zipf is never touched. Without it
  the corrector eats ordinary English (`then` → `the`, `hand` → `and`) at ~1%
  of every English token, which is far too much to do to correct text.
- `MARGIN = 1.0` — below the floor, take the best candidate within ED 2 only
  if it is this much more common. Inert at floor 2.5; kept as a guard for when
  the dictionary changes.

**Calibrated on the training split, never on the 393.** Training references
turned out not to be perfectly clean — the corrector's top "errors" against
them were `algorithom` → `algorithm`, `truncess` → `princess`, which are fixes
— so a raw change count is useless as a safety number. The metric that works
is *did it rewrite a token that was already a real English word*. Floor 2.5 is
the knee: at 2.5 and below that number is **0** while all 585 tail fixes
survive; at 3.0 damage appears (`unhappiness` → `happiness`, `slab` → `lab`).

**What the floor gives up, deliberately:** `grammer` sits at zipf 2.9, above
the floor, so it is not fixed. Unigram frequency cannot separate `grammer`
(a misspelling at 2.9) from `neutrally` (a real word at ~3.0). Nothing at this
floor can. That separation needs the surrounding words — the bigram context in
Phase 7 — and this is the measurement that says when to build it. The
concession is asserted in `correct.py --selftest`, so raising the floor
without re-calibrating fails loudly.

**Caveat on the safety number:** "real word" is wordfreq's top 50k, which
misses inflected forms — `sunbeams` and `neutrally` counted as benign when
they are words. It under-counts damage. That is the safe direction here, since
2.5 is already the most protective setting swept; it would matter only if a
looser one were chosen.

**Headroom left:** 309 misspellings remain. 90% are within ED 3, so Double
Metaphone is the next lever, then bigram context for the real-word 26%.

#### The Rust port — 2026-08-31. **Phase 1 is closed.**

`spike/src/correct.rs`, wired into the pipeline where `emit()` prints. The port
is faithful, and that is checked rather than asserted:

```bash
live --correct < normalised.txt   |   correct.py < normalised.txt
```

Both were run over the same 393 normalised hypotheses, 257 of which the
corrector rewrote. **They agree on every line but one, and that one is not a
disagreement about spelling — it is an exact tie.**

`yeas` is not in the dictionary. `year` and `years` are both edit distance 1
from it and both sit at zipf 5.96, so the two candidates are indistinguishable
on distance *and* on frequency. The prototype iterates a Python `set` and keeps
whichever it yields first, which depends on the process's string-hash seed:
across six seeds it produced `years` on three and `year` on the other three.
The Rust scan walks the dictionary in order and takes the alphabetically first,
every time.

So the port is the *more* deterministic of the two, and this is the only token
in the 393 where they can differ at all. The +5.21 F1 carries across unchanged;
it does not need re-measuring. Asserted in `correct.rs`'s tests so a future
change to the tie-break fails loudly.

*(An earlier run of this comparison came out byte-identical and was written up
that way. That was one hash seed being lucky, not a stronger result.)*

**This is a third measurement pointing at bigram context.** `grammer` was the
first, the real-word residue the second. Nothing about `yeas` can be resolved
by unigram frequency — only the surrounding words could say whether a year or
several were meant.

**The dictionary is baked, which was the real decision.** `build_dictionary()`
reads `wordfreq` and the training manifest at *runtime*, and neither exists on
a user's machine. `correct.py --dump-dict` writes `spike/dict.tsv` — word, tab,
zipf, 50,264 lines, 860 KB — and Rust `include_str!`s it. Committed, because it
is an input to the build, not an output of it. Regenerate it if the dictionary
changes, and re-run the parity diff after.

**No delete index, and no `symspell` crate.** The prototype's SymSpell index is
1.2 M keys, which in Rust is ~150 MB of `HashMap` and a second of startup. It
buys speed nothing needs: only tokens *below the floor* are looked up, one or
two per sentence, so a length-filtered brute-force scan of 50 k words does the
job in **5.1 ms per sentence** — invisible against a multi-second decode. That
also drops a dependency and a dictionary-format conversion.

A side effect worth recording: the scan is a *superset* of what the prototype
finds. The prototype's index is built with each dictionary word's own edit
budget while its lookup uses the token's, so it can miss a candidate when the
two differ. The parity run says that never fires on real output.

**One thing the port had to add.** The token-level rule cannot be pointed at
raw model output. The model writes `blog।` and `footbal,`, and a bare lookup
rewrites `footbal,` to `football` — eating the comma. So `correct_text()`
splits each whitespace token into leading punctuation, a core and trailing
punctuation, considers only the core, and restores capitalisation if the core
changed. `--correct` keeps the prototype's plain token-level behaviour, because
that is what the parity diff needs to compare.

**A toy-dictionary assumption did not survive contact.** The prototype's
selftest asserts `carector` stays put, being edit distance 3 from `character`.
Against the *real* 50 k dictionary it becomes `creator`, at edit distance 2.
The prototype does the same — its toy dictionary simply held no ED-2
neighbour. Nothing changed; the Rust test now records the real behaviour, and
it is inside the measured +5.21. It is another instance of the real-word error
the floor cannot see.

#### Double Metaphone: measured, then dropped — 2026-08-31

The plan said Double Metaphone next, because 90% of the 309 surviving
misspellings sit within edit distance 3 and SymSpell stops at 2. That framing
was right about edit distance and wrong about the binding constraint.

`gap.py --corrected` now prints the ceiling. A fallback can only fire on a
token **below the corrector's floor** — above it the token is protected on
purpose, and no amount of phonetic matching changes that:

| Of the 309 misspellings still standing | count | share |
|---|---|---|
| above the floor, protected by design | 278 | 90.0% |
| below the floor, a fallback could act | 31 | 10.0% |
| below the floor **and** beyond ED 2 | **17** | **5.5%** |

**17 tokens out of 4,299.** Perfect correction of every one of them, with zero
damage, is worth about +0.2 strict F1. Double Metaphone matches without an
edit-distance bound, so zero damage is not what it would deliver — the same
loose net that reaches `ambacerer` → `ambassador` (ED 5) reaches a great deal
else in a 50k dictionary. Correct-or-abstain says do not take that trade for
0.2 points.

Two things are visible in the residue and worth writing down:

- **The floor, not edit distance, is what is left.** 88.7% of the surviving
  misspellings are real English words — `one`/`phone`, `letter`/`leather`,
  `mill`/`meal`, `foster`/`coaster`. The corrector refuses them by design and
  is right to. Unigram frequency cannot tell them apart from correct text.
  **This is the second measurement pointing at bigram context** (Phase 7); the
  first was `grammer`.
- **Of the 31 touchable, 14 are within ED 2 already** and were abstained on by
  the short-token budget (`gim` → `gym`, seven times; `ead`, `goe`, `psr` — all
  three characters, budget 0). That guard exists to stop `hal` → `hall`, and it
  is doing its job. If it is ever revisited, it must be swept on the **training
  split** with `--calibrate`'s risky/benign metric, not against the list above.

*Method note:* the residue was inspected on held-out output, which `gap.py`
labels understanding-only. Nothing here changed a threshold. The one candidate
change it suggests (the short-token budget) is deliberately left unmade for
exactly that reason.

**So Phase 1 is finished.** The corrector took +5.21 points, everything below
the floor that ED 2 can reach is reached, and the next real lever is context,
not a wider net.

#### The gap, split — 2026-08-31

`spike/gap.py` walks the jiwer word alignment over all 393 utterances and
classifies every **Latin reference token** by what the model actually put in
its place. Hypotheses are the Rust spike's (int8, greedy, `suppress_tokens=[]`)
— the engine the corrector will run behind. Scoring imports the paper's
`bnasr_eval.py` read-only, so "Latin token" and "tolerant match" mean exactly
what they mean in the paper. References are `test_manifest.csv`, which
HANDOVER §3 names as the test set; `listen_v13/test_repaired.csv` disagrees on
122 of 393 rows and is the earlier, worse version (`englishe` for `english`).

| What happened to a Latin reference token | count | share |
|---|---|---|
| written in Latin, spelled right | 2952 | 68.7% |
| **written in Latin, misspelled** | **545** | **12.7%** |
| written in Latin, unrecognisably wrong | 194 | 4.5% |
| **written in Bangla script, recognisable** | **37** | **0.9%** |
| written in Bangla, unrecognisable | 60 | 1.4% |
| dropped entirely | 510 | 11.9% |
| replaced by a digit / other | 1 | 0.0% |
| | 4299 | |

**The gap is 13.54 points, and 93.6% of it is spelling.** Only 37 tokens —
6.4% of the gap, 0.9% of all English tokens — are the Bengali-script case.
CLAUDE.md §6's caveat ("part of that gap is not spelling") is answered: almost
none of it. The corrector owns essentially the whole 13.6 points.

That the split reproduces the paper's gap at 13.54 on a different engine and
precision, with strict F1 71.81 against the paper's 72.2, is the check that
the measurement is sound.

| Strict English-token score | P | R | F1 |
|---|---|---|---|
| now | 74.54 | 69.27 | **71.81** |
| every misspelling fixed | 88.19 | 81.95 | 84.95 |
| plus the unrecognisable Latin | 93.04 | 86.46 | 89.63 |

Both ceilings assume perfect correction and zero damage to tokens already
right. They are upper bounds, not forecasts. A misspelling costs twice in
strict scoring — a missed reference token *and* a spurious hypothesis token —
which is why the F1 ceiling (+13.1) exceeds the recall gap.

**Edit distance from the model's spelling to the reference:** 56.9% are
distance 1, 84.8% are ≤ 2. So **SymSpell at ED≤2 can reach 462 of 545**; the
remaining 83 need distance 3+ and are what the Double Metaphone fallback is
for. This settles the ED≤2 choice as measured rather than conventional.

**Two things the corrector cannot fix, both larger than they look:**

- **510 dropped tokens (11.9%)** — the model emits nothing at all. Bigger than
  the entire spelling problem's *recall* share, and nothing downstream can
  recover a word that was never written. This is the ceiling on English recall
  no matter how good the corrector gets.
- **Real-word errors.** `mill`→`meal`, `word`→`words`, `collar`→`color`,
  `throw`→`through`, `diner`→`dinner`, `blog`→`vlog`. A dictionary lookup only
  fires on tokens that are *not* words, so these are invisible to it.
  **Quantify this before building** — it decides whether the corrector needs
  the bigram context now (Phase 7) rather than later. It is the one number
  that could change the design.

`gap.py` is resumable: a power cut during the first decode cost 30 utterances
and left a NUL tail where the page cache never flushed, so it strips NULs,
decodes only what is missing, and appends. It also warns and scores the subset
rather than blocking if a decode is still incomplete.

Not a tuning run. The per-word lists it prints are for understanding. The
dictionary must come from `en_50k` plus the **training-side** vocabulary —
reading a misspelling off this output would be fitting the held-out set.

---

### Phase 2 — The loop, no interface  ⟵ PASSES

- [x] Record → VAD → transcribe → clipboard (`spike/src/bin/live.rs`)
- [x] Model stays warm in memory between presses
- [x] Threads pinned to the measured-best count — physical cores, via `num_cpus`
- [ ] Hotkey — **deliberately deferred to Phase 4**, see below
- [x] Say something into the microphone and watch it appear

**Accept when:** a terminal app prints what you said, twice in a row, with no
model reload between them.

#### Result — 2026-08-31

```
cargo run --release --bin live -- <model-dir>          # microphone
cargo run --release --bin live -- <model-dir> x.wav    # same path, no mic
```

The chain runs end to end:

```
cpal → downmix → resample → earshot VAD → segment → Engine → clipboard
```

Three utterances from one 0.3 s model load, on 23 s of concatenated test audio
with one-second pauses between:

| # | audio | decode | |
|---|---|---|---|
| 1 | 7.2 s | 5.7 s | 1.26× realtime |
| 2 | 7.6 s | 5.9 s | 1.29× realtime |
| 3 | 7.9 s | 3.4 s | 2.32× realtime |

The VAD cut at both pauses, on the beat. English came out in Latin script with
spaces intact — no fusion, so `suppress_tokens` is still behaving on this path.
Decode time tracks the *token count*, not the audio length, which is why the
short third utterance is nearly twice as fast per second of audio.

**Live, into the microphone — 2026-08-31.** Eight utterances, one model load,
code-switching handled as designed:

```
[1] 3.8s → 3.7s   hey how are you I all fine thank you
[2] 3.5s → 3.3s   আচ্ছা তোমার সাথে কিছু কথা বলতে চাই।
[4] 4.0s → 3.6s   কালকে আমার examp please আমার জন্য একটু pray করো।
[8] 4.6s → 3.3s   আমি শুনলাম তোমাকে তুমি নাকি একটু sick?
```

English in Latin script, inside Bangla matrix speech, at roughly 1× realtime on
short utterances. `examp` for `exam` is exactly the corrector's target. PipeWire
hands over 16 kHz mono I16 directly, so no resampling happens on this machine,
and fourteen seconds of room silence produced zero false triggers.

#### The one real defect the live run exposed: repetition loops

Two of the eight utterances collapsed into a loop:

```
[6] আর that that that that that ... that thathat that      (3.4s audio, 9.1s decode)
[7] কথা কথা কথা কথা কথা করতে করতে করতে করতে হবে।
```

This is Whisper's classic greedy-decode failure. faster-whisper hides it with
the temperature-fallback ladder — decode greedily, and if the compression ratio
looks degenerate, re-decode with sampling. Phase 0 deliberately turned that
ladder off to get a reproducible byte-diff, and never turned it back on.

The decode time gives it away on its own: 9.1 s for 3.4 s of audio, against
~1× realtime everywhere else. That makes it cheaply detectable without any
text analysis.

`WhisperOptions` carries two levers, both unused today:

- `repetition_penalty` (default 1.0) — the safer one.
- `no_repeat_ngram_size` (default 0) — **probably wrong for this language.**
  Reduplication is grammatical in Bengali: `করতে করতে` means "while doing" and
  is correct at n=2. A hard n-gram ban would break real speech to fix a decode
  bug. [7] is a loop *because it runs to four*, not because it repeats at all.

So the fix is not a one-liner to be guessed at — it needs a sweep on the 393
with the strict-F1 and CER harness that already exists. Deferred by decision,
2026-08-31, not overlooked. **Detecting** the loop is much easier than
preventing it, and re-decoding just those utterances is the cheap first move.

**The engine moved to `spike/src/lib.rs`** so the gate binary and the loop share
one decode path rather than drifting apart. The move is proven faithful twice
over: `check_mel.py` still reports max |diff| 0.0, and five files re-decoded
through the gate are byte-identical to the cached 393-utterance decode.

#### Two things left out on purpose

**The hotkey.** The acceptance test is "a terminal app prints what you said",
and a global hotkey is not what that gates. It also arrives for free in Phase 4:
the shell is Tauri, so the hotkey is `tauri-plugin-global-shortcut`, and wiring
`rdev` now — which on Wayland needs its own permission dance — would be building
something Phase 4 deletes. `live` listens continuously instead, which is
strictly more demanding of the VAD than push-to-talk. **`rdev` should probably
come off the stack table in CLAUDE.md §3** once Phase 4 confirms the plugin
covers it.

**The spelling corrector.** ~~It plugs in where `emit()` prints, once ported
to Rust.~~ **Done, 2026-08-31** — see Phase 1, *The Rust port*. The baked
dictionary this predicted was indeed the whole of the work; the rest was a
transliteration.

#### Segmentation numbers

Ordinary dictation values, not tuned against anything — there is no held-out set
for "where should a chunk end", and the only real failure mode (cutting
mid-word) is audible.

| | value | why |
|---|---|---|
| frame | 256 samples / 16 ms | earshot's fixed contract |
| voice threshold | 0.5 | earshot's own documented split |
| onset | 3 frames | a click or a door must not open a recording |
| hangover | ~600 ms | a natural sentence pause |
| pre-roll | ~256 ms | the VAD trips late; without this the first consonant is clipped and the model has to guess at it |
| minimum voiced | ~300 ms | below this it is noise, not speech |
| forced cut | 25 s | under Whisper's 30 s window, so a monologue splits on *some* boundary instead of being truncated inside the model |

`cargo test --bin live` covers the segmenter state machine (cut on silence,
keep the pre-roll, discard a blip, never exceed the window) and that resampling
preserves duration at 48 kHz, 44.1 kHz stereo and 16 kHz passthrough. The VAD's
own judgement is not unit-tested — that needs real speech, which is what the
file mode is for.

---

### Phase 3 — Text at the cursor  ⟵ PASSES ON LINUX

The OS-specific part. Three mechanisms underneath.

- [x] **Copy-only mode that needs no permissions**, so the app is useful before
      anything is granted — and it is the *default*, because firing synthetic
      keystrokes into whatever window is focused should be asked for
- [x] Restore the previous clipboard contents afterwards — in paste mode only;
      in copy-only mode the clipboard *is* the delivery, so restoring it would
      throw the transcript away
- [x] Linux: X11 vs Wayland split — three routes, one chosen at run time.
      **libei through the desktop portal works on KDE Wayland**, verified
      2026-08-31. See below
- [x] Preserve whatever the user already had on the clipboard — text *or*
      image, and never clobber a fresh copy. See below
- [ ] macOS: Accessibility permission, synthetic ⌘V — the code path exists
      (`Key::Meta`), untested, needs the M2
- [ ] Windows: SendInput — enigo's own path, untested
- [x] Verify it in a real editor — **done, 2026-08-31**: dictated text
      landed at the cursor, via `KOTHA_PASTE=portal`

**Accept when:** dictated text appears in an editor, a browser field, and a
terminal, without the focused window changing. **Met on Linux, 2026-08-31** —
text landed at the cursor in a real editor over the libei portal route. The
browser field and the terminal are the same mechanism and were not separately
watched; macOS and Windows are untested because neither has been built.

**Why clipboard and not keystrokes:** Bengali conjuncts and combining marks
break character-by-character injection in many apps. A paste is atomic.

#### What Linux actually does — 2026-08-31, KDE Plasma 6 / kwin_wayland

Two findings, both of which change what can be promised on Linux.

**1. KWin does not offer `zwp_virtual_keyboard_v1`.** Built against enigo's
`wayland` backend alone, `Enigo::new()` fails outright: *no successful
connection*. So on KDE Wayland the only route is XTEST through XWayland, and
**a synthetic paste reaches XWayland clients but not native Wayland ones.**
The app reports this rather than pretending otherwise:

```
output  clipboard + synthetic paste (x11/xwayland only — native Wayland apps
        will not receive it)
```

**libei through the XDG RemoteDesktop portal is the answer, and it works.**
Verified on this machine 2026-08-31:

```
output  clipboard + synthetic paste (libei via the desktop portal —
        reaches every window)
```

This is the sanctioned replacement for what X11 used to give away freely: the
same ability to synthesise input, but granted by the user through a system
dialog instead of taken. It reaches native Wayland windows, which XTEST cannot.

So Linux has three routes, and the binary picks one and names it:

| route | reaches | costs |
|---|---|---|
| `KOTHA_PASTE=portal` — **libei** | every window | one permission dialog |
| `KOTHA_PASTE=1` — wayland virtual keyboard | every window | nothing, but wlroots only |
| `KOTHA_PASTE=1` — x11 XTEST *(KDE's fallback)* | XWayland clients only | nothing |
| unset — clipboard only *(default)* | everywhere, user presses paste | nothing |

**The dialog used to come back every launch. Fixed 2026-09-04** — see
"Asking once instead of every time" below.

**2. enigo sends every keystroke through *all* its live connections**, not the
first that works (`linux/mod.rs`, `impl Keyboard for Enigo`). A session with
both a Wayland and an XWayland connection therefore pastes **twice** — silently,
and only on some compositors. KDE is safe by accident, because its Wayland
connection never opens; **wlroots compositors (Sway, Hyprland) are not.**

`Settings` has no switch to disable a backend, and the session is a runtime fact
while enigo's backends are compile-time features. So `connect_keyboard()` picks
exactly one by pointing the unwanted backend at a display name that cannot
exist: Wayland first on a Wayland session, X11 otherwise. One connection, one
paste, and the binary says which it got.

*Neither of these is a reason to change the clipboard-and-paste design.* The
clipboard half works everywhere, needs no permission, and is the default.

#### What happens to what you already had on the clipboard

Pasting means *owning* the clipboard, so whatever the user copied has to be
displaced for a moment. Getting that wrong is what makes an app feel
untrustworthy: you copy a link, dictate a sentence, and the link is gone.

`Borrowed` takes it, holds it, and gives it back:

- **Text and images both.** A copied screenshot is an ordinary thing to have,
  and losing it to a dictation would cost more than the dictation was worth.
- **A fresh copy is never clobbered.** The old contents go back only if our
  dictated text is *still* what is on the clipboard. If the user copied
  something during the paste, that is theirs and it stays.
- **What cannot be saved is reported, not silently destroyed.** A file list or
  an app-private format cannot be read back, and pasting overwrites it either
  way — so the app says so rather than pretending nothing happened.
- **In copy-only mode nothing is restored at all**, because there the clipboard
  *is* the delivery: putting the old contents back would throw the transcript
  away.

The one soft spot is timing. There is no portable way to be told "the paste has
been read" — on Wayland our own process serves the selection, but arboard does
not surface that request — so the hand-back waits a fixed 400 ms. That errs
long deliberately: restoring too early would make the target paste the
*previous* text, which is worse than never restoring. 400 ms is invisible next
to a multi-second decode.

`cargo test --bin live` covers both halves against the real system clipboard:
that it is restored, and that a fresh copy survives.

---

### Phase 4 — The pill  ⟵ PASSES ON LINUX

- [x] The pill itself — `app/ui/`, four states, plain HTML and CSS
- [x] Frameless, transparent, always-on-top, **never takes focus**
- [x] Waveform at 30 fps from Rust — the stream works; the numbers in it are
      still fake, see below
- [x] Four states: appearing, listening, thinking, done. CSS only
- [x] Tray icon — Dictate and Quit. **Settings is greyed out**, because it
      belongs to Phase 5 and a menu item that does nothing is worse than one
      that says it is not ready yet
- [x] Global hotkey — `Ctrl+Alt+Space`, registers **and fires** on KDE Wayland
- [ ] Linux: stop the hotkey leaking into the focused app — see below
- [x] Wire the real chain in place of `fake_level` and the sleep
- [x] Say something into it and watch the text land — **done, 2026-08-31**
- [x] Repetition loops are now the dominant defect in real use — symptom
      collapsed in the text; the cause is still open, see below
- [ ] macOS needs an **NSPanel**, not a normal window — a normal one steals the
      cursor and the whole feature collapses (`tauri-nspanel`)

**Accept when:** the pill animates over a focused text field and the caret keeps
blinking in that field. **Met on Linux, 2026-08-31.** The text arrived in the
editor that had focus before the pill appeared, which it could not have done if
the pill had taken focus — the same event settles Phase 3 and this. Two boxes
above stay open: the hotkey leak is a defect on this platform, not a failure of
the acceptance test, and macOS has not been built.

#### The frontend is a directory, and that is the whole design

`app/ui/` is four files and no build step. There is no bundler, no framework
and no npm install; Tauri embeds the directory as-is.

| file | what |
|---|---|
| `index.html` | thirty lines of structure |
| `pill.css` | every colour, dimension and transition, as tokens at the top |
| `pill.js` | the state machine, the level shaping, and the Rust contract |
| `mock.js` | a fake backend, so the pill runs in a browser |

**The contract between Rust and the UI is two events wide**, and it only goes
one way:

```
emit("kotha://state", "idle" | "listening" | "thinking" | "done" | "error")
emit("kotha://level", <number 0..1>)          // ~30 per second
```

The UI never calls into Rust. `level` is a raw RMS with no shaping — all the
curve fitting is in `pill.js`, so making the pill look better never means
recompiling. States are a `data-state` attribute on `<html>`; every transition
is CSS. JavaScript sets that attribute and writes one custom property per bar,
and does nothing else.

`error` was added 2026-09-14. Everything below this line in Phase 4 describes
the thirty-cell waveform the pill used to draw, and is kept as the record of
how the level shaping was arrived at — that calibration carried over unchanged.
What is on screen now is in *The glyph* below.

**`mock.js` is why any of this is workable.** It notices there is no Tauri
around it and drives the same two entry points itself, so
`app/ui/index.html` opens in a browser and animates — no Rust, no model, no
build. A CSS change costs a reload instead of a ten-minute CTranslate2 compile.
It also paints a checkerboard behind the pill, which is the only way to see
whether the border and shadow hold up over light content. They did not, at
first: see below.

#### What the window gate found — 2026-08-31, KDE Plasma 6 / kwin_wayland

The pill is built and the shell around it runs, with `fake_level` standing in
for the microphone so the binary does not pull in `ct2rs`. That was the point:
everything *above* the window is already proven, and what was not known is
whether a compositor would hand over the window this app needs. Four findings.

**1. `set_position` is ignored on native Wayland.**  ⟵ *worked around*

The window stays at (0, 0) a full second after the request — the compositor
owns placement and an ordinary client cannot ask. Through XWayland the same
call lands it exactly where it belongs, so `prefer_x11()` sets
`GDK_BACKEND=x11` before GTK initialises.

It has to be Kotha's own variable (`KOTHA_GDK_BACKEND=wayland` to override)
rather than deferring to an existing `GDK_BACKEND`: a KDE session exports
`GDK_BACKEND=wayland` to every GTK app, so "respect the user's setting" would
have meant never firing on the one desktop that needs it. That cost a build to
find out.

XWayland costs sharpness under fractional scaling and nothing else — in
particular it does **not** affect text delivery, because Phase 3 settled on
libei through the desktop portal, which goes over D-Bus and does not care what
display server this process talks to.

*The proper fix is `gtk-layer-shell`* (`zwlr_layer_shell_v1`, which KWin
supports): an overlay layer with anchors, which would place the pill natively
*and* make it structurally incapable of taking focus. It must be initialised
before the GtkWindow is realised, which is inside Tauri's builder. Worth doing
when Linux is a target rather than the development machine.

**2. `set_ignore_cursor_events` aborts the process on a hidden window.**  ⟵ *fixed*

tao 0.35.3 handles the request with `window.window().unwrap()`
(`linux/event_loop.rs:457`) — the GDK window, which does not exist until GTK
realises the widget. A window created `visible: false` has not been realised,
so calling this from `setup()` panics *inside the GTK main loop*, where it
cannot unwind: "panic in a function that cannot unwind", process gone. Moved to
just after the first `show()`. Worth reporting upstream; the code already has
the `Option` in hand.

**3. `outer_size()` reads 0×0 before the window is mapped.**  ⟵ *fixed*

Which made the centring arithmetic divide by nothing, and put the pill half a
window to the right of centre. Silent, and it looks like a layout opinion
rather than a bug. Fixed by making Rust own the window size (`PILL_WINDOW`)
instead of `tauri.conf.json`, so placement never has to ask a window that
cannot answer yet. One number, one place.

**4. Where a window lands is not where it says it is.** `BOTTOM_MARGIN` is a
calibration knob, not a layout constant. At 72 the pill's bottom edge measured
75 px above the screen, which is right; the relationship is close enough to
linear that `KOTHA_BOTTOM` can retune it per desktop without a rebuild.

**What passed.** Frameless, transparent, always on top, click-through, bottom
centre of the screen, `focused = false` on every show, tray icon built, hotkey
registered. Verified by screenshotting the live desktop and measuring the
pill's position in the pixels, not by looking at it.

**What is still untested**: whether the tray menu opens, and the acceptance
test itself — a caret that keeps blinking.

**5. The hotkey fires, and also leaks into the focused application.**
⟵ *open, Linux only*

Confirmed by hand, 2026-08-31: `Ctrl+Alt+Space` raises the pill, and
`focused = false` holds. But the terminal that had focus also received the
keystroke, echoing `^[^@` — ESC from Alt, NUL from Ctrl+Space. In a text editor
that is a stray character inserted every time a dictation starts, which is
worse than no hotkey at all.

*Observed:* the key reaches both us and the focused window. *Inferred, from the
focused app having Wayland libraries mapped and `xlsclients` listing no X11
clients at all:* `global-hotkey` binds with `XGrabKey` through XWayland, and an
X11 grab has no authority over a native Wayland surface, so KWin delivers the
key to us via the grab **and** to the focused client. By the same reasoning the
grab probably does consume it when the focused window is an XWayland client —
untested, and not many windows are, these days.

**The fix is the same shape as Phase 3's.** `org.freedesktop.portal.GlobalShortcuts`
is the sanctioned route on Wayland, KDE implements it, and `ashpd` is already
in the dependency tree via enigo's libei feature. The compositor owns the
binding, the user grants it once, and it is consumed properly. It changes the
UX — the shortcut becomes something the user rebinds in system settings rather
than in Kotha — which is a Phase 5 conversation.

**macOS is not affected.** `global-hotkey` uses Carbon `RegisterEventHotKey`
there, which consumes the key. So this is a Linux defect, with the tray as a
workaround, and it is not on the critical path to the rest of Phase 4.

**F9 is now the default** (2026-09-02), for both reasons at once: it dodges the
Avro collision *and* the leak is invisible on it. The cost is that a bare
function key is easier for another application to claim, which is what the other
three entries in `HOTKEYS` are for.

**Downgraded, 2026-09-02, by trying a different key.** On F9 the defect is
invisible: Kayes ran a dictation on F9 and nothing stray appeared. That fits the
mechanism rather than contradicting it — the key is still delivered twice, but
F9 inserts no text, where `Ctrl+Alt+Space` inserts `^[^@`. So what the portal
would buy is *consumption*, and consumption only shows up on keys that type
something, or in an application that binds F9 itself (an IDE's build key). With
the hotkey now settable from the tray, the workaround is a supported choice
rather than advice, and this drops below Phase 6. It is not fixed.

#### The real chain, behind the pill — 2026-08-31

`fake_level` and the sleep are gone. The app drives
`kotha_spike::live` — the same microphone, VAD, segmenter, engine, corrector
and clipboard the Phase 2 loop ran, and the same code Phase 0 measured.

**`live.rs` was promoted from a binary to a library module** so that could
happen at all: the audio path lived inside `src/bin/live.rs`, where nothing
else could reach it. `src/bin/live.rs` is now a three-line shim over
`kotha_spike::live::run()`. This is the same move the engine made in Phase 2
and for the same reason — two copies of a pipeline drift, and the ways they
drift are silent.

Three things are now shared rather than duplicated, each because writing the
second copy would have been the beginning of a divergence:

| shared | why it had to be |
|---|---|
| `open_microphone()` | one answer to "what does Kotha do with a 44.1 kHz stereo mic" |
| `decode_threads()` | a decode at a different thread count depending on which front end started it would make every timing in PLAN.md ambiguous |
| `rms()` | the waveform is drawn from the same number the segmenter sees |

**Two things the app needs that the CLI never did:**

- **`Segmenter::flush()`.** The CLI runs until the stream ends, so an
  utterance in progress is never a problem. Pressing the hotkey to stop
  mid-sentence is completely ordinary, and without a flush that sentence is
  thrown away. The `MIN_VOICED` floor still applies, so flushing silence
  yields nothing.
- **A worker thread that owns the engine.** The model is loaded on *first
  dictation*, not at startup — a tray app that has not dictated yet has no
  business holding 1.4 GB resident — and then stays warm for the life of the
  process. Keeping it on one thread also means it never has to cross a thread
  boundary, and `cpal::Stream` is not `Send` under ALSA anyway, so the
  microphone was going to pin the loop to one thread regardless.

`toggle()` does no work: it flips a flag and posts to a channel. Everything
slow happens on the worker, because `toggle` runs on the UI thread and a frozen
pill is worse than no pill.

**What ran:** microphone opened at 16 kHz mono with no resampling, model loaded
in **0.4 s** on 6 threads, pill shown with `focused = false` at the bottom
centre, corrector loaded in 4 ms, clean shutdown. Zero utterances, because the
room was silent — **the one thing left is somebody speaking into it.**

#### First real dictation, and what it broke — 2026-08-31

Five utterances, spoken into the app on the Arch machine. It worked: English in
Latin script, Bengali in Bengali script, the corrector firing, text on the
clipboard. Four defects came out of it, three now fixed.

**1. The waveform did not move.**  ⟵ *fixed*

Not a shaping problem — the curve was measured against a real speech file and
puts the median bar at 0.58 with only 5% clipped. It was the *rate*. The app
emitted one level per capture block, and this machine's driver hands over
**2048 samples at a time — 128 ms**, so the pill updated **7.8 times a second**
against the 30 the UI is designed around. A 21-bar history took 2.7 seconds to
scroll across, which reads as frozen.

Each block is now sliced into `Microphone::level_chunk` pieces of ~33 ms, so
the animation runs at the same speed whatever buffer size the driver picked,
and every value is still the RMS of a real window. The block size is logged at
the start of each dictation, because it is a hardware decision that silently
changes how the app feels.

**2. There was no way to stop.**  ⟵ *fixed*

PLAN.md has always described the app as "press again, **or stay silent for two
seconds**, and it fades out". Only the first half was built. So when the hotkey
was missed — and on Wayland it can be, see the leak above — a dictation could
not be ended at all.

`IDLE_STOP` now ends it after 3 seconds of silence (`KOTHA_IDLE` to change it,
0 to disable). It is measured from the VAD's opinion rather than a level
threshold, so there is one definition of quiet in the program instead of two,
and it has to stay comfortably above the segmenter's own ~600 ms hangover or it
would fire between sentences.

*The Stop path itself was never broken* — that was checked deterministically
with `KOTHA_IDLE=0`, which disables the timeout and leaves the hotkey as the
only way out: `toggle stop` → `stopped on hotkey`. Every toggle is now logged,
so a hotkey that never arrived and a hotkey that arrived and did nothing stop
looking identical from outside.

**A caveat that is not fixed:** a genuinely noisy room keeps the VAD in
`speaking`, so the idle timeout never fires. One of the five utterances ran to
10.3 s for that reason. The tray menu is the reliable way out, and the VAD
threshold is not something to tune on one session.

**3. `KOTHA_PASTE=portal` was passed after the binary**, so it was an argument
rather than an environment variable and the app ran clipboard-only. Worth
recording only because the app said exactly what it was doing —
`output  clipboard only` — and that line is the reason it took seconds rather
than an evening to spot. Phase 5's settings window removes the whole class.

**4. Repetition loops, at 2 of 5 utterances.**  ⟵ *symptom repaired; cause open*

```
তোমার কথা কথা কথা কথা কথা ... ধরো ভালো আছে।
আমি কথা কথা কথা কথা কথা ... বলি।
```

Phase 2 saw this at 2 of 8 and deferred the fix by decision. At 40% of real
utterances it is now the thing most likely to make the app unusable, and it is
worse than the Phase 2 measurement suggested in one specific way: **decode time
no longer gives it away.** Phase 2's loop took 9.1 s for 3.4 s of audio; these
took 3.7 s for 3.6 s, which is unremarkable. So the cheap detector that was
proposed then — watch the clock — would not have caught either of these.

Detecting it in the *text* is still easy, and is still much easier than
preventing it. So that is what was built, 2026-08-31: `collapse_loops()` in
`lib.rs`, called from **inside `Engine::transcribe`**. Not from `emit` and
`deliver` — those are two copies of the same twelve lines, one in the CLI and
one in the app, and a defect fixed in one of them is a defect that comes back.
Every front end reaches the model through `transcribe`, so the repair sits
where they all pass.

**The thresholds are chosen to protect Bengali, not to catch every loop.** A run
of two is grammatical reduplication (`করতে করতে`, `ধীরে ধীরে`); three is
ordinary emphatic speech (`না না না`); four in a row is not something a person
says. So `MIN_RUN` is 4 and the run collapses to a *single* copy — for a real
loop the true count is one, and leaving two behind would be inventing a word the
speaker did not say, which is the failure CLAUDE.md §5 rules out. Phrases loop
as readily as single words, so the repeat unit is up to four tokens and the
shortest unit is tried first: `কথা কথা কথা কথা` is one word four times, not one
pair twice.

The raw text is logged whenever it fires, because a loop is evidence of a decode
bug and the collapsed text no longer carries it.

**This is symptom repair and it is marked as such in the source.** The cause
still wants a `repetition_penalty` sweep on the 393 with the existing strict-F1
and CER harness — which is a *paper-project* measurement and not something to
guess at from here — and `no_repeat_ngram_size` is probably the wrong knob for
Bengali, for the same reason `MIN_RUN` is 4: reduplication is grammatical, and
`করতে করতে` is correct at n=2. `ct2rs`'s `WhisperOptions` exposes both, so the
sweep is a config change and nothing more when someone runs it.

#### One trap in the workspace move

`spike/` and `app/src-tauri/` are now one workspace, so they share a target
directory and CTranslate2 is compiled once rather than twice. Moving to it
costs one full rebuild, and **an interrupted C++ build does not fail — it
leaves empty object files behind.**

A killed `cargo test` left 21 zero-length members inside
`libonednn_src-*.rlib`. Cargo's fingerprint said the crate was fresh, so
nothing rebuilt it, and `cargo test` even passed: the test binaries never
referenced the missing symbols. Only a plain `cargo build` linked something
that did, and then it failed with a wall of undefined oneDNN symbols preceded
by the actual clue, easy to miss among them:

```
ld.lld: warning: libonednn_src-*.rlib: archive member
        'jit_uni_eltwise_injector.cpp.o' is neither ET_REL nor LLVM bitcode
```

`ar tv <rlib> | awk '$3==0'` lists the damaged members. `cargo clean -p
onednn-src` does **not** fix it — it removes nothing, because the damage is in
the build-script output that CMake still considers up to date. Remove
`target/release/build/onednn-src-*`, `target/release/.fingerprint/onednn-src-*`
and the rlib by hand, then rebuild.

Nothing measured before this is affected: every earlier number came from
`spike/target`, which was never damaged. But it is worth knowing that a build
interrupted by a session ending can leave a green test suite on top of a
corrupt library.

#### On making it look like Wispr Flow's

The waveform is a **scrolling history**, not a spectrum: the newest sample
enters at the right and travels left, and the oldest bar fades out so the wave
ends in flow rather than a hard vertical cut. Every bar is a measurement that
really happened, which is the only reason it is honest to draw twenty-one of
them from one number per frame. In `thinking` there is no data at all, so the
bars switch to a self-driven travelling wave and change colour — it must not
pretend to be a signal.

Two things came out of looking at it rather than reasoning about it:

- **The first waveform was a fence.** The level curve compressed everything
  into the top of the range and the bars nearly touched the capsule edge.
  `GAIN`/`CURVE` were re-derived to map a quiet 0.02 RMS to a tenth of the
  height and a loud 0.25 to nearly full, and `--bar-max` came down.
- **Translucency cannot be what makes it legible.** At 0.72 alpha the pill
  vanished against a black background. A Tauri window is transparent, and
  `backdrop-filter` blurs *page* content, not the desktop behind the window —
  so the frosted look is a bonus where the platform gives real vibrancy (macOS
  `NSVisualEffectView`, KWin blur) and never load-bearing. At 0.92 with a
  brighter border it reads on white, black and colour alike.

---

### Phase 5 — First run

Where "easy to start" is won or lost.

- [x] Model download: 778 MB, resumable, checksum — **done, 2026-08-31**.
      The progress *bar* is not built; progress goes to stdout, see below
- [x] A first-run window that asks before spending 778 MB, and shows it
      arriving — **done, 2026-08-31**
- [ ] Permission walkthrough with buttons that open the right settings panel
- [x] A short "press this key and talk" demo — it is the last panel of the
      first-run window. Text, not an animation
- [x] Settings: paste mode (clipboard / paste / portal) — **done, 2026-09-02**,
      as a tray submenu, see below
- [x] Settings: hotkey — **done, 2026-09-02**, tray submenu, see below
- [ ] Settings: microphone

**Accept when:** a fresh user account gets from downloaded installer to first
dictated sentence without reading anything.

#### Where the model comes from, and where it lives — 2026-08-31

`setup.sh` already downloaded the model correctly, and a shipped app has no
shell script next to it, so that logic moved into the binary as `fetch_model`.
Two of the script's details came with it, both learned the hard way:

- **Resume, and retry around a dropped connection.** HuggingFace resets long
  transfers often enough that a single request is not reliable for a 775 MB
  file — Phase 0 lost one at 226 MB. Each attempt asks for a byte range
  starting from whatever is on disk, so a failure costs the remainder and not
  the whole thing.
- **Verify against HuggingFace's own manifest, not a hard-coded number.** A
  truncated or badly-resumed `model.bin` fails much later and very confusingly,
  and eyeballing the size does not catch a corrupt resume. Reading the expected
  size and hash from the API also means re-uploading the model never requires
  editing a constant here.

The manifest is fetched first and drives everything, which is a better
completeness test than `curl -C -` had: a local file is complete when its length
matches, partial when it is shorter, and junk when it is longer.

**Where it lives is now three answers**, because three different people are
asking. `KOTHA_MODEL` for someone testing another checkpoint; `./models/…` for a
developer in the repo, taken only if it really has a `model.bin` so a stale
empty directory cannot shadow a real download; and the app data directory for
everybody else. The relative path used to be the only answer, and a shipped app
launched from a menu has its working directory set to `/` or `$HOME` — so it was
an answer for nobody but case two.

**On the HTTP client.** `reqwest` looked free, because Tauri lists it. It is
Android and iOS only: on desktop it is an entire new tree — hyper, h2, tokio, a
TLS stack — for two GETs. `ureq` is blocking, which is what a worker thread that
owns the engine wants, and it gives three separate deadlines where a
total-timeout client gives one: connect and response-headers are on a clock and
the 775 MB body deliberately is not, because a slow link is not an error. It
cost 22 crates and 45 seconds of build. Not `curl` either, tempting as it was
with `setup.sh`'s invocation already proven — a shipped app should not require
a binary it does not ship on the one path where first run either works or the
user gives up.

**What is tested.** `cargo test -p kotha` fetches a 1.4 KB file from
HuggingFace, truncates it in half, resumes it, and checks the bytes match the
whole download. That is the part worth a test: the size and hash checks fail
loudly by themselves, but a resume onto the wrong offset produces a file of
exactly the right length made of the wrong bytes, and only the hash would ever
notice — after 775 MB. It talks to the real server for the same reason the
clipboard test talks to the real clipboard.

#### The first-run window — 2026-08-31

The download used to fire on the first hotkey press, which is wrong in the
obvious way: somebody pressed a key expecting to dictate and got 778 MB instead.
So there is a second window now, and it is **the pill's opposite in every way
that matters** — decorated, focusable, ordinary — because it has a button on it.

It is also where the progress bar went. The pill never needed a `downloading`
state: it is twenty-one bars of waveform floating over another application's
text, and a 778 MB transfer is not that kind of news. Three panels, one
`data-phase` attribute on `<body>`, same idea as the pill's `data-state` —
offer, downloading, ready — and the third panel is Phase 5's "press this key and
talk" demo, as text.

`worker` no longer downloads on `Cmd::Start`. If the model is missing it opens
the window and stops, and `Cmd::Fetch` — sent by the button and by nothing else
— is the only thing that spends the bandwidth.

**The pill's one-way contract survives.** `pill.js` still listens and never
calls. The first-run window is the single exception in the app and it is one
call wide, `invoke("start_download")`, for the reason above.

**Three things this cost, all found by running it rather than reading it:**

1. **`capabilities/default.json` was scoped to `windows: ["pill"]`,** so the new
   window could not listen for anything. The failure was confusing because the
   *button worked*: commands an app defines itself are not gated by
   capabilities, only core and plugin ones are, so `invoke` went through while
   every progress event was silently dropped. The window sat on "Starting…"
   while the terminal counted to 742 MB.
2. **`.bar` was already taken, twenty-one times over,** by the waveform. Putting
   the setup styles in `pill.css` to share the token block meant an unscoped
   `.bar` restyled the pill's own bars as well — one stylesheet, two windows,
   one namespace. Renamed to `.meter`, and every setup rule is now scoped under
   `body.setup` so it cannot recur.
3. **The bar stopped just short of full.** Progress was reported per chunk
   written, and the files at the end of the list are usually already complete —
   they report nothing, so the last number the window saw was short of the
   total. `fetch_model` now says `progress(total, total)` explicitly at the end
   rather than relying on the last chunk to land on it.

**What was verified**, on the Arch machine with `KOTHA_MODEL` pointed at an
empty directory: the window comes up on a cold start and not otherwise; the
button starts the download; the meter and the megabyte count track it; killing
the app at 517 MB and pressing the button again **resumed rather than
restarted**, and the completed `model.bin` still passed its sha256 — which is
the resume path proven against the real server on the real file, not a 1.4 KB
stand-in. `model.bin` came to 774,731,149 bytes. The pill's waveform still
renders, which is not a rhetorical check given finding 2.

#### Text output became a setting — 2026-09-02

The paste route was an environment variable, which is an answer for a developer
and for nobody else: a shipped app has no one to set `KOTHA_PASTE`. It is also
the setting most worth having first — getting it wrong is what cost an evening
in Phase 3.

It is a **tray submenu**, "Text output", with the three routes as check items.
Not a settings window: the window, its page, and its entry in
`capabilities/default.json` would all be new, to draw three fixed choices the
platform already draws. The real settings window is Phase 7's dashboard, and
this submenu is what moves into it.

- **`~/.config/app.kotha/settings.json`,** one JSON object, one key today. The
  hotkey and the microphone add keys rather than files.
- **`KOTHA_PASTE` still wins,** because every note in Phase 3 is written in
  terms of it and the spike binaries have nothing else. When it is set the
  submenu is titled "Text output  (KOTHA_PASTE)" and its items are disabled —
  showing what was forced beats offering a click that would not take effect.
- **The change lands on the next dictation, not this one.** The worker compares
  the setting before each dictation and reopens `Output` only when it differs:
  reopening costs a clipboard handle and, on the portal route, the permission
  dialog.
- **Check items are not radio items.** muda has no radio item, and a check item
  toggles only itself when clicked — so the handler sets all three, or the menu
  shows two ticks.

**What is tested:** `paste_choice`/`save_paste_choice` round-trip, and the two
quiet failures — an unknown mode falling back to clipboard-only instead of being
handed to `Output::open`, and a corrupt file that would otherwise panic on the
index-assign inside a tray click handler, where nobody is watching for a
backtrace.

**What was verified** on the Arch machine: the app starts and the submenu builds;
`{"paste":"portal"}` in the file alone selects the libei route with `KOTHA_PASTE`
unset; `KOTHA_PASTE=1` over that file selects the virtual keyboard; the file back
at `copy` gives clipboard-only. Clicking the submenu is the one thing not
verified from here — the write path is the tested function and the read path is
proven above, so what remains unproven is muda's tick.

#### The hotkey became a setting too — 2026-09-02

Same shape as the paste mode, and for a sharper reason: **the default collides.**
`Ctrl+Alt+Space` is fcitx's and ibus's input-method switch, which on a
Bangladeshi desktop is very likely already bound to Avro — so the users most
likely to want Kotha are the ones most likely to find its hotkey already taken.

A **fixed list of four**, not a key-capture widget. Capturing a chord needs a
focused window and a page to draw it on; what this has to solve is a collision,
not a preference. The list is only what the menu *offers* — `hotkey_choice`
validates by parsing, so any accelerator Tauri understands can be hand-written
into `settings.json`, and the menu shows it alongside the four rather than
hiding a setting the app is obeying.

- **Rebind before saving.** A key another application already owns fails at
  `register`, which is exactly the case this menu exists for. On failure the
  previous key is taken back and nothing is written — the one outcome that must
  not happen is Kotha ending up with nothing bound.
- **Junk in the file falls back to the default** rather than leaving the app
  unbindable.
- The shortcut handler no longer compares against a constant. Exactly one
  shortcut is registered at a time — the menu unregisters before it binds — so
  whatever arrives is the one.

`paste_choice`/`save_paste_choice` generalised into `setting(path, key)` and
`save_setting(path, key, value)` on the way; the two settings share one file and
the test now checks that writing one does not drop the other.

The first-run window's "Ready" panel no longer hardcodes the key. It asks
Rust (`hotkey_label`), which answers with the bound accelerator or `null` — and
`null` is the case that matters: if another application already owns the key,
telling a brand-new user to press it is Phase 5's acceptance test failing on the
last screen. The panel then names the tray's Dictate item instead and says why.
That makes the first-run window two calls wide rather than one; the pill's
one-way contract is untouched.

**What was verified** on the Arch machine: the default binds; `Alt+Shift+D` in
`settings.json` binds instead, and **pressing it actually starts a dictation**
(`xdotool key alt+shift+d` → `toggle start`); `Ctrl+Banana` falls back to the
default; and with `Alt+Shift+D` bound, the old default fires nothing, so no stale
registration is left behind. The first-run window was seen naming `Alt Shift D`
from the settings file, and both branches of the panel's sentence were checked in
a browser against `app/ui/setup.html`. Clicking the submenu is the untested step —
the rebind path's failure branch in particular has not been seen live.

#### Two defects Kayes found by using it — 2026-09-02

**1. The dictation could never end on its own once it had started.**

Reported as: silence at launch is fine and the pill goes away, but after
speaking, the pill keeps listening and keeps typing sentences that were never
said, and only F9 stops it. The repetition collapser never fired, because these
hallucinations are fluent and different every time.

Two causes, both needed:

- **Whisper does not answer silence with silence.** Given breath, a fan or a
  keyboard, it returns a sentence from its training data. The VAD opens on 48 ms
  of anything it calls voice and keeps a chunk at 300 ms, so a breath is enough.
- **Every utterance reset the idle timer, including the ones that were
  nonsense.** So each hallucination bought another three seconds, and the loop
  sustained itself indefinitely.

The fix is a confidence gate in `Engine::transcribe`, so both front ends get it,
plus `deliver` now returning whether anything was actually said — a discarded
chunk no longer counts as speech and no longer restarts the timer.

**`no_speech_prob` does not work on this model, and that is worth knowing.**
It is the standard filter — faster-whisper drops a segment above 0.6 — and on
eight synthetic noise files including *digital silence* it came back between
0.0000 and 0.0003. The fine-tune's training data is all speech, so it never
learnt to emit `<|nospeech|>`; the token is still there and its calibration is
gone. Anyone reaching for that threshold on a Whisper fine-tune should measure
it first.

The average log-probability does separate them:

| | n | worst | median | best |
|---|---|---|---|---|
| real speech | 40 | **-0.142** | -0.040 | -0.008 |
| synthetic noise | 8 | -1.047 | -0.35 | **-0.198** |

Real speech is 40 random clips from the **training** corpus, chosen so that no
threshold is ever fitted to the paper's 393. Noise is synthetic: hiss at three
levels, 50 and 60 Hz hum, clicks, a breath-shaped envelope, digital silence.

`LOGPROB_FLOOR = -0.20` sits in the gap, biased towards keeping speech: 0.058
below the worst real utterance, and it rejects seven of the eight noise files
(the eighth, 50 Hz hum, misses by 0.002). Verified afterwards: **40 of 40 real
clips kept, 0 discarded; 7 of 8 noise files discarded.** Real rooms are not
synthetic noise, so `KOTHA_MIN_LOGPROB` overrides it without a rebuild and
`KOTHA_MIN_LOGPROB=-99` turns it off. The confidence is now printed for every
utterance, kept or not, because tuning the floor needs to see both sides.

**2. The pill dimmed the window behind it and took its caret.**

Reported as: after placing the cursor and starting a dictation the cursor is
gone and has to be placed again, and the whole background darkens as if
something is focusing attention on the pill.

Nothing was drawing that. **KDE's "Dim Inactive" effect was on**
(`diminactiveEnabled=true` in `kwinrc`), and it was dimming the editor because
the editor had stopped being the active window — which is also why its caret
stopped. One cause, both symptoms.

`set_focusable(false)` was not enough. It maps to GTK's `accept_focus`, which
stops the pill taking *keyboard* focus, and `is_focused()` duly reported
`false` — the very reading Phase 4 recorded as "evidence rather than proof". It
was evidence for the wrong proposition: KWin still made the pill the **active
window**. Measured with `xdotool getactivewindow`:

```
  300 ms  active=2097152   (the editor)
 4200 ms  active=16777229  Kotha       ← the pill takes over
 6000 ms  active=2097152   (back)
```

The cure is the X11 window type hint, which is what this window always was:
`Notification`. No window manager promotes a notification to active. `tao` does
not expose type hints, so `no_activate` reaches through Tauri's `gtk_window()`,
which is why `gtk` is now a Linux-only dependency for exactly one call. After
the change the active window does not move at all while the pill is up.

No shadow was added: the pill already has two (`--pill-shadow`). The darkening
was never Kotha's.

---

### Phase 6 — Ship

- [x] Prove one bundle locally before writing any CI — **`.deb` and `.AppImage`
      both build and run, 2026-09-02**. See below: the AppImage is proof of
      mechanics only, and must be built on the oldest base in CI
- [x] GitHub Actions matrix: `.dmg` (Intel + Apple Silicon), `.msi`,
      `.AppImage`, `.deb` — **written 2026-09-04, never run**. See below
- [ ] Tauri updater (free, no certificate needed)
- [ ] One-page download site
- [ ] AUR entry — **recipe written and proven locally 2026-09-04**
      (`packaging/PKGBUILD`). Licence, tag and release all done. Blocked on two
      things outside the code: the repo is private, and AUR registration is
      down. See below
- [x] Install it the way a user would — **2026-09-04**, first time ever. See
      "Kotha had never actually been installed"

#### What the binary is made of — 2026-09-02

Measured before writing a line of CI, because the CI matrix is only worth
building on top of a bundle that works, and two things about this binary could
have made bundling hard.

Neither did.

- **CTranslate2 is statically linked.** `ldd` lists no `libctranslate2.so`; the
  engine is inside the executable. That is why it is 94 MB — 68 MB of `.text`,
  which is CTranslate2 compiling a kernel per ISA (SSE4.1, AVX, AVX2, AVX-512
  are all present, selected at runtime). It also means there is no shared
  library to ship alongside, and no `LD_LIBRARY_PATH` problem in the bundle.
- **The only numeric library from outside is `libgomp.so.1`,** OpenMP's runtime.
  Everything else in `ldd` is GTK, WebKit and their transitive dependencies —
  what any Tauri app on Linux needs. On Debian that is `libgomp1`, and it must
  be declared: Tauri's deb bundler does not read `ldd`.

**And the size claim in CLAUDE.md §3 survives.** 94 MB unstripped, 78 MB
stripped, **13 MB under `xz -9`** — which is what a `.deb` compresses with. A
statically linked inference engine turns out to cost very little once compressed,
because a per-ISA kernel is highly repetitive code.

Also noted for the M2: **oneDNN is compiled in** (745 references) and so is
`ruy` (6). CLAUDE.md §7 says the M2 wants `ruy` rather than `dnnl`, so that is a
runtime selection to make there, not a rebuild.

#### The first bundle — 2026-09-02

`cargo tauri build --bundles deb` produces **Kotha_0.1.0_amd64.deb, 23.5 MB**.
The gate question was whether the packaged binary runs away from the build tree,
and it does: extracted to a temp root and launched from `/` with no environment
at all, it starts, finds its frontend (Tauri embeds it), falls back to the app
data directory for the model, and loads the corrector. That is the case PLAN.md
worried about when `model_dir` grew its third answer — a menu launch has no
useful working directory — and it now has a run behind it rather than an argument.

Four things the first `.deb` got wrong, all invisible until one was built:

1. **`libgomp1` was not in `Depends`.** Tauri's deb bundler does not read `ldd`;
   it ships a fixed list of GTK and WebKit packages. OpenMP's runtime is the one
   thing CTranslate2 needs from outside the executable, and a Debian box without
   it would install Kotha successfully and then fail to launch it.
2. **`Categories=` was empty**, so the app would install and appear in no
   application-menu category at all. `bundle.category` fixes both this and the
   macOS one.
3. **`Description: (none)`** — literally that string, in every package manager
   that lists it.
4. **`Maintainer: kotha`**, which is the package name standing in for a person.
   Still open: it wants a name and an email, and both are Kayes's to choose.

**Not stripping the binary, deliberately.** `--strip-all` takes it from 94.4 MB
to 77.5 MB, but almost all of that is the symbol table rather than debug info,
and it is highly compressible: the download only falls from 14.2 MB to 13.0 MB.
Paying 1.2 MB — 5% of the `.deb` — to keep function names in every crash report
is the right side of that trade for an app whose author diagnoses from logs.

#### The AppImage, and why CI cannot build it on Arch — 2026-09-02

**`Kotha_0.1.0_amd64.AppImage`, 112.7 MB**, and it runs: launched from `/` with
no environment it starts, binds F9 and loads the corrector, same as the `.deb`.
It is five times the `.deb` because an AppImage carries GTK and WebKit itself
rather than depending on them, which is the whole point of the format.

Two obstacles, neither about Kotha:

- **The bundler's tools would not download.** It fetches four helpers into
  `~/.cache/tauri/` on first use and the 13 MB `linuxdeploy-x86_64.AppImage`
  timed out repeatedly. `curl` fetched it in one go. The other two — the GTK and
  GStreamer plugin scripts — turned out to be **vendored inside the
  `tauri-bundler` crate**, so they were copied from `.cargo/registry` and never
  downloaded at all.
- **`linuxdeploy` ships its own `strip`, and it is too old for this distro.**
  Every bundled library failed with ``unknown type [0x13] section `.relr.dyn` ``
  — DT_RELR relative relocations, which current Arch libraries all use and which
  that binutils predates. `NO_STRIP=1` skips the step and the build completes.
  That is also why the artifact is 112.7 MB rather than smaller.

**The finding that matters for Phase 6: this artifact is not shippable, and no
AppImage built here ever will be.** `readelf -V` puts the binary's floor at
**GLIBC_2.39**, so it would refuse to start on anything older than Ubuntu 24.04
— from a format whose entire promise is that it runs anywhere. An AppImage takes
its glibc floor from the machine that builds it, so the CI matrix must build it
on the **oldest** base we intend to support (`ubuntu-22.04`, glibc 2.35), not on
whatever is newest. The same old base makes `linuxdeploy`'s `strip` work again,
so `NO_STRIP` is a local workaround and should not go in the workflow.

What this run proves is the mechanics — config, icons, desktop entry, embedded
frontend, a binary that runs detached from its build tree. The shippable
artifact comes from CI.

**One warning worth carrying forward:** re-running the bundler against an
already-patched binary logs `__TAURI_BUNDLE_TYPE variable not found`, which
leaves the updater unable to tell how it was installed. Build every target in a
single `cargo tauri build --bundles ...` invocation rather than one per run.

#### The CI matrix — 2026-09-04

`.github/workflows/release.yml`, one job across four runners. A manual run
(Actions → Release → Run workflow) builds and uploads artifacts; a `v*` tag does
the same and opens a **draft** release. The difference is one expression — with
no tag, `tagName` is empty and `tauri-action` creates nothing.

**It has never been run.** Three of its four legs build for platforms this
project has never compiled on. That is the point of dispatching it: the source
is already cfg-gated for them — `connect_keyboard`, `prefer_x11` and
`no_activate` each have a `#[cfg(not(target_os = "linux"))]` twin, and every
Linux-only crate dependency is target-gated inside its own manifest — so there
is a real chance the other three compile untouched. Nobody knows yet.

Four decisions in it worth keeping:

- **`ubuntu-22.04`, not the newest runner.** An AppImage inherits its glibc
  floor from the machine that built it. This is the finding from the local
  bundle above, and it is the one thing in the workflow that would be silently
  wrong if changed to `ubuntu-latest`: the artifact would still build, still
  run on the runner, and refuse to start on Ubuntu 22.04.
- **No `NO_STRIP`.** It is an Arch workaround, and 22.04's `linuxdeploy` strips
  fine. Setting it in CI would cost ~30 MB for nothing.
- **`macos-15-intel` and `macos-15`.** `macos-13` — the label every older Tauri
  workflow uses for Intel — has been retired, and `macos-14` is deprecated.
- **One `cargo tauri build` per platform, all bundles in it.** Re-running the
  bundler against an already-patched binary breaks the updater's install-source
  detection.

The cache is doing the real work here. CTranslate2's object tree lives under
`target/<triple>/release/build/ct2rs-*/out`, and `Swatinem/rust-cache` keeps it;
a cold matrix is close to an hour of runner time, a warm one a few minutes.

**Neither the `.dmg` nor the `.msi` will be signed,** so both will greet the
first user with Gatekeeper or SmartScreen. Apple's certificate is $99/year and
Windows' is more; whether Kotha buys them is Kayes's call and not a blocker for
a first release.

---

#### Asking once instead of every time — 2026-09-04

Kayes reported the *"allow remote control?"* dialog on every launch. It was the
one caveat Phase 3 left open and marked "worth fixing before shipping", and it
is now fixed.

**What it was.** The portal offers a `restore_token`: say yes once, get a token
back, hand that token in next time and the portal restores the same grant with
no dialog. enigo 0.6.1 passes `None` for it — a hardcoded `None` with a `TODO`
beside it in `linux/libei.rs` — so every launch was a fresh ask.

**The fix.** Upstream enigo has since added the round trip:
`Settings::restore_token` in, `Enigo::restore_token()` out (behind the
`platform_specific` feature). It is not in a crates.io release, so `spike`
pins the git commit `a88d9b7`. `Output` grew a public `restore_token` field,
and `open_output` in the app reads the token out of `settings.json` before
connecting and writes the new one back after. `libei_smol` became `libei` +
`smol` upstream, which is the only other change the bump needed.

**Measured, same machine, KDE Plasma 6 / kwin_wayland:**

| | time to a working connection |
|---|---|
| no saved token — dialog, human clicks Allow | 5846 ms |
| saved token replayed | **215 ms** |

**The token rotates.** Two runs produced `zEvPcKgbeJMv3kDOxD97cA` and then
`_3iiefJQ7yQCEJ05Ezqj0g`. Saving it once would work until it did not, so it is
written back on every open, not only the first — that is why `open_output`
exists rather than a one-time write at startup.

It is stored in `settings.json` beside `hotkey` and `paste`, keeping settings to
one file. It is not a secret: it names a grant this user already made to this
app on this machine, and is worthless anywhere else.

**This changes which mode is the sensible one on KDE.** *Paste at the cursor*
uses XTEST and reaches XWayland windows only; *Paste at the cursor (portal)*
reaches everything and now costs one dialog ever instead of one per launch.
Whether it should become the default is a product call and Kayes's — the
argument against it is still CLAUDE.md's, that synthetic keystrokes should be
asked for rather than assumed.

---

#### Getting it into the AUR — 2026-09-04

`packaging/PKGBUILD` is the recipe. It is `kotha-bin`, and there is deliberately
no source package: building Kotha from source compiles CTranslate2 and oneDNN,
which is 15-30 minutes, and nobody installing a dictation app should wait for
that.

**It unpacks the `.deb`.** Arch and Debian want the same files in the same
places — `/usr/bin/kotha`, one `.desktop`, four icons — so the release carries
one artifact and both distributions eat it. makepkg's bsdtar reads `ar`
archives, so the `.deb` arrives already split and `package()` is one `tar -xf`.

Dependencies were read off the built binary rather than guessed. The only
non-obvious one is **`libayatana-appindicator`**, which does not appear in
`ldd`: Tauri's tray opens it with `dlopen` at run time, so nothing links it and
nothing warns when it is missing. Without it there is no tray icon.

**Built and checked locally**, against the real `.deb` with `source` pointed at
a local file: `makepkg` produces `kotha-bin-0.1.0-1-x86_64.pkg.tar.zst`,
20.6 MB, containing `/usr/bin/kotha`, the `.desktop` and four icons.

That test earned its keep immediately. **makepkg strips binaries by default**,
and it took Kotha from 99 MB to 81 MB — undoing the deliberate decision in "The
first bundle" not to strip, which costs 1 MB of download and buys function names
in every crash report. `options=('!strip' '!debug')` restores it; `!debug` also
stops makepkg building an empty `-debug` package out of what it removed.

One cosmetic wart inherited from Tauri's deb bundler: it writes an icon into
`hicolor/256x256@2/`, which is not a directory name the icon spec knows —
`@2x` is. That icon is simply ignored. Not worth a patch.

**The release exists.** `v0.1.0`, tagged on `main` 2026-09-04, carrying
`Kotha_0.1.0_amd64.deb` (24,657,996 bytes, sha256 `90b41b7a…`). That hash is in
the PKGBUILD.

**Two things still block publishing:**

1. **The repository is private.** This is the one that matters, and it was found
   by curling the release URL rather than trusting that it worked: an anonymous
   download 404s, while `gh release download` — which authenticates — fetches
   the same asset and its hash matches the local build byte for byte. So the
   artifact is fine and the URL is right; it is simply not reachable by anyone
   who is not Kayes. **A PKGBUILD cannot fetch from a private repository**, so
   `kotha-bin` would fail on every machine but this one. The repo has to go
   public first, and publishing the source is Kayes's call, not a packaging
   step.
2. **An AUR account with an SSH key**, which only Kayes can create — and as of
   2026-09-04 **AUR registration is down**, so this is blocked from outside.
   Nothing to work around: the AUR only accepts pushes over SSH from a
   registered account. Wait for it.

Then, on this machine:

```
cargo tauri build --bundles deb          # the artifact for the release
cd packaging && updpkgsums               # fills in the sha256sums
makepkg -si                              # proves it installs before anyone else tries
git clone ssh://aur@aur.archlinux.org/kotha-bin.git
cp PKGBUILD kotha-bin/ && cd kotha-bin
makepkg --printsrcinfo > .SRCINFO        # the AUR rejects a push without this
git add PKGBUILD .SRCINFO && git commit -m "kotha-bin 0.1.0" && git push
```

Every later version is the same three lines: bump `pkgver`, `updpkgsums`,
regenerate `.SRCINFO`, push.

---

---

#### The repository became the product — 2026-09-04

Kayes is going to make the repository public once he is happy with it, and
asked that what people find there be a finished product rather than a workshop.
Three things followed.

**The working notes left the repository.** `CLAUDE.md` and `PLAN.md` — this
file — are no longer tracked. They are still on disk and still the source of
truth for state; they are simply `.gitignore`d. `README.md` is the only
markdown in the repo now.

**The history was rewritten**, because deleting the files only hides them from
the tip: a public clone would still carry every old version, and 22 of 31
commits carried a `Co-Authored-By: Claude` trailer. `git filter-repo` removed
both. Verified afterwards: zero mentions of Claude in any message or author
field across all 30 remaining commits, and no trace of either file at any
revision. The `v0.1.0` tag survived and still points at the right tree.

**From here on, commit messages carry no Claude attribution.** That is Kayes's
call about his own repository and it stands.

**The full pre-rewrite history is in `.backup/`** (git-ignored): a mirror clone
plus copies of both notes and the handoff. Delete it once the new history has
been on GitHub long enough to trust.

**Two follow-ups that were nearly missed:**

- **23 comments across 8 source files cited `CLAUDE.md §5` and `PLAN.md Phase 4`
  by name.** With those files gone from the repo, every one of those pointers
  became a dead end for anyone reading the source. Each now carries the point it
  was making instead of a reference to where the point was written down. This is
  the cost of citing a document from code, and worth remembering before writing
  another such comment.
- **The `README.md` claimed macOS and Windows support.** Neither has ever been
  built. It now says Linux, because a first-time reader believing otherwise is
  worse than a shorter list.

**Still pending:** the force-push. `git filter-repo` removes the remote, it has
been re-added, and `git push --force origin main linux-app` plus the same for
`v0.1.0` is what actually makes any of this true on GitHub. Until that runs, the
old history is still up there.

#### Kotha had never actually been installed — 2026-09-04

Kayes asked why Kotha was not in his application menu. It was not installed:
every run in this project's life, from Phase 2 onwards, has been
`./target/release/kotha` out of the build tree. No `/usr/bin/kotha`, no
`.desktop`, no icon, `pacman -Q kotha` empty.

Worth naming because it hid a whole class of question. "Does the packaged app
find its model, its frontend, its settings?" had been *argued* from an extracted
`.deb` launched from `/`, which is good evidence, but nobody had ever used Kotha
the way a user would.

`target/pkgbuild-local/` now holds a package built from the current tree with
`source` pointed at the local `.deb` and `makepkg --skipchecksums` (the real
PKGBUILD pins the *release* hash, which a fresh local build will never match).

**All seven `depends` were verified present on this machine** — which is the
first real check of that list against a system, rather than against `ldd`.

One thing to carry: once installed, launch from the menu, not the build tree.
Two copies cannot both hold the hotkey, and the second one's failure to bind it
is expected rather than a bug.

#### The float32 model: converted, measured, dropped — 2026-09-04

Kayes asked for a second, better model for users with a GPU. The reasoning was
sound and is written in CLAUDE.md already — int8 loses to fp16 on a GPU at batch
size 1, because the card pays a dequantisation cost. What that reasoning does
not survive is contact with what is actually publishable.

**`kayees/whisper-medium-bn-en-cs` cannot be loaded by Kotha.** It is
`model.safetensors`, a transformers checkpoint; `ct2rs` wants CTranslate2's
`model.bin`. Different format, not a policy.

So it was converted — `ct2-transformers-converter --quantization float32`,
producing a 3.05 GB CTranslate2 model. Two checks before anything else:
`tokenizer.json` and `vocabulary.json` came out **byte-identical** to the
shipped model, and `config.json` matched on the one field that matters,
`suppress_ids: []` — the catastrophic silent bug did not appear, because the
fine-tune's own generation config already carries empty lists.
`preprocessor_config.json` differed by one cosmetic key (`processor_class`) and
was made to match.

**Then it was measured, 8 training clips, same machine, same binary:**

| | 98.4 s of audio decoded in | |
|---|---|---|
| int8 (shipped) | **68.2 s** | 1.44x real time |
| float32 | **125.9 s** | 0.78x — slower than speech |

**1.85x slower, and across the line that matters** — it cannot keep up with a
person talking, which for a dictation app is the whole game.

Quality was a wash. Six of eight clips differed; `singapore`/`singapoor` and
`30 লাখ`/`30 4 লাখ` favoured int8, `was really good`/`goode` favoured float32,
`insudent`/`incudent` were both wrong. The one that decided it: float32 rendered
a company name as **`মোসালেব`** where int8 wrote **`mossalab`** — Bengali script
for an English word, which is the exact failure this entire project exists to
fix.

**Eight clips is not a measurement and must not be quoted as one.** It is enough
to say the premise did not hold: int8 is not losing accuracy worth recovering,
and the ~3 GB download would buy users a slower app.

Deleted, along with the conversion venv and the cached download — 4 GB back on a
disk that was at 95%.

**What is left of the idea, and where it belongs:** the real question — *how
much accuracy does int8 quantisation cost?* — is a paper measurement on the 393,
not a product one, and CLAUDE.md §2 says to stop at that line. Kayes has Kaggle
T4s, so it is answerable without an NVIDIA card in this desktop. **And even a
win there would not obviously change Kotha:** using it needs an NVIDIA card in
the *user's* machine and a separate CUDA build, and most people running a Bangla
dictation app are on a laptop with no such card.

---

### Phase 7 — Polish

- [x] **The dashboard.** A real settings window — see below. It did not grow
      out of the tray menu, it *replaced* it: the Hotkey and Text output
      submenus were deleted the day it landed
- [~] Language mode — **built, measured, removed.** See below. The checklist
      entry stays struck rather than deleted so it is not proposed a third time
- [x] Dark and light themes, across all three windows
- [ ] Choose a different Whisper model
- [ ] Push-to-talk as an alternative to toggle
- [ ] History window
- [ ] Start/stop sounds
- [ ] Bigram context for the corrector (SymSpell supports it; still sub-ms)
- [ ] Per-app "never dictate here" list

#### The settings window, two languages and a light theme — 2026-09-14

Three things at once because they are one thing: a place to put settings, a
new setting worth putting there, and the first setting that has to reach every
window at the same time.

**The tray menu was deleted, not extended.** It is Dictate / Settings… / Quit
now. The two submenus it lost were three synchronisation problems wearing a
trench coat: a `CheckMenuItem` toggles only *itself* when clicked, so each
group had to hand-set every sibling's tick or show two at once; the bound key
had to be written back into the "Dictate" label; and the whole lot was a second
place that had to agree with `settings.json` about what the app was doing. The
window is a net deletion in `main.rs` and it can do what a check item cannot —
put a sentence under each option saying what it is for.

**`settings_get` sends the option lists, not just the values.** The hotkeys,
paste routes, languages and themes are Rust constants, and the reply carries
all four lists along with the current choice. Nothing in `app/ui/` knows what a
language code is. A list written into the HTML would be a list that drifts from
what `settings_set` will accept, which is exactly how a settings window starts
offering something that silently does nothing.

**The language switch was built and then removed the same day, and the
removal is the finding.** It shipped complete — a `language` key, a worker read
per dictation, a `language` parameter on `Engine::transcribe`, a radio group —
and Kayes caught it immediately: he picked English and still got good Bangla,
which is not what English mode should sound like.

It was not a bug. Five test clips, decoded twice through the same binary with
`KOTHA_LANG=bn` and `KOTHA_LANG=en`: **three came back byte-identical**, and
the two that differed differed like this —

```
bn: আল্লাহামদুলিল্লাহ এটা একটা great opportunity
en: alhamdullah         এটা একটা grate opportunity
```

Still Bangla. One word pushed out of Bangla script, and "great" misspelled.
The token reaches the model — the average logprob moves, -0.058 to -0.056 — and
the model does not care: it is fine-tuned on `<|bn|>` hard enough that the
language embedding is inert. The switch was offering a choice between Bangla
and slightly worse Bangla, so it came out: `Engine::transcribe` pins `<|bn|>`
again, and the measurement is in the comment above that prompt so nobody
rebuilds it.

**The lesson worth keeping is not about Whisper.** Every layer of that feature
was correct — the setting saved, the worker read it, the prompt changed, the
logprob moved. Correct wiring is not the same as a feature existing. Measure
the decode before building the UI on top of it.

A first diagnosis blamed the wrong thing, and that is worth writing down too:
`/usr/bin/kotha` is the AUR `kotha-bin` 0.1.0, so the obvious theory was that
the dictation had gone through the old binary that has no language support.
It had not. The theory was cheap and wrong, and the two-minute A/B against the
CLI was what actually answered it.

**The theme is two palettes, not three.** `app/ui/theme.js` resolves `system`
against `prefers-color-scheme` and writes a concrete `data-theme` on `<html>`,
so `pill.css` has a dark `:root` and a light override and no media query
anywhere. It runs from `<head>` in all three pages — from `<body>` it would
paint one frame of dark before correcting itself on a light desktop.

Rust tells the pill with a third event, `kotha://theme`, emitted on change
*and* again inside `show()`. The second one is the point: the pill is hidden
between dictations and its contract is one-way, so it cannot ask. Being told
again every time it is about to be seen costs one event per dictation and
removes the whole class of "the pill is the wrong colour until you restart".

In light mode the glyph **darkens** with the voice and the bloom becomes a soft
shadow. That is not a compromise, it is the correct inversion: the pill floats
over content it does not control, and a capsule lighter than the page behind it
needs its marks to be the dark thing. Verified over a white-to-near-black
gradient in the browser mock (`t` cycles the theme), in all four states.

**The icon became the k-and-matra mark.** A Latin lowercase `k` under the
horizontal headline stroke that joins the letters of a Bengali word — the app
in one glyph, and it inverts cleanly for the light theme and flattens cleanly
to a macOS template image. Drawn as geometry in `icons/kotha.svg` rather than
set in a font, because the smallest cut is 32 px. One non-obvious number in it:
a butt cap is cut square to its *own* stroke, so a diagonal told to stop at the
baseline pokes about 20 px through it — the arm and leg endpoints are pulled
back by half the stroke width times the stroke's horizontal component. The
template PNG was re-checked the way the old one was: 36×36, every opaque pixel
`#000000`, corners fully transparent.

**The permission dialog on every launch was a one-line bug, found the same
day.** `Output::open` reads a portal restore token off whichever backend
connected and saves it — `fresh_token` asks the *connection* and does not care
which branch opened it. But `connect_keyboard` only passed a saved token back
on the `portal` branch, because a comment claimed the token "only means
anything on the portal route". It does not: on KDE Wayland the plain `paste`
route reaches the same libei, and the desktop asks for the same grant. So the
token was written to `settings.json` on every launch and handed back on none of
them, and the dialog came up every single time. Every branch takes it now.

The two outcomes are indistinguishable from inside the process — the desktop
either restores the grant silently or re-asks, and never says which — so
`Output::open` now logs whether it offered a saved token. A dialog on a launch
whose log says *offering the saved permission* is a refused token, which is a
different bug from never having sent one.

**Two open ends.**

- The **first-run window does not follow an explicit theme choice**, only the
  desktop. It is shown before `settings.json` exists, where `system` is the
  right answer anyway; the gap only shows if it is reopened later because the
  model is missing. Left alone rather than given a fourth emit for a window you
  see once.
- **No tray, no settings.** Everything is reachable through the tray icon and
  nothing else, so a desktop with no StatusNotifier host plus a hotkey another
  application owns is an app that cannot be used *or* configured. Not seen
  here — the tray builds fine on KDE — so it is noted rather than fixed.

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
| Hotkey | `Ctrl+Alt+Space`, wired and registering. Still not final |
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
| Strict/tolerant gap that is spelling | **93.6%** (545 of 582 tokens) | 2026-08-31 |
| Misspellings reachable at SymSpell ED≤2 | 84.8% (462 of 545) | 2026-08-31 |
| Misspellings that are real English words | 26.4% — invisible to a dictionary | 2026-08-31 |
| English tokens dropped entirely | 11.9% (510 of 4299) | 2026-08-31 |
| Corrector latency per sentence | **5.1 ms** — Rust, brute-force scan | 2026-08-31 |
| Rust corrector vs Python prototype | **392/393 lines identical**; the one difference is an exact tie the prototype breaks by hash order | 2026-08-31 |
| Baked dictionary | 50,264 words, 860 KB, `include_str!`d | 2026-08-31 |
| Strict English-F1, before corrector | **71.81** (spike, int8, 393 utts) | 2026-08-31 |
| Strict English-F1, after corrector | **77.02** (+5.21, held out) | 2026-08-31 |
| Non-Latin tokens modified by corrector | **0** of 393 utterances | 2026-08-31 |
| Misspellings corrected | 236 of 545, ~91% precision | 2026-08-31 |
| Live loop, utterances per model load | 3, load 0.3 s — Ryzen | 2026-08-31 |
| Live decode, 7 s utterance — Ryzen, 6 threads | 5.7 s (1.26× realtime) | 2026-08-31 |
| Mic format taken on Arch/PipeWire | 16 kHz mono I16, no resampling | 2026-08-31 |
| False triggers in 14 s of room silence | 0 | 2026-08-31 |
| Live speech, utterances per model load | 8, one load | 2026-08-31 |
| Live utterances lost to repetition loops | 2 of 8 — known, deferred | 2026-08-31 |
| Pill window takes focus | **no** — `focused = false` on every show | 2026-08-31 |
| `set_position` on native KDE Wayland | **ignored** — stays at (0, 0) | 2026-08-31 |
| `set_position` through XWayland | honoured, bottom centre | 2026-08-31 |
| Pill bottom edge above the screen, margin 72 | 75 px, measured in the pixels | 2026-08-31 |
| Tauri shell build, no engine | 2m 02s cold, 24 s warm | 2026-08-31 |
| Hotkey fires on KDE Wayland | **yes** — and leaks to the focused app | 2026-08-31 |
| Model load, from the app | 0.4 s, 6 threads — Ryzen | 2026-08-31 |
| Corrector load, from the app | 4 ms | 2026-08-31 |
| Capture block size — Arch/PipeWire | 2048 samples, 128 ms | 2026-08-31 |
| Waveform update rate, before the fix | 7.8/s against a design of 30 | 2026-08-31 |
| Live utterances lost to repetition loops, in the app | **2 of 5** | 2026-08-31 |
| Decode confidence, real speech (40 training clips) | -0.142 worst, -0.040 median | 2026-09-02 |
| Decode confidence, synthetic noise (8 files) | -1.047 to -0.198 | 2026-09-02 |
| `no_speech_prob` on digital silence | **0.0000** — uncalibrated by the fine-tune | 2026-09-02 |
| Pill steals the X11 active window | **yes** before the type hint, no after | 2026-09-02 |
| `.deb` | 23.5 MB | 2026-09-02 |
| `.AppImage` (GTK + WebKit inside, unstripped) | 112.7 MB | 2026-09-02 |
| glibc floor of an Arch-built binary | **GLIBC_2.39** — Ubuntu 24.04 or newer | 2026-09-02 |
| App binary, release, unstripped | 94 MB | 2026-09-02 |
| ...stripped | 78 MB — 68 MB of it `.text` | 2026-09-02 |
| ...stripped and xz -9 | **13 MB** | 2026-09-02 |
| CTranslate2 in the binary | **static** — no `libctranslate2.so` in `ldd` | 2026-09-02 |
| Numeric libraries from outside | `libgomp.so.1` only | 2026-09-02 |
| Misspellings left standing | 309 of 4,299 English tokens | 2026-08-31 |
| ...of those, protected by the floor | **90.0%** — real words, need context | 2026-08-31 |
| ...reachable by a phonetic fallback | **17 tokens, ~+0.2 F1** — not built | 2026-08-31 |


---

## The glyph — 2026-09-14

### What is on screen

Five bars of light in a row, tapering out from the middle — 12 / 20 / 32 / 20 /
12 px, 4 px tall, fully rounded, in a 170x36 capsule. **They never move and
never resize.** Everything the pill says is said with brightness and colour.

| state | the glyph |
|---|---|
| `idle` | nothing on screen |
| `listening` | bars follow the voice, filling outward from the middle. A silent room leaves five bars at `--rest`, which is what "armed" looks like |
| `thinking` | bars drop to the dim ink and keep following the voice; a red chase runs across them once a second |
| `done` | every bar to full white at once — the only moment the glyph is evenly lit, because the voice always lights the middle harder |
| `error` | the row goes red and the middle bar goes out, leaving a visible break |

The capsule grows out of a horizontal line when it arrives and falls back into
one when it leaves. That is the only movement left in the app.

**Why it replaced the waveform.** Thirty cells scrolling right to left, each a
real measurement, was honest and was far too much to have flickering at the
bottom of the screen while someone is trying to think of what to say next.
Motion draws the eye and then holds it; light can be read out of the corner of
one. The filling is outward from the middle and not left to right on purpose:
left to right is a meter, and a meter invites you to read a value off it, which
is not a thing anybody needs here. It also keeps the decode chase, which *does*
run left to right, unmistakably a different thing.

**There is no `speaking` state, and there should not be.** It would be one bit
derived from the same level number by a guessed threshold, so it cannot arrive
earlier than the level it comes from, and it would add a boundary to flicker
at. The continuous number is strictly more information at the same moment.

**`error` closes a real gap.** A failed dictation used to emit `idle` and
vanish without a word — indistinguishable from the hotkey not registering. The
comment in `main.rs` already said the user "deserves to be told", and nobody
running the bundled app is watching stderr.

### The delay between voice and light

Measured, ~140 ms down to ~30 ms. Two terms, both real:

| term | before | after |
|---|---|---|
| capture block, mean staleness | ~48 ms | ~7 ms |
| Tauri IPC | ~3 ms | ~3 ms |
| CSS transition to 90% brightness | 88 ms | 0.6 ms |

**The capture block.** cpal takes the driver's own block size unless asked, and
this box picks 64 ms — two whole 1/30 s windows landing at once, then nothing
for 64 ms. Measured against the real device with a standalone cpal probe:

```
range   Range { min: 64, max: 1048576 }
default      21 blocks in 1.50s -> 64.0 ms per block,  14 blocks/s
Fixed(160)  153 blocks in 1.50s -> 10.0 ms per block, 102 blocks/s
```

`BLOCKS_PER_SEC = 100` in `live.rs` asks for 10 ms. A `Fixed` buffer size is a
request, not a setting — devices may refuse it at stream-build time — so it
falls back to the driver's own and says which it got. CoreAudio already hands
the M2 10.7 ms, so that machine barely moves. **Still to confirm on a real run:
the `buffer` line in the app's own output.**

**The CSS transition.** A transition eases in both directions or neither, so
the easing that makes the light die away instead of strobe was also easing the
rise, where it is indistinguishable from lag. `pill.js` now switches `--ease`
per direction — 0 ms up, `--fall` down — and only on frames where direction
flips, since an inherited custom property invalidates style below it. A/B on
the same page, six runs each: 89.8-90.2 ms against 0.4-0.6 ms.

What is left is the 1/30 s window itself. Halving it means re-measuring
`RELEASE` and `FLOOR_RISE`, which are tuned per frame at 30 Hz, for maybe
15 ms. Not worth changing one calibrated variable to chase.

**The dBFS calibration carried over untouched.** It is measured against a real
microphone and none of this changed what a level means.

### What went wrong while doing it

A build left **13 of oneDNN's 427 objects at 0 bytes**, and `ld.lld` skips an
empty archive member with only a warning, so the link failed with pages of
undefined `dnnl_stream_create` / `jit_avx512_*` symbols that looked like a
missing system dependency and were not. Cargo fingerprints inputs and never
checksums outputs, so it reported `Fresh onednn-src` and handed the linker the
same broken archive every time; `cargo check` passed throughout, because check
never links.

Two causes, neither provable after the fact: the disk was at 98%, and the
builds had been wrapped in a 570 s `timeout` against a 9 m 00 s cold build.
Both are now avoided — see CLAUDE.md, which carries the one-line check.
