# Kotha — plan and live state

Read `CLAUDE.md` first for the durable rules. The same plan with the reasoning
laid out visually is at
<https://claude.ai/code/artifact/3107084e-3472-4798-918f-e47e3410785b>. This file is the working state:
update it as phases complete. Last touched 2026-08-31.

**Status: Phases 0, 1 and 2 pass on the Arch/Ryzen machine.** The loop runs
end to end — microphone to VAD to model to clipboard — with the model held warm
between utterances, verified against live speech. One known defect carried
forward: Whisper repetition loops on roughly a quarter of live utterances, with
the fix deferred by decision (Phase 2, *repetition loops*).

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
- [ ] Port to Rust (~200 lines)

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

**The spelling corrector.** It is a pure `String → String` at the end of this
pipeline. It plugs in where `emit()` prints, once ported to Rust — and the port
is not the transliteration it looks like: `build_dictionary()` reads `wordfreq`
and the training manifest at *runtime*, and neither exists on a user's machine.
The port therefore needs a baked dictionary artifact (word → zipf, ~50 k
entries) shipped with the app. That is a real design decision, not a
translation, and it is the first thing to settle when the port starts.

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

### Phase 3 — Text at the cursor  ⟵ IN PROGRESS

The OS-specific part. Three mechanisms underneath.

- [x] **Copy-only mode that needs no permissions**, so the app is useful before
      anything is granted — and it is the *default*, because firing synthetic
      keystrokes into whatever window is focused should be asked for
- [x] Restore the previous clipboard contents afterwards — in paste mode only;
      in copy-only mode the clipboard *is* the delivery, so restoring it would
      throw the transcript away
- [x] Linux: X11 vs Wayland split — investigated, and it does not land where
      the plan assumed. See below
- [ ] macOS: Accessibility permission, synthetic ⌘V — the code path exists
      (`Key::Meta`), untested, needs the M2
- [ ] Windows: SendInput — enigo's own path, untested
- [ ] Verify it in a real editor

**Accept when:** dictated text appears in TextEdit, a browser field, and a
terminal, without the focused window changing. **Not yet met** — see the
Wayland finding.

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

The remaining route for KDE is libei via the XDG RemoteDesktop portal, which
KWin 6 does support and enigo has behind `libei_smol`/`libei_tokio`. It costs a
permission dialog on first use, which is Phase 5's walkthrough arriving early.
**Not attempted yet** — it puts a portal prompt on the desktop and that is
Kayes's call to make.

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
| Strict/tolerant gap that is spelling | **93.6%** (545 of 582 tokens) | 2026-08-31 |
| Misspellings reachable at SymSpell ED≤2 | 84.8% (462 of 545) | 2026-08-31 |
| Misspellings that are real English words | 26.4% — invisible to a dictionary | 2026-08-31 |
| English tokens dropped entirely | 11.9% (510 of 4299) | 2026-08-31 |
| Corrector latency per sentence | — | |
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
| Misspellings left standing | 309 of 4,299 English tokens | 2026-08-31 |
| ...of those, protected by the floor | **90.0%** — real words, need context | 2026-08-31 |
| ...reachable by a phonetic fallback | **17 tokens, ~+0.2 F1** — not built | 2026-08-31 |
