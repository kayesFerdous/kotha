#!/usr/bin/env python3
"""Phase 1's diagnostic: what is the 13.6-point English-F1 gap actually made of?

The paper reports two English-token scores on the 393-utterance test set:

    strict    P 75.3  R 69.4  F1 72.2   -- Latin script AND spelled right
    tolerant  P 89.4  R 82.4  F1 85.8   -- heard it, any script, near-miss ok

The 13.6-point gap is not all ours. It mixes two failures that need different
fixes, and only the first is a spelling corrector's job:

    the model wrote the English word in Latin, but misspelled it
    the model wrote the English word in Bangla script

This script splits the gap by walking the word alignment and classifying every
Latin REFERENCE token by what the model actually put in its place.

Scoring is bnasr_eval.py from the paper project, imported READ-ONLY, so that
"Latin token" and "tolerant match" mean exactly what they mean in the paper.
Nothing under ~/Documents/ASR is written.

Hypotheses come from the Rust spike -- the engine the corrector will actually
run behind -- decoded greedy, int8, suppress_tokens=[]. The paper's numbers are
fp16 with faster-whisper's defaults, so the absolute scores here will not match
it exactly. The SPLIT is the point, not the absolute level.

    ~/Documents/ASR/bangla-asr-test/.venv/bin/python gap.py [--examples N]

NOT A TUNING RUN. The per-word lists below are for understanding where the
errors are. The corrector's dictionary must come from en_50k plus the
TRAINING-side corpus vocabulary. Reading a misspelling off this output and
adding it to the dictionary would be fitting the held-out test set.
"""
import csv
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ASR = Path.home() / "Documents/ASR"
sys.path.insert(0, str(ASR / "notebooks/eval/files"))
import bnasr_eval as B  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

HERE = Path(__file__).parent
MANIFEST = ASR / "bangla-asr-test/test_manifest.csv"
LEXICON = ASR / "bangla-asr-test/listen_v13/lexicon.csv"
CHUNKS = ASR / "bangla-asr-test/chunks/test"
# The workspace shares one target directory at the repo root, so
# CTranslate2 is compiled once rather than once per crate.
SPIKE = HERE.parent / "target/release/kotha-spike"
MODEL = HERE.parent / "models/whisper-medium-bn-en-cs-faster"
DECODE = HERE / "decode_393.txt"
THREADS = "6"

FUZZY = 72.0  # score_utterance's default; the tolerant threshold the paper uses
MAX_ED_REACH = 2  # SymSpell's index depth in correct.py; anything past it needs a fallback

# What happened to a Latin reference token. Ordered worst-understood last.
CATS = [
    ("exact",      "written in Latin, spelled right          "),
    ("latin_near", "written in Latin, MISSPELLED             "),
    ("latin_far",  "written in Latin, unrecognisably wrong   "),
    ("bangla",     "written in BANGLA script, recognisable   "),
    ("bangla_far", "written in Bangla, unrecognisable        "),
    ("deleted",    "dropped entirely                         "),
    ("other",      "replaced by a digit / other              "),
]
# The corrector's target is latin_near. latin_far is arguably also its target
# but the model may simply have misheard. bangla* is the model's problem, not
# the corrector's -- and the corrector may not touch those.
CORRECTABLE = {"latin_near"}


def hypotheses():
    """Spike output for all 393, decoded once and cached in decode_393.txt.

    Resumable. The decode is ~50 minutes and a power cut during the first
    attempt cost 30 utterances, so this appends only what is missing rather
    than starting over. A cut also leaves a tail of NUL bytes where the page
    cache never flushed, hence the rstrip.
    """
    def parse():
        if not DECODE.exists():
            return {}
        text = DECODE.read_bytes().rstrip(b"\x00").decode("utf-8", errors="replace")
        got, cur = {}, None
        for line in text.splitlines():
            m = re.match(r"── (\S+\.wav)", line)
            if m:
                cur = m.group(1)
            elif cur:
                # An empty transcript is a real result, not a parse gap.
                got[cur], cur = line.strip(), None
        return got

    wavs = sorted(CHUNKS.glob("*.wav"))
    got = parse()
    todo = [w for w in wavs if w.name not in got]
    if todo:
        print(f"have {len(got)}/{len(wavs)}; decoding the missing {len(todo)} "
              f"(~{len(todo) * 8 // 60 + 1} min) ...")
        proc = subprocess.run(
            [str(SPIKE), str(MODEL), *map(str, todo)],
            capture_output=True, text=True,
            env={**os.environ, "KOTHA_THREADS": THREADS},
        )
        if proc.returncode != 0:
            sys.exit(f"spike failed:\n{proc.stderr[-2000:]}")
        with DECODE.open("r+b") as fh:          # drop any NUL tail, then append
            fh.truncate(len(DECODE.read_bytes().rstrip(b"\x00")))
        with DECODE.open("a", encoding="utf-8") as fh:
            fh.write(proc.stdout)
        got = parse()
    return got


def classify(ref_tok, hyp_tok, unifier):
    """One Latin reference token: what did the model put there?"""
    if hyp_tok is None:
        return "deleted"
    if hyp_tok == ref_tok:
        return "exact"
    script = B.script_of(hyp_tok)
    near = unifier.equivalent(ref_tok, hyp_tok, FUZZY)
    if script == "latin":
        return "latin_near" if near else "latin_far"
    if script in ("bangla", "mixed"):
        return "bangla" if near else "bangla_far"
    return "other"


def prf(tp, n_ref, n_hyp):
    p = tp / n_hyp if n_hyp else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * p, 100 * r, 100 * f


def selftest():
    """The classifier and the alignment map, on hand-made cases.

    Runs without the model or the test set:
        ~/Documents/ASR/bangla-asr-test/.venv/bin/python gap.py --selftest
    """
    u = B.ScriptUnifier({"ফাংশন": "function"})
    cases = [
        ("function", "function",  "exact"),
        ("function", "funcshun",  "latin_near"),   # Latin, misspelt, recoverable
        ("function", "elephant",  "latin_far"),    # Latin, but a different word
        ("function", "ফাংশন",     "bangla"),       # right word, wrong script
        ("function", "ভাত",       "bangla_far"),   # wrong word AND wrong script
        ("function", None,        "deleted"),
        ("function", "42",        "other"),
    ]
    for ref, hyp, want in cases:
        got = classify(ref, hyp, u)
        assert got == want, f"classify({ref!r},{hyp!r}) = {got!r}, want {want!r}"

    # The alignment map must pair a substituted token with the token that
    # replaced it, not with the position it happens to sit at. A deletion
    # earlier in the utterance shifts every later hypothesis index.
    ref_toks = B.tokenize(B.normalize_text("আমি একটা meeting এ যাব"))
    hyp_toks = B.tokenize(B.normalize_text("একটা meting এ যাব"))
    _, alignment = B._word_alignment(ref_toks, hyp_toks)
    hyp_for_ref = {}
    for c in alignment:
        if c.type in ("equal", "substitute"):
            for k in range(c.ref_end_idx - c.ref_start_idx):
                hyp_for_ref[c.ref_start_idx + k] = c.hyp_start_idx + k
    i = ref_toks.index("meeting")
    assert hyp_toks[hyp_for_ref[i]] == "meting", hyp_toks[hyp_for_ref.get(i, 0)]
    assert classify("meeting", "meting", u) == "latin_near"

    # normalize_text must not hide a script boundary: "meeting-এ" is two
    # tokens, and the Latin half has to stay classifiable as Latin.
    toks = B.tokenize(B.normalize_text("meeting-এ"))
    assert toks == ["meeting", "এ"], toks
    assert B.script_of(toks[0]) == "latin"

    print("selftest ok")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    corrected = "--corrected" in sys.argv
    n_examples = 25
    if "--examples" in sys.argv:
        n_examples = int(sys.argv[sys.argv.index("--examples") + 1])

    refs = {Path(r["audio_path"]).name: r["text"]
            for r in csv.DictReader(open(MANIFEST))}
    hyps = hypotheses()
    missing = [u for u in refs if u not in hyps]
    if missing:
        # Score what we have rather than blocking, but say so loudly: a
        # partial decode is a different denominator, not a smaller version
        # of the same number.
        print(f"WARNING: {len(missing)} of {len(refs)} utterances have no "
              f"hypothesis; scoring the other {len(refs) - len(missing)}.")
        print(f"         first few: {missing[:5]}")
        refs = {u: t for u, t in refs.items() if u in hyps}

    n_touched_non_latin = 0
    if corrected:
        # THE EVALUATION. Phase 1's rule is to run this once on held-out data.
        # correct.py's thresholds were set on the training split and nothing
        # below feeds back into them.
        import correct as C
        freq, _, _ = C.build_dictionary()
        corr = C.Corrector(freq, C.build_index(freq))
        print(f"corrector  floor {corr.floor}  margin {corr.margin}  "
              f"dictionary {len(freq)}")
        for u, h in hyps.items():
            toks = B.tokenize(B.normalize_text(h))
            out = [corr.correct_token(x) for x in toks]
            # Bengali must be unmodifiable by construction.
            # Verified here rather than trusted.
            n_touched_non_latin += sum(
                1 for a, b in zip(toks, out)
                if a != b and B.script_of(a) != "latin")
            hyps[u] = " ".join(out)
        print(f"non-Latin tokens modified: {n_touched_non_latin}  "
              f"{'← MUST be 0' if n_touched_non_latin else '(the guarantee holds)'}")
        assert n_touched_non_latin == 0, "corrector touched non-Latin text"

    lex = B.load_lexicon_csv(str(LEXICON))
    unifier = B.ScriptUnifier(lex)
    print(f"lexicon    {len(lex)} bangla→latin entries")
    print(f"romanizer  {'available' if unifier.romanizer_available else 'MISSING — tolerant matching will under-count'}")
    print(f"utterances {len(refs)}\n")

    counts = Counter()
    pairs = Counter()          # (ref, hyp) for the correctable category
    bangla_pairs = Counter()
    dists = Counter()          # edit distance of the misspellings
    # Bag-of-words strict counts, exactly as score_utterance does them, so the
    # alignment view below can be checked against the paper's own metric.
    bag_tp = bag_ref = bag_hyp = 0

    for utt, reference in refs.items():
        ref_toks = B.tokenize(B.normalize_text(reference))
        hyp_toks = B.tokenize(B.normalize_text(hyps[utt]))

        ref_eng = [t for t in ref_toks if B.script_of(t) == "latin"]
        hyp_eng = [t for t in hyp_toks if B.script_of(t) == "latin"]
        bag_ref += len(ref_eng)
        bag_hyp += len(hyp_eng)
        bag_tp += sum((Counter(ref_eng) & Counter(hyp_eng)).values())

        _, alignment = B._word_alignment(ref_toks, hyp_toks)
        hyp_for_ref = {}
        for c in alignment:
            if c.type in ("equal", "substitute"):
                for k in range(c.ref_end_idx - c.ref_start_idx):
                    hyp_for_ref[c.ref_start_idx + k] = c.hyp_start_idx + k

        for i, t in enumerate(ref_toks):
            if B.script_of(t) != "latin":
                continue
            j = hyp_for_ref.get(i)
            h = hyp_toks[j] if j is not None and j < len(hyp_toks) else None
            cat = classify(t, h, unifier)
            counts[cat] += 1
            if cat == "latin_near":
                pairs[(t, h)] += 1
                dists[Levenshtein.distance(t, h)] += 1
            elif cat == "bangla":
                bangla_pairs[(t, h)] += 1

    total = sum(counts.values())
    print(f"{'':43s}  count   share")
    for key, label in CATS:
        n = counts[key]
        print(f"{label}  {n:5d}  {100 * n / total:5.1f}%")
    print(f"{'':43s}  {total:5d}")

    # ---- what the gap is made of ----------------------------------------
    strict_hit = counts["exact"]
    tolerant_hit = strict_hit + counts["latin_near"] + counts["bangla"]
    gap = tolerant_hit - strict_hit
    print(f"\nrecall, aligned view")
    print(f"  strict    {100 * strict_hit / total:5.2f}%   ({strict_hit}/{total})")
    print(f"  tolerant  {100 * tolerant_hit / total:5.2f}%   ({tolerant_hit}/{total})")
    print(f"  gap       {100 * gap / total:5.2f} points  ({gap} tokens)")
    if gap:
        print(f"\nTHE SPLIT — of those {gap} tokens:")
        print(f"  misspelled in Latin  {counts['latin_near']:4d}   "
              f"{100 * counts['latin_near'] / gap:5.1f}%  ← the corrector's")
        print(f"  written in Bangla    {counts['bangla']:4d}   "
              f"{100 * counts['bangla'] / gap:5.1f}%  ← not the corrector's")

    # ---- what fixing them would be worth --------------------------------
    # A misspelled Latin token costs twice in strict scoring: it is a missed
    # reference token AND a spurious hypothesis token. Correcting it moves
    # both, which is why the ceiling is larger than the recall gap alone.
    print(f"\nstrict English-token score, bag-of-words (the paper's metric)")
    p, r, f = prf(bag_tp, bag_ref, bag_hyp)
    print(f"  now                          P {p:5.2f}  R {r:5.2f}  F1 {f:5.2f}")
    for label, fixed in (("+ every latin_near fixed     ", counts["latin_near"]),
                         ("+ latin_near and latin_far   ", counts["latin_near"] + counts["latin_far"])):
        p, r, f = prf(bag_tp + fixed, bag_ref, bag_hyp)
        print(f"  {label}  P {p:5.2f}  R {r:5.2f}  F1 {f:5.2f}   (ceiling)")
    print("  ceiling assumes perfect correction and no damage to tokens that")
    print("  were already right — an upper bound, not a forecast.")

    # ---- can SymSpell at edit distance 2 even reach them? ---------------
    if dists:
        print(f"\nedit distance, hypothesis → reference, for the {sum(dists.values())} misspellings")
        cum = 0
        for d in sorted(dists):
            cum += dists[d]
            print(f"  {d}  {dists[d]:4d}   cumulative {100 * cum / sum(dists.values()):5.1f}%")
        reach = sum(n for d, n in dists.items() if d <= 2)
        print(f"  SymSpell at ED≤2 can reach {reach}/{sum(dists.values())} "
              f"({100 * reach / sum(dists.values()):.1f}%) of them at all.")

    # ---- how many misspellings are themselves real English words? ------
    # A dictionary corrector only fires on tokens it does not recognise. A
    # misspelling that IS a word ("mill" for "meal") is invisible to it and
    # needs context. This number decides whether bigram context is Phase 1
    # work or Phase 7 work.
    try:
        from wordfreq import top_n_list
    except ImportError:
        print("\nwordfreq not installed — skipping the real-word check")
    else:
        vocab = set(top_n_list("en", 50000))
        real = {pr: n for pr, n in pairs.items() if pr[1] in vocab}
        n_real = sum(real.values())
        n_all = sum(pairs.values())
        print(f"\nof the {n_all} misspellings, how many are real English words?")
        print(f"  invisible to a dictionary (hyp IS a word)  {n_real:4d}  "
              f"{100 * n_real / n_all:5.1f}%   needs context")
        print(f"  flaggable (hyp is not a word)              {n_all - n_real:4d}  "
              f"{100 * (n_all - n_real) / n_all:5.1f}%   ← SymSpell's reach")
        flag_ed2 = sum(n for (r, h), n in pairs.items()
                       if h not in vocab and Levenshtein.distance(r, h) <= 2)
        print(f"  flaggable AND within ED≤2                  {flag_ed2:4d}  "
              f"{100 * flag_ed2 / n_all:5.1f}%   ← the realistic target")
        p_, r_, f_ = prf(bag_tp + flag_ed2, bag_ref, bag_hyp)
        print(f"  strict F1 if all of those are fixed: {f_:.2f}  "
              f"(from {prf(bag_tp, bag_ref, bag_hyp)[2]:.2f})")
        print(f"\n  top real-word errors a dictionary cannot see:")
        for (ref_t, hyp_t), n in sorted(real.items(), key=lambda kv: -kv[1])[:12]:
            print(f"    {n:3d}  {hyp_t!r:24s} → {ref_t!r}")

    # ---- what is even left for a fallback to reach? ---------------------
    # Only a token BELOW the corrector's floor can ever be touched; above it
    # the token is protected on purpose, and no fallback changes that. So this
    # bounds a phonetic fallback's yield without implementing one.
    if corrected:
        n_all = sum(pairs.values())
        touchable = sum(n for (_, h), n in pairs.items()
                        if freq.get(h, 0.0) < corr.floor)
        far = sum(n for (r, h), n in pairs.items()
                  if freq.get(h, 0.0) < corr.floor
                  and Levenshtein.distance(r, h) > MAX_ED_REACH)
        print(f"\nof the {n_all} misspellings still standing after correction:")
        print(f"  above the floor, protected by design    {n_all - touchable:4d}  "
              f"{100 * (n_all - touchable) / n_all:5.1f}%")
        print(f"  below the floor, a fallback could act   {touchable:4d}  "
              f"{100 * touchable / n_all:5.1f}%   ← ceiling on ANY fallback")
        print(f"  below the floor AND beyond ED {MAX_ED_REACH}          {far:4d}  "
              f"{100 * far / n_all:5.1f}%   ← Double Metaphone's actual target")

    print(f"\ntop {n_examples} misspellings — hypothesis → reference  (understanding only,")
    print("NOT dictionary input; these are held-out test utterances)")
    for (ref_t, hyp_t), n in pairs.most_common(n_examples):
        print(f"  {n:3d}  {hyp_t!r:28s} → {ref_t!r}")

    print(f"\ntop {n_examples} written in Bangla instead — not the corrector's to fix")
    for (ref_t, hyp_t), n in bangla_pairs.most_common(n_examples):
        print(f"  {n:3d}  {hyp_t!r:28s} → {ref_t!r}")


if __name__ == "__main__":
    main()
