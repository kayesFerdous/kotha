#!/usr/bin/env python3
"""Phase 1: the spelling corrector. Python prototype, to be ported to Rust.

The model hears English well and writes it in Latin script, but misspells it.
gap.py measured the damage on the 393-utterance test set: of the 13.54-point
strict/tolerant English-F1 gap, 93.6% is misspelling and only 6.4% is Bengali
script. This fixes the first kind.

WHAT IT WILL AND WILL NOT TOUCH
-------------------------------
Only tokens that bnasr_eval.script_of() calls "latin". Bengali output is
unmodifiable by construction, which is what makes it safe to correct English
aggressively.

Correct-or-abstain, never guess. A confidently wrong real word is worse than a
visible misspelling: the user can fix what they can see.

WHY THE GATE IS A FREQUENCY MARGIN, NOT "IS IT A WORD"
-----------------------------------------------------
The obvious design corrects only tokens missing from the dictionary. It fails
twice here:

  * 26.4% of the model's misspellings ARE real English words -- "mill" for
    "meal", "throw" for "through". Nothing flags those.
  * The dictionary itself contains common misspellings. wordfreq's top 50k
    has "grammer" in it, so an is-it-a-word gate would never fix "grammer".

So instead, two gates, both set by calibrate() on training-side data only:

  KNOWN_FLOOR  A token that is itself solidly attested is never touched, at
               any margin. Without this the corrector eats ordinary English:
               then -> the, hand -> and, band -> and, form -> for. Those are
               all large frequency jumps between two real words, so a margin
               alone waves them through -- damage bottomed out at 1% of every
               English token in the training references, which is far too
               much to do to text that was already correct.

  MARGIN       Below the floor, take the best candidate within edit distance 2
               only if it is this many orders of magnitude more common than
               what the model wrote.

WHAT THE CALIBRATED FLOOR GIVES UP
----------------------------------
calibrate() put the floor at 2.5, and "grammer" sits at zipf 2.9. So this
corrector does NOT fix grammer -> grammar. That is a deliberate concession,
not an oversight: at floor 3.0 the corrector starts rewriting real words
(unhappiness -> happiness, slab -> lab), and a confidently
wrong real word is worse than a visible misspelling.

Unigram frequency cannot separate "grammer" (a misspelling at 2.9) from
"neutrally" (a real word at ~3.0). Nothing at this floor can. That separation
needs the surrounding words -- bigram context, still unbuilt --
and this is the measurement that says when to build it.

"mill" (4.38) is protected twice over: by the floor, and by having no margin
on "meal" (4.48). That is the case it would be most embarrassing to get
wrong.

DICTIONARY
----------
wordfreq's top 50k English (the OpenSubtitles-blend register we wanted,
already installed, no download) plus the English side of the TRAINING corpus,
which supplies the domain words a general list lacks: biryani, shawarma, vlog.

The test-side lexicon at bangla-asr-test/listen_v13/lexicon.csv is deliberately
NOT used. gap.py may read it because scoring is measurement; feeding it to the
corrector would be fitting the held-out set.

    ~/Documents/ASR/bangla-asr-test/.venv/bin/python correct.py --selftest
    ~/Documents/ASR/bangla-asr-test/.venv/bin/python correct.py --calibrate
"""
import csv
import math
import sys
from collections import Counter
from pathlib import Path

ASR = Path.home() / "Documents/ASR"
sys.path.insert(0, str(ASR / "notebooks/eval/files"))
import bnasr_eval as B  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

TRAIN = ASR / "bangla-asr/train_manifest.csv"

MAX_ED = 2          # SymSpell index depth; gap.py measured ED≤2 as 84.8% reach
MIN_CORPUS_COUNT = 2  # a Latin token seen once in training is probably noise
# Both set by calibrate() on the training split, 2026-08-31. Floor 2.5 is the
# knee: at 2.5 and below, zero real English words are rewritten and all 585
# tail fixes survive; at 3.0 damage appears (unhappiness -> happiness, slab ->
# lab). Margin is inert at this floor -- anything under zipf 2.5 is beaten by
# more than 2.0 by whatever candidate wins -- but it stays as a guard for when
# the dictionary changes.
MARGIN = 1.0
KNOWN_FLOOR = 2.5

# Short tokens are reachable from everything at ED 2, so the budget scales
# with length. "hal" must not become "had", "has", "hat", "ham" or "hall" on a
# coin flip.
def ed_budget(token):
    n = len(token)
    return 0 if n <= 3 else (1 if n <= 5 else MAX_ED)


def build_dictionary():
    """zipf frequency per word, from wordfreq plus the training corpus."""
    from wordfreq import top_n_list, zipf_frequency

    freq = {w: zipf_frequency(w, "en") for w in top_n_list("en", 50000)}

    corpus = Counter()
    for row in csv.DictReader(open(TRAIN)):
        for t in B.tokenize(B.normalize_text(row["text"])):
            if B.script_of(t) == "latin":
                corpus[t] += 1
    total = sum(corpus.values())
    for w, n in corpus.items():
        if n < MIN_CORPUS_COUNT:
            continue
        z = math.log10(n / total * 1e9)
        if z > freq.get(w, 0.0):
            freq[w] = z
    return freq, corpus, total


def build_index(freq):
    """SymSpell's delete index: delete-variant -> words that produce it.

    This is the structure the Rust `symspell` crate builds too, so the port is
    a transliteration rather than a redesign.
    """
    index = {}
    for w in freq:
        for d in deletes(w, ed_budget(w)):
            index.setdefault(d, []).append(w)
    return index


def deletes(word, budget):
    """Every string reachable from `word` by deleting up to `budget` chars."""
    out = {word}
    frontier = {word}
    for _ in range(budget):
        nxt = set()
        for s in frontier:
            for i in range(len(s)):
                nxt.add(s[:i] + s[i + 1:])
        out |= nxt
        frontier = nxt
    return out


class Corrector:
    def __init__(self, freq, index, margin=MARGIN, floor=KNOWN_FLOOR):
        self.freq, self.index = freq, index
        self.margin, self.floor = margin, floor

    def candidates(self, token):
        budget = ed_budget(token)
        seen = set()
        for d in deletes(token, budget):
            for w in self.index.get(d, ()):
                if w not in seen and Levenshtein.distance(token, w) <= budget:
                    seen.add(w)
        return seen

    def correct_token(self, token):
        """Return the token unchanged, or a strictly better-supported word."""
        if B.script_of(token) != "latin":
            return token          # the guarantee: Bengali is never touched
        here = self.freq.get(token, 0.0)
        if here >= self.floor:
            return token          # solidly attested; not ours to second-guess
        best, best_key = token, None
        for w in self.candidates(token):
            if w == token:
                continue
            key = (-Levenshtein.distance(token, w), self.freq[w])
            if best_key is None or key > best_key:
                best, best_key = w, key
        if best_key is None:
            return token
        if self.freq[best] - here < self.margin:
            return token          # correct-or-abstain
        return best

    def correct(self, text):
        return " ".join(self.correct_token(t) for t in B.tokenize(text))


# --------------------------------------------------------------------------
# Calibration -- training-side only. The 393 test utterances are held out.
# --------------------------------------------------------------------------

def calibrate(freq, index, floors=(1.5, 2.0, 2.5, 3.0, 3.5), margins=(0.5, 1.0, 2.0)):
    """Pick the two thresholds using training-side data only.

    Training references are LLM-normalised labels, and the first run of this
    showed they are not perfectly clean: the corrector's top "errors" were
    algorithom -> algorithm, truncess -> princess, behand -> behind, which are
    fixes, not damage. So a raw change count is useless as a safety number --
    it mixes true fixes with false ones.

    The split that does work: was the token the model wrote ALREADY a real
    English word? Rewriting "neutrally" or "sunbeams" is damage. Rewriting
    "algorithom" is not, whatever it becomes. So:

        risky   token is in wordfreq's top 50k -- a real word got rewritten
        benign  token is not a word -- it was misspelled either way

    `risky` is the number to minimise. It is still an over-count (some
    top-50k entries are themselves common misspellings) which is the safe
    direction for a threshold.

    NOTE ON DIRECTION: a LOWER floor protects MORE tokens, because the floor
    is the bar a token must clear to be left alone. Lower is safer.

    Recall cannot be measured here -- clean text has nothing to recover -- so
    the choice is the safest setting that still fixes the rare tail, not the
    one that maximises a score. Recall comes from the held-out evaluation,
    once.
    """
    from wordfreq import top_n_list
    real_words = set(top_n_list("en", 50000))

    toks = []
    for row in csv.DictReader(open(TRAIN)):
        toks += [t for t in B.tokenize(B.normalize_text(row["text"]))
                 if B.script_of(t) == "latin"]
    uniq = Counter(toks)
    print(f"training English tokens: {len(toks)} ({len(uniq)} types)\n")
    print(f"{'floor':>6} {'margin':>7} {'risky':>7} {'rate':>8} {'benign':>7}   "
          f"examples of risky rewrites")
    for fl in floors:
        for m in margins:
            c = Corrector(freq, index, margin=m, floor=fl)
            risky, benign = Counter(), 0
            for t, n in uniq.items():
                out = c.correct_token(t)
                if out == t:
                    continue
                if t in real_words:
                    risky[(t, out)] += n
                else:
                    benign += n
            n_risky = sum(risky.values())
            ex = ", ".join(f"{a}->{b}" for (a, b), _ in risky.most_common(3))
            print(f"{fl:6.1f} {m:7.2f} {n_risky:7d} {100 * n_risky / len(toks):7.3f}% "
                  f"{benign:7d}   {ex}")
    print("\nrisky  = rewrote a real English word. This is the damage number.")
    print("benign = rewrote something that was not a word either way.")
    print("Lower floor protects more. Take the setting where risky is ~0 and")
    print("benign is still large.")


def selftest():
    """The guarantees, on cases that need no corpus."""
    freq = {"grammar": 4.6, "grammer": 2.9, "meal": 4.48, "mill": 4.38,
            "character": 4.9, "carector": 0.0, "football": 4.5, "footbal": 1.6,
            "hall": 4.3, "had": 5.5}
    c = Corrector(freq, build_index(freq), margin=1.0)

    # The calibrated floor is 2.5 and "grammer" sits at 2.9, so it is left
    # alone. This asserts the CONCESSION, so that raising the floor without
    # re-running calibrate() fails loudly here.
    assert Corrector(freq, build_index(freq), margin=1.0,
                     floor=KNOWN_FLOOR).correct_token("grammer") == "grammer"
    # Below the floor it is fixable, which is what the floor is trading away.
    assert Corrector(freq, build_index(freq), margin=1.0,
                     floor=3.5).correct_token("grammer") == "grammar"
    # Fixes an out-of-vocabulary misspelling.
    assert c.correct_token("footbal") == "football"
    # The known ceiling: "carector" -> "character" is edit distance 3, out of
    # SymSpell's reach. gap.py measured 15% of misspellings in this bucket.
    # It must ABSTAIN, not reach for something closer and wrong. Double
    # Metaphone is what would pick these up.
    assert c.correct_token("carector") == "carector"
    # Abstains when the two words are equally common: no margin, no guess.
    assert c.correct_token("mill") == "mill"
    # The floor protects ordinary English from large frequency jumps between
    # two perfectly real words. Without it this becomes "the".
    assert Corrector({"then": 5.6, "the": 7.2}, build_index({"then": 5.6, "the": 7.2}),
                     margin=1.0, floor=3.5).correct_token("then") == "then"
    # Leaves a correct word alone.
    assert c.correct_token("meal") == "meal"
    # Short tokens get no edit budget, so "hal" cannot become "had"/"hall".
    assert c.correct_token("hal") == "hal"
    # THE non-negotiable: Bengali is unmodifiable by construction.
    for t in ["আমি", "ফাংশন", "meeting-এ".split("-")[1]]:
        assert c.correct_token(t) == t, t
    # Mixed-script tokens are not "latin" either, so they are left alone.
    assert B.script_of("student এর".split()[0]) == "latin"
    assert c.correct_token("গ্রামার") == "গ্রামার"
    # A whole sentence: only the English moves, and the Bengali survives
    # byte-for-byte alongside it.
    out = c.correct("আমার footbal টা weak")
    assert out == "আমার football টা weak", out

    print("selftest ok")


def dump_dict(freq, path):
    """Bake the dictionary out for the Rust port.

    build_dictionary() reads wordfreq and the training manifest at *runtime*.
    Neither exists on a user's machine, so the shipped app cannot build its own
    dictionary -- it has to carry one. This writes that artifact: word, tab,
    zipf, one per line, sorted. ~700 KB, which is nothing next to a 775 MB
    model, and it is `include_str!`d straight into the binary so there is no
    path to resolve and no missing-file case to handle.

    Six decimal places is far more precision than the thresholds need (the
    floor is 2.5) and costs a few bytes.
    """
    with open(path, "w") as f:
        for w in sorted(freq):
            f.write(f"{w}\t{freq[w]:.6f}\n")
    print(f"wrote {path}: {len(freq)} words")


def main():
    if "--selftest" in sys.argv:
        return selftest()
    freq, corpus, total = build_dictionary()
    index = build_index(freq)
    print(f"dictionary {len(freq)} words   index {len(index)} delete keys")
    print(f"corpus     {len(corpus)} English types, {total} tokens "
          f"(training split only)\n")
    if "--calibrate" in sys.argv:
        return calibrate(freq, index)
    if "--dump-dict" in sys.argv:
        return dump_dict(freq, "dict.tsv")
    c = Corrector(freq, index)
    for line in sys.stdin:
        print(c.correct(B.normalize_text(line)))


if __name__ == "__main__":
    main()
