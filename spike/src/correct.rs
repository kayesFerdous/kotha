//! Phase 1, ported: the spelling corrector.
//!
//! A faithful port of `spike/correct.py`, which measured +5.21 strict
//! English-F1 (71.81 → 77.02) on the held-out 393 with zero non-Latin tokens
//! modified. The rules are not re-derived here and must not be re-tuned here —
//! the thresholds came from `correct.py --calibrate` on the *training* split,
//! and PLAN.md holds the reasoning. This file is a translation.
//!
//! Parity with the prototype is checked, not assumed:
//!
//! ```text
//! live --correct < normalised.txt        # this file
//! correct.py     < normalised.txt        # the prototype
//! ```
//!
//! Over the 393 held-out hypotheses the two agree on every line but one, and
//! that one is not a disagreement about spelling — it is an exact tie. See
//! `Corrector::words` for why this file is the deterministic half of the pair.
//!
//! # The dictionary is baked in
//!
//! `build_dictionary()` in the prototype reads `wordfreq` and the training
//! manifest at runtime. Neither exists on a user's machine, so the shipped app
//! carries the result instead: `dict.tsv`, word and zipf, regenerated with
//! `correct.py --dump-dict`. 860 KB compiled into the binary — nothing next to
//! a 775 MB model, and it removes both a file path to resolve and a
//! missing-file case to handle.
//!
//! # Why there is no delete index
//!
//! The prototype builds SymSpell's delete index: 1.2 M keys, and in Rust
//! roughly 150 MB of `HashMap` plus a second or so of startup. It is buying
//! speed we do not need. Only tokens *below the floor* are ever looked up —
//! one or two per sentence — and a length-filtered brute-force scan of 50 k
//! words costs single-digit milliseconds against a multi-second decode.
//!
//! That also makes the candidate set exactly "every word within the budget",
//! where the prototype's index is built with each *dictionary word's* own
//! budget and its lookup uses the *token's*, so it can miss a candidate when
//! the two differ. This scan is a superset. The parity run is what says
//! whether that difference ever fires on real output.
//!
// ponytail: linear scan over 50k words per corrected token. The delete index
// is the upgrade path if a sentence ever holds enough sub-floor tokens to
// matter; measure before building it, it costs ~150 MB.

use std::time::Instant;

/// Baked by `correct.py --dump-dict`. Sorted, so lookups binary-search it.
const DICT: &str = include_str!("../dict.tsv");

/// Below this zipf a token may be corrected; at or above it, never.
///
/// Calibrated on the training split (`correct.py --calibrate`), where 2.5 is
/// the knee: zero real English words rewritten, all 585 tail fixes surviving.
/// At 3.0 damage appears (`unhappiness` → `happiness`, `slab` → `lab`).
/// Raising this without re-calibrating is how the corrector starts eating
/// correct text.
const KNOWN_FLOOR: f32 = 2.5;

/// How much more common a candidate must be before it is taken. Inert at floor
/// 2.5; kept as a guard for when the dictionary changes.
const MARGIN: f32 = 1.0;

/// Edit budget by token length.
///
/// Short tokens are reachable from half the dictionary at distance 2, so the
/// budget scales with length: `hal` must not become `had`, `has`, `hat` or
/// `hall` on a coin flip.
fn ed_budget(n: usize) -> usize {
    match n {
        0..=3 => 0,
        4..=5 => 1,
        _ => 2,
    }
}

/// Is this a token the corrector is allowed to touch?
///
/// `bnasr_eval.script_of() == "latin"`: contains an ASCII letter and no
/// Bengali codepoint. Mixed-script tokens are *not* latin, so they are left
/// alone too. This is the non-negotiable from CLAUDE.md §5 — Bengali output is
/// unmodifiable by construction, and that is what makes it safe to correct
/// English aggressively.
pub fn is_latin(token: &str) -> bool {
    let mut has_latin = false;
    for c in token.chars() {
        if ('\u{0980}'..='\u{09FF}').contains(&c) {
            return false;
        }
        has_latin |= c.is_ascii_alphabetic();
    }
    has_latin
}

struct Entry {
    word: &'static str,
    zipf: f32,
    /// Character length, precomputed for the length filter.
    len: usize,
}

pub struct Corrector {
    /// Sorted by word, so exact lookup is a binary search and the candidate
    /// scan is deterministic.
    ///
    /// That last part is not incidental. The prototype iterates a Python
    /// *set*, so when two candidates tie on both edit distance and frequency
    /// it keeps whichever the set happened to yield first — which depends on
    /// the process's string-hash seed. Measured across six seeds on the 393:
    /// `yeas` becomes `years` on three of them and `year` on the other three,
    /// and it is the only token in the whole test set where they can differ.
    /// This scan walks the dictionary in order and takes the alphabetically
    /// first, every time.
    ///
    /// Neither answer is better. `year` and `years` are both edit distance 1
    /// from `yeas` and both sit at zipf 5.96; unigram frequency cannot
    /// separate them and nothing at this floor can. It is the same finding as
    /// `grammer` and `mill` — the next lever is context, not a wider net.
    words: Vec<Entry>,
    floor: f32,
    margin: f32,
}

impl Default for Corrector {
    fn default() -> Self {
        Self::new()
    }
}

impl Corrector {
    /// Parse the baked dictionary. ~50 k lines; costs a few milliseconds.
    pub fn new() -> Self {
        let t = Instant::now();
        let words: Vec<Entry> = DICT
            .lines()
            .filter_map(|line| {
                let (word, zipf) = line.split_once('\t')?;
                Some(Entry {
                    word,
                    zipf: zipf.parse().ok()?,
                    len: word.chars().count(),
                })
            })
            .collect();

        debug_assert!(
            words.windows(2).all(|w| w[0].word < w[1].word),
            "dict.tsv must be sorted — binary search depends on it"
        );
        eprintln!(
            "corrector {} words, {:.0} ms",
            words.len(),
            t.elapsed().as_secs_f64() * 1000.0
        );

        Self { words, floor: KNOWN_FLOOR, margin: MARGIN }
    }

    fn zipf(&self, token: &str) -> f32 {
        match self.words.binary_search_by(|e| e.word.cmp(token)) {
            Ok(i) => self.words[i].zipf,
            Err(_) => 0.0,
        }
    }

    /// Return the token unchanged, or a strictly better-supported word.
    ///
    /// Expects a bare lowercase token — no punctuation, no capitals. Use
    /// [`Corrector::correct_text`] on anything a user will see.
    pub fn correct_token<'a>(&self, token: &'a str) -> &'a str {
        if !is_latin(token) {
            return token; // the guarantee
        }
        let here = self.zipf(token);
        if here >= self.floor {
            return token; // solidly attested; not ours to second-guess
        }

        let n = token.chars().count();
        let budget = ed_budget(n);
        if budget == 0 {
            return token;
        }

        // (-distance, zipf), maximised — nearest first, then most common.
        let mut best: Option<(&Entry, usize)> = None;
        for e in &self.words {
            if e.len.abs_diff(n) > budget || e.word == token {
                continue;
            }
            let Some(d) = levenshtein_within(token, e.word, budget) else {
                continue;
            };
            let better = match best {
                None => true,
                Some((b, bd)) => d < bd || (d == bd && e.zipf > b.zipf),
            };
            if better {
                best = Some((e, d));
            }
        }

        match best {
            // Correct-or-abstain: a confidently wrong real word is worse than a
            // visible misspelling, because the user can fix what they can see.
            Some((e, _)) if e.zipf - here >= self.margin => e.word,
            _ => token,
        }
    }

    /// Correct the English in a line of dictated text, leaving everything else
    /// exactly as it was.
    ///
    /// The token-level rule cannot be applied to raw output directly: the
    /// model writes `you,` and `hello.`, and a bare lookup would "correct"
    /// `you,` to `you` and silently eat the comma. So each whitespace token is
    /// split into leading punctuation, a core, and trailing punctuation, and
    /// only the core is considered. Capitalisation is put back if the core
    /// changed.
    pub fn correct_text(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        for (i, tok) in text.split_whitespace().enumerate() {
            if i > 0 {
                out.push(' ');
            }
            let head = tok.len() - tok.trim_start_matches(|c: char| !c.is_alphanumeric()).len();
            let tail = tok.len() - tok.trim_end_matches(|c: char| !c.is_alphanumeric()).len();
            let core = &tok[head..tok.len() - tail];

            if core.is_empty() || !is_latin(core) {
                out.push_str(tok);
                continue;
            }
            let lower = core.to_lowercase();
            let fixed = self.correct_token(&lower);

            out.push_str(&tok[..head]);
            if fixed == lower {
                out.push_str(core); // unchanged: keep the user's own casing
            } else {
                out.push_str(&recase(core, fixed));
            }
            out.push_str(&tok[tok.len() - tail..]);
        }
        out
    }
}

/// Give `fixed` the capitalisation shape `original` had.
fn recase(original: &str, fixed: &str) -> String {
    let mut chars = original.chars();
    let first_upper = chars.next().is_some_and(char::is_uppercase);
    let rest_upper = chars.clone().count() > 0 && chars.all(char::is_uppercase);

    match (first_upper, rest_upper) {
        (true, true) => fixed.to_uppercase(),
        (true, false) => {
            let mut c = fixed.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        }
        _ => fixed.to_string(),
    }
}

/// Levenshtein distance, or `None` if it exceeds `max`.
///
/// Two-row DP with an early bail once every cell in a row is over budget.
/// Unit costs, matching `rapidfuzz.distance.Levenshtein` in the prototype.
fn levenshtein_within(a: &str, b: &str, max: usize) -> Option<usize> {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len().abs_diff(b.len()) > max {
        return None;
    }

    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];

    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        let mut row_min = cur[0];
        for (j, &cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(ca != cb);
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
            row_min = row_min.min(cur[j + 1]);
        }
        if row_min > max {
            return None;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    (prev[b.len()] <= max).then_some(prev[b.len()])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The prototype's selftest, on the real baked dictionary.
    ///
    /// These are the guarantees, not a sample of behaviour. Each one is a
    /// decision PLAN.md records the reasoning for; a failure here means the
    /// port has drifted from what was measured.
    #[test]
    fn guarantees() {
        let c = Corrector::new();

        // Fixes an out-of-vocabulary misspelling.
        assert_eq!(c.correct_token("footbal"), "football");
        // The live run's own miss: examp → exam.
        assert_eq!(c.correct_token("examp"), "exam");

        // THE non-negotiable: Bengali is unmodifiable by construction.
        for t in ["আমি", "ফাংশন", "গ্রামার", "এর"] {
            assert_eq!(c.correct_token(t), t);
        }
        // Mixed script is not "latin" either.
        assert_eq!(c.correct_token("student-এর"), "student-এর");

        // Leaves ordinary English alone. Without the floor these become
        // "the", "and", "for" — large frequency jumps between real words.
        for t in ["then", "hand", "band", "form", "meal", "meeting"] {
            assert_eq!(c.correct_token(t), t);
        }
        // "mill" is protected twice: by the floor, and by having no margin on
        // "meal". The case it would be most embarrassing to get wrong.
        assert_eq!(c.correct_token("mill"), "mill");

        // The concession the floor buys, asserted so that raising the floor
        // without re-calibrating fails loudly: "grammer" sits above it.
        assert!(c.zipf("grammer") > KNOWN_FLOOR);
        assert_eq!(c.correct_token("grammer"), "grammer");

        // Short tokens get no budget, so "hal" cannot become "had" or "hall",
        // and the three-character residue gap.py listed stays untouched.
        for t in ["hal", "gim", "ead", "psr"] {
            assert_eq!(c.correct_token(t), t);
        }

        // The ceiling: beyond ED 2 it abstains rather than reaching for
        // something wrong. `ambacerer` → `ambassador` is edit distance 5, and
        // is the case PLAN.md used to argue Double Metaphone's loose net was
        // not worth its damage.
        assert_eq!(c.correct_token("ambacerer"), "ambacerer");

        // Real fixes from the 393, at the two edit distances that matter.
        assert_eq!(c.correct_token("annother"), "another");
        assert_eq!(c.correct_token("colloberation"), "collaboration");

        // Ties are broken deterministically, alphabetically. `year` and
        // `years` are both ED 1 from `yeas` and both zipf 5.96; the prototype
        // picks between them by hash order. This is the one token in the 393
        // where the two implementations can disagree, and it is a coin flip,
        // not a defect.
        assert_eq!(c.correct_token("yeas"), "year");
        assert_eq!(c.zipf("year"), c.zipf("years"));

        // NOT a guarantee, recorded because it is easy to mistake for one:
        // against the real dictionary `carector` becomes `creator` (ED 2), not
        // the `character` it was meant to be. The prototype does exactly the
        // same — its selftest only abstained here because its toy dictionary
        // held no ED-2 neighbour. This is the real-word failure the floor
        // cannot see, and it is inside the measured +5.21.
        assert_eq!(c.correct_token("carector"), "creator");
    }

    #[test]
    fn whole_lines_keep_everything_but_the_misspelling() {
        let c = Corrector::new();

        // Only the English moves; the Bengali survives byte for byte.
        assert_eq!(c.correct_text("আমার footbal টা weak"), "আমার football টা weak");
        // Punctuation is not eaten — a bare token lookup would drop the comma
        // and the danda.
        assert_eq!(
            c.correct_text("hello footbal, আমি examp দিব।"),
            "hello football, আমি exam দিব।"
        );
        // Capitalisation is put back on a corrected word, and left alone on an
        // untouched one.
        assert_eq!(c.correct_text("Footbal FOOTBAL Meeting"), "Football FOOTBALL Meeting");
        // Runs of whitespace collapse to single spaces; that is the only
        // change made to text with no English in it.
        assert_eq!(c.correct_text("আমি ভালো আছি।"), "আমি ভালো আছি।");
    }

    #[test]
    fn distance_respects_the_cutoff() {
        assert_eq!(levenshtein_within("kitten", "sitting", 3), Some(3));
        assert_eq!(levenshtein_within("kitten", "sitting", 2), None);
        assert_eq!(levenshtein_within("abc", "abc", 0), Some(0));
    }
}
