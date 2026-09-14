//! The engine: model, tokenizer and feature extraction behind one warm handle.
//!
//! Lifted out of the Phase 0 gate binary unchanged so that the Phase 2 loop
//! (`src/bin/live.rs`) runs the *same* code the gate measured, rather than a
//! second copy that could drift from it. The mel path in particular is the one
//! part of the inference chain we implement ourselves, and it is verified
//! bit-exact against Whisper's reference — see `MelExtractor` and check_mel.py.
//!
//! Nothing here changed in the move. `KOTHA_DUMP_MEL=... check_mel.py` is the
//! proof of that and must be re-run after any edit to this file.

pub mod correct;
pub mod live;

use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use ct2rs::sys::{StorageView, Whisper, WhisperOptions};
use ct2rs::tokenizers::hf;
use ct2rs::{Config, Tokenizer};
use mel_spec::mel::mel;
use ndarray::Array2;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Whisper's fixed input rate. Audio arrives here already resampled.
pub const SAMPLE_RATE: u32 = 16_000;

/// Below this average log-probability, a decode is thrown away as a
/// hallucination rather than typed at the user.
///
/// **Whisper does not answer silence with silence.** Given breath, a fan or a
/// keyboard it returns a fluent sentence it learnt — confidently, and different
/// every time, so the repetition collapser never sees it.
///
/// The usual filter for this is faster-whisper's `no_speech_prob > 0.6`. **It
/// does not work on this model.** Measured 2026-09-02: `no_speech_prob` came
/// back between 0.0000 and 0.0003 on eight synthetic noise files including
/// digital silence — the fine-tune never learnt to emit `<|nospeech|>`, because
/// its training data is all speech. The token is there and its calibration is
/// gone.
///
/// The average log-probability does separate them, cleanly:
///
/// | | n | worst | median | best |
/// |---|---|---|---|---|
/// | real speech | 40 | **-0.142** | -0.040 | -0.008 |
/// | noise | 8 | -1.047 | -0.35 | **-0.198** |
///
/// The real speech is 40 random clips from the *training* corpus, deliberately
/// not the paper's 393 — a threshold fitted to the evaluation set would be
/// tuning on it. The noise is synthetic: hiss at three levels, 50 and 60 Hz
/// hum, clicks, a breath-like envelope, and digital silence.
///
/// **-0.20** sits in the gap, biased towards keeping speech: it leaves 0.06 of
/// margin below the worst real utterance and still rejects seven of the eight
/// noise files. That is a small sample and real rooms are not synthetic noise,
/// so it is a **calibration knob, not a constant** — `KOTHA_MIN_LOGPROB`
/// overrides it, and `KOTHA_MIN_LOGPROB=-99` turns the gate off entirely.
const LOGPROB_FLOOR: f32 = -0.20;

fn logprob_floor() -> f32 {
    std::env::var("KOTHA_MIN_LOGPROB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(LOGPROB_FLOOR)
}

// Whisper's feature geometry. These match the model's preprocessor_config.json
// and are fixed for every Whisper checkpoint, so they are constants rather than
// config reads.
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 80;
const N_FRAMES: usize = 3000; // 30 s at one frame per 160 samples
/// Whisper's 30-second window, in samples. Nothing longer can be seen at once.
pub const N_SAMPLES: usize = N_FRAMES * HOP;
const N_BINS: usize = N_FFT / 2 + 1; // 201 — DC through Nyquist

/// A loaded model, held warm across calls.
///
/// Loading costs ~4 s and 1.4 GB, so this is built once and reused. Phase 2's
/// acceptance test is precisely that two utterances in a row share one of
/// these.
pub struct Engine {
    whisper: Whisper,
    tokenizer: hf::Tokenizer,
    mels: MelExtractor,
    options: WhisperOptions,
}

impl Engine {
    /// `threads` of 0 means "let CTranslate2 decide".
    pub fn load(model_dir: &Path, threads: usize) -> Result<Self> {
        // NOTE: this is `ct2rs::sys::Whisper`, the low-level handle, not the
        // `ct2rs::Whisper` convenience wrapper, whose feature extraction is
        // wrong in three ways. See `MelExtractor` below.
        let whisper = Whisper::new(
            model_dir,
            Config {
                // Everything else is left alone: the default device is CPU, and
                // the default compute type keeps the model's own int8
                // quantisation rather than re-casting it.
                num_threads_per_replica: threads,
                ..Default::default()
            },
        )
        .context(
            "Whisper::new failed — is this the CTranslate2 model directory, \
             with model.bin, tokenizer.json and preprocessor_config.json in it?",
        )?;

        let tokenizer = hf::Tokenizer::new(model_dir)
            .context("could not load tokenizer.json from the model directory")?;

        Ok(Self {
            whisper,
            tokenizer,
            mels: MelExtractor::new(),
            // These two settings are not negotiable.
            //
            // `suppress_tokens: vec![]` — under this model's Bengali BPE, token
            // 220 is the space. faster-whisper's Python layer computes a
            // suppression list against OpenAI's token IDs and bans it, fusing
            // English words to their neighbours: English-F1 collapses from ~70
            // to ~10 while CER barely moves. The C++ engine reads its default
            // from the model's config.json, which carries "suppress_ids": [],
            // so this path is probably already safe. We pass the empty vector
            // anyway, because the failure is silent.
            //
            // `beam_size: 1` — beam 5 is roughly three times slower, which is
            // unusable for live dictation.
            options: WhisperOptions {
                beam_size: 1,
                suppress_tokens: vec![],
                // Off by default, and the whole silence gate below depends on
                // it. See `NO_SPEECH`.
                // The gate below is built on `scores`. `no_speech_prob` is
                // deliberately not requested: it is ~0 on this model even for
                // digital silence. See `LOGPROB_FLOOR`.
                return_scores: true,
                ..Default::default()
            },
        })
    }

    /// Transcribe 16 kHz mono audio, chunking at Whisper's 30-second window.
    ///
    /// Audio longer than 30 s is split into fixed windows here, which is crude
    /// — a boundary can land mid-word. The live loop cuts on silence before it
    /// ever gets here, so this path is a backstop, not the normal route.
    pub fn transcribe(&self, samples: &[f32]) -> Result<String> {
        let mut out = Vec::new();

        for chunk in samples.chunks(N_SAMPLES) {
            let mut features = self.mels.log_mel(chunk);

            // Dump the first window's features so `check_mel.py` can diff them
            // against Whisper's reference implementation. Feature extraction is
            // the one part of this path we wrote ourselves, so it is the one
            // part that needs independent proof.
            if let Ok(path) = std::env::var("KOTHA_DUMP_MEL") {
                if !path.is_empty() {
                    let bytes: Vec<u8> =
                        features.iter().flat_map(|v| v.to_le_bytes()).collect();
                    std::fs::write(&path, bytes)?;
                    std::env::set_var("KOTHA_DUMP_MEL", "");
                }
            }

            // Shape is [batch, n_mels, n_frames]; we run one window at a time.
            let view = StorageView::new(
                &[1, N_MELS, N_FRAMES],
                features
                    .as_slice_mut()
                    .ok_or_else(|| anyhow!("mel not contiguous"))?,
                Default::default(),
            )?;

            // Language is pinned to Bengali, and this is not a default that
            // is waiting for a setting — it is the only value that means
            // anything on this model. **Measured 2026-09-14**, five test clips
            // decoded twice with `<|bn|>` and `<|en|>`: three came back
            // byte-identical, and the two that differed differed like this —
            //
            //     bn: আল্লাহামদুলিল্লাহ এটা একটা great opportunity
            //     en: alhamdullah         এটা একটা grate opportunity
            //
            // — still Bangla, with one word pushed out of Bangla script and
            // "great" misspelled. The token reaches the model (the average
            // logprob moves, -0.058 to -0.056) and the model does not care:
            // it was fine-tuned on `<|bn|>` hard enough that the language
            // embedding is inert. A user-facing language switch was built on
            // top of this and removed the same day, because what it offered
            // was a choice between Bangla and slightly worse Bangla.
            //
            // Auto-detection is separately a coin flip on code-switched
            // speech, which is the other reason there is nothing to choose.
            // `<|notimestamps|>` matches faster-whisper's without_timestamps
            // path.
            let prompt = vec![
                "<|startoftranscript|>",
                "<|bn|>",
                "<|transcribe|>",
                "<|notimestamps|>",
            ];

            let res = self.whisper.generate(&view, &[prompt], &self.options)?;
            let r = res.into_iter().next().ok_or_else(|| anyhow!("model returned no result"))?;

            // How sure the model is, averaged over the tokens it produced.
            // Printed every time, not only on a rejection: tuning the floor
            // needs to see what the accepted decodes score too.
            let logprob = r.scores.first().copied().unwrap_or(0.0);
            let floor = logprob_floor();
            println!("    confidence {logprob:.3}");
            if logprob < floor {
                println!("    ↳ discarded as noise (floor {floor:.2}) — nothing was typed");
                continue;
            }

            let seq = r
                .sequences
                .into_iter()
                .next()
                .ok_or_else(|| anyhow!("model returned no sequence"))?;

            // Repair a stuck decode here rather than in each front end: the
            // CLI and the app both go through this call, and a defect fixed in
            // one of them only is a defect that comes back. The raw text is
            // logged when it fires, because a loop is a decode bug and the
            // collapsed text is no longer evidence of one.
            let text = self.tokenizer.decode(seq)?;
            let collapsed = collapse_loops(&text);
            if collapsed != text {
                eprintln!("    \u{26a0} repetition loop collapsed, raw was: {text:?}");
            }
            out.push(collapsed);
        }

        Ok(out.join(" "))
    }
}

/// Whisper's log-mel spectrogram.
///
/// We compute this ourselves rather than calling `ct2rs::Whisper::generate`,
/// which gets it wrong in three separate ways. None of them crashes or warns;
/// they surface only as text that quietly disagrees with faster-whisper's.
///
///   1. **Normalisation scope.** `ct2rs` calls `mel_spec`'s `norm_mel` on one
///      frame at a time, so the `max - 8.0` dynamic-range floor is computed
///      from each frame's own peak. Whisper takes that maximum over the whole
///      30-second window. Per-frame normalisation acts as an automatic gain
///      control: it lifts silence and flattens loud frames.
///   2. **Padding.** `ct2rs` computes frames only where audio exists and leaves
///      the rest of the array at 0.0. Whisper zero-pads the *audio* to 30 s, so
///      trailing frames carry the mel of silence — a large negative number that
///      clamps to the floor. 0.0 is nowhere near it.
///   3. **Framing.** `mel_spec`'s `Spectrogram` is overlap-and-save with no
///      centring, where Whisper uses `torch.stft(center=True)` — reflection-pad
///      by `n_fft/2`, then frame. Its 400-sample buffer also advances by a
///      160-sample hop, putting frames at offset 80 from each hop boundary.
///      Because 400 is not a multiple of 160 that offset cannot be tuned away,
///      so `mel_spec`'s STFT cannot reproduce Whisper's framing at these
///      parameters at all. (Its `log_mel_spectrogram` additionally drops the
///      Nyquist bin and substitutes a literal 0.0.)
///
/// So `mel_spec` is kept only for its filterbank matrix, which is the librosa
/// Slaney-normalised one Whisper expects, and the STFT is done here.
struct MelExtractor {
    filters: Array2<f64>,
    window: Vec<f64>,
    fft: std::sync::Arc<dyn Fft<f64>>,
}

impl MelExtractor {
    fn new() -> Self {
        Self {
            filters: mel(SAMPLE_RATE as f64, N_FFT, N_MELS, None, None, false, true),
            // Periodic Hann, matching torch.hann_window's default.
            window: (0..N_FFT)
                .map(|i| {
                    0.5 * (1.0
                        - (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos())
                })
                .collect(),
            fft: FftPlanner::new().plan_fft_forward(N_FFT),
        }
    }

    /// Returns the normalised log-mel, shape `[N_MELS, N_FRAMES]`.
    fn log_mel(&self, samples: &[f32]) -> Array2<f32> {
        // Zero-pad the audio out to the full 30-second window first, so silent
        // frames get a real (very negative) mel value rather than staying 0.
        let mut audio: Vec<f64> = samples.iter().map(|&v| v as f64).collect();
        audio.resize(N_SAMPLES, 0.0);

        // Reflection padding, which is what `center=True` does: mirror around
        // the edge sample without repeating it.
        let pad = N_FFT / 2;
        let mut padded = Vec::with_capacity(N_SAMPLES + N_FFT);
        padded.extend((1..=pad).rev().map(|i| audio[i]));
        padded.extend_from_slice(&audio);
        padded.extend((1..=pad).map(|i| audio[N_SAMPLES - 1 - i]));

        // Power spectrum, one column per frame.
        let mut power = Array2::<f64>::zeros((N_BINS, N_FRAMES));
        let mut buf = vec![Complex::new(0.0, 0.0); N_FFT];
        let mut scratch = vec![Complex::new(0.0, 0.0); self.fft.get_inplace_scratch_len()];

        for j in 0..N_FRAMES {
            let frame = &padded[j * HOP..j * HOP + N_FFT];
            for (b, (&x, &w)) in frame.iter().zip(self.window.iter()).enumerate() {
                buf[b] = Complex::new(x * w, 0.0);
            }
            self.fft.process_with_scratch(&mut buf, &mut scratch);
            // Whisper keeps bins 0..=Nyquist and uses squared magnitude.
            for b in 0..N_BINS {
                power[[b, j]] = buf[b].norm_sqr();
            }
        }

        // Filterbank, then log10 with Whisper's 1e-10 floor.
        let mut logspec = self.filters.dot(&power);
        logspec.mapv_inplace(|v| v.max(1e-10).log10());

        // One dynamic-range floor for the whole window, then Whisper's scaling.
        let global_max = logspec.fold(f64::NEG_INFINITY, |a, &v| a.max(v));
        let floor = global_max - 8.0;
        logspec.mapv(|v| ((v.max(floor) + 4.0) / 4.0) as f32)
    }
}

/// Read a WAV as mono f32 in [-1, 1].
///
/// Refuses anything that is not already 16 kHz. Silently resampling here would
/// make the gate measure the wrong thing — we want to know how the model does
/// on the exact audio faster-whisper saw. (The live loop resamples on the way
/// in instead, because a microphone gives us no choice.)
pub fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();

    if spec.sample_rate != SAMPLE_RATE {
        bail!(
            "{} is {} Hz; this spike needs {} Hz.\n    \
             Convert it:  ffmpeg -i in.wav -ar 16000 -ac 1 out.wav",
            path.display(),
            spec.sample_rate,
            SAMPLE_RATE
        );
    }

    let mut samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<Result<_, _>>()?
        }
    };

    // Downmix by averaging. Fine for speech; the real app records mono anyway.
    if spec.channels > 1 {
        let n = spec.channels as usize;
        samples = samples.chunks(n).map(|f| f.iter().sum::<f32>() / n as f32).collect();
    }

    if samples.is_empty() {
        bail!("{} contains no audio", path.display());
    }

    Ok(samples)
}

/// Heuristic for the failure mode we are most afraid of.
///
/// When the space token is suppressed, English words fuse into long runs of
/// Latin letters — `goodfoodandthen` instead of `good food and then`. Real
/// English words in this domain are short, so an unusually long Latin run is a
/// strong signal that something is wrong. CER barely moves when this happens,
/// which is exactly why it needs its own check.
pub fn suspicious_fusion(text: &str) -> Option<String> {
    let worst = text
        .split_whitespace()
        .filter(|t| t.chars().all(|c| c.is_ascii_alphabetic()))
        .max_by_key(|t| t.len())?;

    (worst.len() > 20).then(|| {
        format!("suspiciously long Latin run ({} chars): {worst:?}", worst.len())
    })
}

/// Collapse a decode that got stuck repeating itself.
///
/// Whisper loops on roughly a quarter of live utterances — `তোমার কথা কথা কথা
/// কথা … ধরো ভালো আছে।` — and it is the app's worst defect. Phase 2's proposed
/// detector (watch the decode clock) is dead: the observed loops decoded at
/// 1.0x real time, indistinguishable from a healthy one. The text gives it away
/// and nothing else does.
///
/// This repairs the symptom, not the cause. The cause needs a
/// `repetition_penalty` sweep on the paper's 393-utterance set with the strict
/// English-F1 and CER harness, which is not something to guess at — and
/// `no_repeat_ngram_size` is probably the wrong knob for Bengali, where
/// reduplication is grammatical (`করতে করতে` is correct at n=2).
///
/// Which is also why the thresholds are what they are. A run of two is ordinary
/// Bengali; three is ordinary *speech* (`না না না`). Four in a row is not
/// something a person says, so `MIN_RUN` is 4 and the run collapses to a single
/// copy — for a real loop the true count is one, and leaving two behind would
/// be inventing a word the speaker did not say. Phrases loop as readily as
/// single words, so a repeat unit is up to `MAX_PHRASE` tokens; the shortest
/// unit is tried first, so `কথা কথা কথা কথা` is one word four times and not one
/// pair twice.
///
/// ponytail: symptom repair. Delete this once a decode-side penalty is measured.
const MIN_RUN: usize = 4;
const MAX_PHRASE: usize = 4;

pub fn collapse_loops(text: &str) -> String {
    let toks: Vec<&str> = text.split_whitespace().collect();
    let mut out: Vec<&str> = Vec::with_capacity(toks.len());
    let mut i = 0;

    while i < toks.len() {
        let run = (1..=MAX_PHRASE.min(toks.len() - i)).find_map(|n| {
            let mut reps = 1;
            while i + n * (reps + 1) <= toks.len()
                && toks[i..i + n] == toks[i + n * reps..i + n * (reps + 1)]
            {
                reps += 1;
            }
            (reps >= MIN_RUN).then_some((n, reps))
        });

        match run {
            Some((n, reps)) => {
                out.extend_from_slice(&toks[i..i + n]);
                i += n * reps;
            }
            None => {
                out.push(toks[i]);
                i += 1;
            }
        }
    }

    out.join(" ")
}

#[cfg(test)]
mod loop_tests {
    use super::collapse_loops;

    #[test]
    fn collapses_a_stuck_word_but_leaves_reduplication_alone() {
        // The real thing, from the first live dictation.
        assert_eq!(
            collapse_loops("তোমার কথা কথা কথা কথা কথা কথা ধরো ভালো আছে।"),
            "তোমার কথা ধরো ভালো আছে।"
        );
        // Grammatical reduplication, and emphatic speech, must survive.
        assert_eq!(collapse_loops("ধীরে ধীরে করতে করতে"), "ধীরে ধীরে করতে করতে");
        assert_eq!(collapse_loops("না না না ভাই"), "না না না ভাই");
    }

    #[test]
    fn collapses_a_stuck_phrase_and_keeps_the_tail() {
        assert_eq!(
            collapse_loops("ami ভালো আছি ভালো আছি ভালো আছি ভালো আছি thanks"),
            "ami ভালো আছি thanks"
        );
        // Nothing to do is a no-op, including on nothing at all.
        assert_eq!(collapse_loops("hello world"), "hello world");
        assert_eq!(collapse_loops(""), "");
    }
}
