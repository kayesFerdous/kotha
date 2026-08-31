//! Phase 0 — the gate.
//!
//! The smallest program that can tell us whether the whole plan holds: load the
//! published CTranslate2 model with `ct2rs` and transcribe a WAV file.
//!
//! What we are checking, in order of importance:
//!
//!   1. `ct2rs` accepts the model directory as published, with its custom
//!      Bengali BPE tokenizer (vocab 50,364, not Whisper's 51,865).
//!   2. English words come out in **Latin script with spaces around them**.
//!      If they come out fused to their neighbours, the space token is being
//!      suppressed — see the note on `suppress_tokens` below.
//!   3. The output matches faster-whisper's on the same audio.
//!   4. How fast this actually is on an M2.
//!
//! Usage:
//!
//!     cargo run --release -- <model-dir> <file.wav>...
//!
//! Environment:
//!
//!     KOTHA_THREADS   CPU threads (default 0 = let CTranslate2 decide).
//!                     Worth sweeping: 4 (P-cores only) vs 8 on an M2.
//!     KOTHA_DUMP_MEL  Path to write the first window's mel to, for check_mel.py.
//!
//! STATUS (2026-08-30, Ryzen 5600G): works. 50 utterances at RTF 0.639 with the
//! oneDNN backend, against faster-whisper's 0.669 on the same files. English
//! comes out in Latin script with spaces intact — no fusion, so the
//! `suppress_tokens` bug did not bite.
//!
//! This does NOT use `ct2rs::Whisper`. That wrapper's feature extraction is
//! wrong in three ways (see `MelExtractor` below), so we compute the mel here
//! and call `ct2rs::sys::Whisper`, which takes features directly. The result is
//! bit-exact against Whisper's reference — run `check_mel.py` to confirm.
//!
//! Against a faster-whisper baseline decoded with matching settings, the two
//! agree on 99.27% of characters — audio, features and library version are all
//! identical, and the remainder traces to MKL vs oneDNN kernels. Do NOT diff
//! against `results/cpu_bench.json`: that used faster-whisper's default
//! temperature-fallback ladder, which samples, and is not reproducible even
//! against itself. `gate.py --baseline matched` builds the right comparison.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use ct2rs::sys::{StorageView, Whisper, WhisperOptions};
use ct2rs::tokenizers::hf;
use ct2rs::{Config, Tokenizer};
use mel_spec::mel::mel;
use ndarray::Array2;
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

/// Whisper's fixed input rate. Anything else has to be resampled before it gets
/// here, and for the spike we simply refuse rather than resample badly.
const SAMPLE_RATE: u32 = 16_000;

// Whisper's feature geometry. These match the model's preprocessor_config.json
// and are fixed for every Whisper checkpoint, so they are constants rather than
// config reads.
const N_FFT: usize = 400;
const HOP: usize = 160;
const N_MELS: usize = 80;
const N_FRAMES: usize = 3000; // 30 s at one frame per 160 samples
const N_SAMPLES: usize = N_FRAMES * HOP;
const N_BINS: usize = N_FFT / 2 + 1; // 201 — DC through Nyquist

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);

    let model_dir: PathBuf = args
        .next()
        .context("usage: kotha-spike <model-dir> <file.wav>...")?
        .into();

    let wavs: Vec<PathBuf> = args.map(PathBuf::from).collect();
    if wavs.is_empty() {
        bail!("usage: kotha-spike <model-dir> <file.wav>...");
    }

    let threads: usize = std::env::var("KOTHA_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // ---------------------------------------------------------------- load

    println!("model   {}", model_dir.display());
    println!("threads {}", if threads == 0 { "auto".into() } else { threads.to_string() });

    let t0 = Instant::now();
    // NOTE: this is `ct2rs::sys::Whisper`, the low-level handle, not the
    // `ct2rs::Whisper` convenience wrapper. See `log_mel` below for why.
    let whisper = Whisper::new(
        &model_dir,
        Config {
            // 0 means "library default". Everything else is left alone: the
            // default device is CPU, and the default compute type keeps the
            // model's own int8 quantisation rather than re-casting it.
            num_threads_per_replica: threads,
            ..Default::default()
        },
    )
    .context("Whisper::new failed — is this the CTranslate2 model directory, \
              with model.bin, tokenizer.json and preprocessor_config.json in it?")?;

    // The wrapper used to load this for us. Going low-level means loading the
    // tokenizer and building the mel filterbank ourselves.
    let tokenizer = hf::Tokenizer::new(&model_dir)
        .context("could not load tokenizer.json from the model directory")?;
    let mels = MelExtractor::new();

    let load = t0.elapsed();

    println!("loaded  {:.1}s\n", load.as_secs_f64());

    // ------------------------------------------------------------- decode

    // These two settings are not negotiable; see CLAUDE.md §5.
    //
    // `suppress_tokens: vec![]` — under this model's Bengali BPE, token 220 is
    // the space. faster-whisper's Python layer computes a suppression list
    // against OpenAI's token IDs and bans it, fusing English words to their
    // neighbours: English-F1 collapses from ~70 to ~10 while CER barely moves.
    // The C++ engine reads its default from the model's config.json, which
    // carries "suppress_ids": [], so this path is probably already safe. We
    // pass the empty vector anyway, because the failure is silent.
    //
    // `beam_size: 1` — beam 5 is roughly three times slower, which is unusable
    // for live dictation.
    let options = WhisperOptions {
        beam_size: 1,
        suppress_tokens: vec![],
        ..Default::default()
    };

    let mut total_audio = 0.0f64;
    let mut total_decode = 0.0f64;

    for path in &wavs {
        let samples = read_wav_16k_mono(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let audio_s = samples.len() as f64 / SAMPLE_RATE as f64;

        let t = Instant::now();
        let text = transcribe(&whisper, &tokenizer, &mels, &samples, &options)?;
        let decode_s = t.elapsed().as_secs_f64();

        total_audio += audio_s;
        total_decode += decode_s;

        let name = path.file_name().unwrap_or_default().to_string_lossy();
        println!("── {name}  ({audio_s:.1}s audio, {decode_s:.1}s decode, \
                  {:.2}x realtime)", audio_s / decode_s);
        println!("{}\n", text.trim());

        if let Some(warning) = suspicious_fusion(&text) {
            eprintln!("  ⚠ {warning}");
            eprintln!("    This is what the suppress_tokens bug looks like. \
                       Check that suppress_tokens is empty.\n");
        }
    }

    // ------------------------------------------------------------ summary

    println!("────────────────────────────────────────");
    println!("{} file(s)   {total_audio:.1}s audio   {total_decode:.1}s decode",
             wavs.len());
    println!("RTF {:.3}   ({:.2}x realtime)   load {:.1}s",
             total_decode / total_audio,
             total_audio / total_decode,
             load.as_secs_f64());
    println!("\nNow diff this against faster-whisper on the same files:");
    println!("  model.transcribe(path, language=\"bn\", beam_size=1, suppress_tokens=[])");

    Ok(())
}

/// Transcribe, chunking at Whisper's 30-second window.
///
/// Whisper cannot see more than 30 seconds at a time. Audio longer than that is
/// split into fixed windows here, which is crude — the real app cuts on silence
/// instead, so a chunk boundary never lands mid-word. For the gate, where test
/// utterances are all under 30 seconds, this path should never even run.
fn transcribe(
    whisper: &Whisper,
    tokenizer: &hf::Tokenizer,
    mels: &MelExtractor,
    samples: &[f32],
    options: &WhisperOptions,
) -> Result<String> {
    let mut out = Vec::new();

    for chunk in samples.chunks(N_SAMPLES) {
        let mut features = mels.log_mel(chunk);

        // Dump the first window's features so `check_mel.py` can diff them
        // against Whisper's reference implementation. Feature extraction is the
        // one part of this path we wrote ourselves, so it is the one part that
        // needs independent proof.
        if let Ok(path) = std::env::var("KOTHA_DUMP_MEL") {
            if !path.is_empty() {
                let bytes: Vec<u8> = features
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                std::fs::write(&path, bytes)?;
                std::env::set_var("KOTHA_DUMP_MEL", "");
            }
        }

        // Shape is [batch, n_mels, n_frames]; we run one window at a time.
        let view = StorageView::new(
            &[1, N_MELS, N_FRAMES],
            features.as_slice_mut().ok_or_else(|| anyhow!("mel not contiguous"))?,
            Default::default(),
        )?;

        // Language is pinned to Bengali. The model is a Bengali fine-tune and
        // letting Whisper auto-detect on code-switched speech is a coin flip.
        // `<|notimestamps|>` matches faster-whisper's without_timestamps path.
        let prompt = vec![
            "<|startoftranscript|>",
            "<|bn|>",
            "<|transcribe|>",
            "<|notimestamps|>",
        ];

        let res = whisper.generate(&view, &[prompt], options)?;
        let seq = res
            .into_iter()
            .next()
            .and_then(|r| r.sequences.into_iter().next())
            .ok_or_else(|| anyhow!("model returned no sequence"))?;

        out.push(tokenizer.decode(seq)?);
    }

    Ok(out.join(" "))
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
/// on the exact audio faster-whisper saw.
fn read_wav_16k_mono(path: &Path) -> Result<Vec<f32>> {
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
fn suspicious_fusion(text: &str) -> Option<String> {
    let worst = text
        .split_whitespace()
        .filter(|t| t.chars().all(|c| c.is_ascii_alphabetic()))
        .max_by_key(|t| t.len())?;

    (worst.len() > 20).then(|| {
        format!("suspiciously long Latin run ({} chars): {worst:?}", worst.len())
    })
}
