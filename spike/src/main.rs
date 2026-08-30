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
//!
//! STATUS (2026-08-30, Ryzen 5600G): compiles clean and runs. 50 utterances
//! decoded at RTF 0.660 with the oneDNN backend, matching faster-whisper's
//! 0.669 on the same files. English comes out in Latin script with spaces
//! intact — no fusion, so the `suppress_tokens` bug did not bite.
//!
//! One known defect remains, and it is why the output is not yet byte-identical
//! to faster-whisper's: `ct2rs` normalises the mel spectrogram per *frame*
//! rather than per 30-second window (see PLAN.md, "The mel bug"). Fixing it
//! means bypassing `ct2rs::Whisper` for `ct2rs::sys::Whisper`, which accepts a
//! features StorageView directly.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use ct2rs::{Config, Whisper, WhisperOptions};

/// Whisper's fixed input rate. Anything else has to be resampled before it gets
/// here, and for the spike we simply refuse rather than resample badly.
const SAMPLE_RATE: u32 = 16_000;

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
        let text = transcribe(&whisper, &samples, &options)?;
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
fn transcribe(whisper: &Whisper, samples: &[f32], options: &WhisperOptions) -> Result<String> {
    let window = whisper.n_samples();
    let mut out = Vec::new();

    for chunk in samples.chunks(window) {
        // Language is pinned to Bengali. The model is a Bengali fine-tune and
        // letting Whisper auto-detect on code-switched speech is a coin flip.
        let parts = whisper.generate(chunk, Some("bn"), false, options)?;
        out.extend(parts);
    }

    Ok(out.join(" "))
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
