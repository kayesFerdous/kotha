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
//!      suppressed — see the note on `suppress_tokens` in lib.rs.
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
//! The engine itself now lives in `lib.rs`, shared with the Phase 2 loop in
//! `bin/live.rs`, so both run the identical decode path. This binary's job is
//! unchanged: measure it against faster-whisper on files.
//!
//! Against a faster-whisper baseline decoded with matching settings, the two
//! agree on 99.27% of characters — audio, features and library version are all
//! identical, and the remainder traces to MKL vs oneDNN kernels. Do NOT diff
//! against `results/cpu_bench.json`: that used faster-whisper's default
//! temperature-fallback ladder, which samples, and is not reproducible even
//! against itself. `gate.py --baseline matched` builds the right comparison.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use kotha_spike::{read_wav_16k_mono, suspicious_fusion, Engine, SAMPLE_RATE};

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
    let engine = Engine::load(&model_dir, threads)?;
    let load = t0.elapsed();

    println!("loaded  {:.1}s\n", load.as_secs_f64());

    // ------------------------------------------------------------- decode

    let mut total_audio = 0.0f64;
    let mut total_decode = 0.0f64;

    for path in &wavs {
        let samples = read_wav_16k_mono(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let audio_s = samples.len() as f64 / SAMPLE_RATE as f64;

        let t = Instant::now();
        let text = engine.transcribe(&samples)?;
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
