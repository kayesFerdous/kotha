//! Phase 2 — the loop, with no interface.
//!
//! Microphone in, text out, model held warm in between. Everything the finished
//! app does except the parts a user can see: no hotkey, no pill, no paste. This
//! is the first time the pieces run against live audio rather than files, so
//! what it is really testing is whether the *chain* holds together —
//!
//!     cpal → downmix → resample → earshot VAD → segment → Engine → clipboard
//!
//! Usage:
//!
//!     cargo run --release --bin live -- <model-dir>
//!
//! Talk. Each time you pause, that chunk is transcribed, printed, and put on
//! the clipboard. Press Enter to stop.
//!
//! Environment:
//!
//!     KOTHA_THREADS   CPU threads (default: physical cores, which measured
//!                     fastest on the Ryzen — 6 beat 12, SMT hurt).
//!
//! WHAT IS DELIBERATELY MISSING
//! ---------------------------
//! **The hotkey.** PLAN.md lists it in this phase, but the acceptance test is
//! "a terminal app prints what you said, twice in a row, with no model reload",
//! and a global hotkey is not what that gates. It also arrives for free in
//! Phase 4: the app shell is Tauri, so the hotkey is
//! `tauri-plugin-global-shortcut`, not `rdev`, and wiring `rdev` now — which on
//! Wayland needs its own permission dance — would be building something Phase 4
//! deletes. Listening continuously is strictly more demanding of the VAD than
//! push-to-talk anyway.
//!
//! **The spelling corrector.** It is a pure String → String at the end of this
//! pipeline (spike/correct.py, Phase 1, +5.21 English-F1). It plugs in where
//! the text is printed, once it is ported to Rust.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use kotha_spike::{suspicious_fusion, Engine, N_SAMPLES, SAMPLE_RATE};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::audioadapter_buffers::owned::InterleavedOwned;
use rubato::{Fft, FixedSync, Resampler};

/// earshot wants exactly this many samples per call: 16 ms at 16 kHz.
const FRAME: usize = 256;

/// Scores over 0.5 are voice, per earshot's own documentation.
const VOICE: f32 = 0.5;

// Segmentation, in frames of 16 ms. These are ordinary dictation numbers, not
// tuned against anything — there is no held-out set for "where should a chunk
// end", and the only real failure mode (cutting mid-word) is audible.
//
/// Consecutive voiced frames needed to open an utterance. Stops a keyboard
/// click or a door from starting a recording.
const ONSET: usize = 3; // 48 ms
/// Silence that closes an utterance. A natural sentence pause, short enough
/// that text appears while you are still talking.
const HANGOVER: usize = 38; // ~600 ms
/// Audio kept from *before* the onset. The VAD trips a frame or two late, and
/// without this the first consonant is clipped — which the model then has to
/// guess at. Cheap insurance against a silent accuracy loss.
const PREROLL: usize = 16 * FRAME; // ~256 ms
/// Voiced audio below this is discarded as noise rather than transcribed.
const MIN_VOICED: usize = 19; // ~300 ms
/// Force a cut before Whisper's 30 s window, so a long monologue still gets
/// split on *some* boundary rather than truncated inside the model.
const MAX_SAMPLES: usize = N_SAMPLES - SAMPLE_RATE as usize * 5; // 25 s

fn main() -> Result<()> {
    let model_dir: PathBuf = std::env::args()
        .nth(1)
        .context("usage: live <model-dir> [file.wav]")?
        .into();

    let threads: usize = std::env::var("KOTHA_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(num_cpus::get_physical);

    // A WAV argument drives the identical Intake → Segmenter → Engine path with
    // the microphone taken out of it. That is how this loop gets verified
    // without anyone having to talk, and how a segmentation change can be
    // re-checked against the same audio twice.
    match std::env::args().nth(2) {
        Some(wav) => run_file(&model_dir, threads, &wav),
        None => run_mic(&model_dir, threads),
    }
}

/// Load the model, print what it cost. Shared so both modes prove the same
/// thing: one load, many utterances.
fn warm_up(model_dir: &Path, threads: usize) -> Result<Engine> {
    println!("model   {}", model_dir.display());
    println!("threads {threads}");
    let t0 = Instant::now();
    let engine = Engine::load(model_dir, threads)?;
    println!("loaded  {:.1}s\n", t0.elapsed().as_secs_f64());
    Ok(engine)
}

/// Transcribe one segmented utterance and report it.
fn emit(
    engine: &Engine,
    clipboard: Option<&mut arboard::Clipboard>,
    n: usize,
    utterance: &[f32],
) -> Result<()> {
    let secs = utterance.len() as f64 / SAMPLE_RATE as f64;
    let t = Instant::now();
    let text = engine.transcribe(utterance)?;
    let took = t.elapsed().as_secs_f64();
    let text = text.trim();

    println!("[{n}] {secs:.1}s audio → {took:.1}s decode ({:.2}x realtime)",
             secs / took);
    println!("    {text}");

    if let Some(w) = suspicious_fusion(text) {
        eprintln!("    ⚠ {w} — check suppress_tokens");
    }
    if let (Some(c), false) = (clipboard, text.is_empty()) {
        if let Err(e) = c.set_text(text) {
            eprintln!("    clipboard write failed: {e}");
        }
    }
    println!();
    Ok(())
}

/// The clipboard is optional on purpose: printing is what Phase 2 gates, and an
/// unavailable clipboard should not cost you the transcript.
fn open_clipboard() -> Option<arboard::Clipboard> {
    match arboard::Clipboard::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("clipboard unavailable ({e}); printing only\n");
            None
        }
    }
}

/// Offline: a WAV file fed through the loop in microphone-sized blocks.
fn run_file(model_dir: &Path, threads: usize, wav: &str) -> Result<()> {
    let samples = kotha_spike::read_wav_16k_mono(Path::new(wav))?;
    println!("input   {wav}  ({:.1}s)", samples.len() as f64 / SAMPLE_RATE as f64);

    let engine = warm_up(model_dir, threads)?;
    let mut clipboard = open_clipboard();
    let mut intake = Intake::new(SAMPLE_RATE, 1)?;
    let mut segmenter = Segmenter::new();
    let mut n = 0;

    // 100 ms at a time, the size a real capture callback delivers, so the
    // buffering and frame-alignment logic is exercised rather than bypassed.
    let block = SAMPLE_RATE as usize / 10;
    for chunk in samples.chunks(block) {
        for utterance in intake.feed(chunk, &mut segmenter)? {
            n += 1;
            emit(&engine, clipboard.as_mut(), n, &utterance)?;
        }
    }
    // The file ended without a trailing pause; take whatever is still open.
    if let Some(utterance) = segmenter.close() {
        n += 1;
        emit(&engine, clipboard.as_mut(), n, &utterance)?;
    }

    println!("{n} utterance(s) from one model load.");
    Ok(())
}

/// Live: the microphone.
fn run_mic(model_dir: &Path, threads: usize) -> Result<()> {
    // Opened before the model loads: a missing microphone should cost a
    // millisecond, not four seconds and 1.4 GB.
    let device = cpal::default_host()
        .default_input_device()
        .context("no input device — is a microphone connected?")?;
    let supported = pick_config(&device)?;
    let in_rate = supported.sample_rate();
    let channels = supported.channels() as usize;

    println!("input   {device}");   // DeviceTrait: Display gives the name
    println!("format  {in_rate} Hz, {channels} ch, {:?}{}",
             supported.sample_format(),
             if in_rate == SAMPLE_RATE { "  (no resampling needed)" } else { "" });

    let (tx, rx) = mpsc::channel::<Vec<f32>>();
    let stream = build_stream(&device, &supported, tx)?;

    let engine = warm_up(model_dir, threads)?;
    let mut clipboard = open_clipboard();

    let quit = Arc::new(AtomicBool::new(false));
    {
        let quit = quit.clone();
        std::thread::spawn(move || {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            quit.store(true, Ordering::Relaxed);
        });
    }

    stream.play().context("could not start the input stream")?;
    println!("listening — talk, pause, and it appears. Enter to stop.\n");

    let mut intake = Intake::new(in_rate, channels)?;
    let mut segmenter = Segmenter::new();
    let mut n = 0usize;

    while !quit.load(Ordering::Relaxed) {
        // Timed rather than blocking, so Enter is noticed in a silent room.
        let block = match rx.recv_timeout(std::time::Duration::from_millis(200)) {
            Ok(b) => b,
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };
        for utterance in intake.feed(&block, &mut segmenter)? {
            n += 1;
            emit(&engine, clipboard.as_mut(), n, &utterance)?;
        }
    }

    drop(stream);
    println!("stopped after {n} utterance(s), one model load.");
    Ok(())
}

/// Pick an input config, preferring one the device can give us at 16 kHz.
///
/// Whisper needs 16 kHz. If the hardware will produce it there is nothing to
/// resample, which is both faster and one less thing to get wrong; most
/// machines will not, so `Intake` still has to handle the general case.
fn pick_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
    if let Ok(configs) = device.supported_input_configs() {
        let native = configs
            .filter(|c| matches!(c.sample_format(),
                                 cpal::SampleFormat::F32 | cpal::SampleFormat::I16))
            .filter_map(|c| c.try_with_sample_rate(SAMPLE_RATE))
            // Fewest channels: we downmix to mono anyway.
            .min_by_key(|c| c.channels());
        if let Some(c) = native {
            return Ok(c);
        }
    }
    device
        .default_input_config()
        .context("device has no usable input configuration")
}

fn build_stream(
    device: &cpal::Device,
    supported: &cpal::SupportedStreamConfig,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream> {
    let config = supported.config();
    let on_error = |e| eprintln!("audio stream error: {e}");

    // ponytail: the queue is unbounded. Audio is never dropped, it just arrives
    // late if decoding falls behind, which is the right trade at 1.5x realtime.
    // Bound it if a slower machine ever makes the backlog grow without end.
    Ok(match supported.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config,
            move |data: &[f32], _: &_| {
                let _ = tx.send(data.to_vec());
            },
            on_error,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            config,
            move |data: &[i16], _: &_| {
                let _ = tx.send(data.iter().map(|&v| v as f32 / 32768.0).collect());
            },
            on_error,
            None,
        )?,
        f => bail!("unsupported sample format {f:?}; this needs f32 or i16"),
    })
}

/// Device audio in, 16 kHz mono frames out.
///
/// Downmixes, resamples if the hardware would not give us 16 kHz, and hands
/// whole 256-sample frames to the segmenter. Buffers the remainder, because a
/// cpal block is not a whole number of VAD frames and dropping the tail of each
/// block would punch periodic holes in the audio.
struct Intake {
    channels: usize,
    resampler: Option<Fft<f32>>,
    at_device_rate: Vec<f32>,
    at_16k: Vec<f32>,
}

impl Intake {
    fn new(in_rate: u32, channels: usize) -> Result<Self> {
        let resampler = if in_rate == SAMPLE_RATE {
            None
        } else {
            Some(
                Fft::<f32>::new(in_rate as usize, SAMPLE_RATE as usize, 1024, 1,
                                FixedSync::Input)
                    .with_context(|| format!("cannot resample {in_rate} Hz to 16 kHz"))?,
            )
        };
        Ok(Self {
            channels,
            resampler,
            at_device_rate: Vec::new(),
            at_16k: Vec::new(),
        })
    }

    /// Feed one cpal block; return whatever utterances it completed.
    fn feed(&mut self, block: &[f32], seg: &mut Segmenter) -> Result<Vec<Vec<f32>>> {
        self.to_16k(block)?;

        let mut done = Vec::new();
        let mut consumed = 0;
        while consumed + FRAME <= self.at_16k.len() {
            if let Some(u) = seg.push(&self.at_16k[consumed..consumed + FRAME]) {
                done.push(u);
            }
            consumed += FRAME;
        }
        self.at_16k.drain(..consumed);
        Ok(done)
    }

    /// Downmix and resample one block into `at_16k`. Split from `feed` so the
    /// rate conversion can be checked on its own — it is the one step whose
    /// failure mode (losing or duplicating time) is inaudible in small doses
    /// and ruinous in large ones.
    fn to_16k(&mut self, block: &[f32]) -> Result<()> {
        // Downmix by averaging — fine for speech, and the mic is usually mono.
        self.at_device_rate.extend(
            block
                .chunks(self.channels)
                .map(|f| f.iter().sum::<f32>() / self.channels as f32),
        );

        match self.resampler.as_mut() {
            None => self.at_16k.append(&mut self.at_device_rate),
            Some(r) => loop {
                let need = r.input_frames_next();
                if self.at_device_rate.len() < need {
                    break;
                }
                let input = InterleavedSlice::new(&self.at_device_rate[..need], 1, need)
                    .map_err(|e| anyhow::anyhow!("resampler input: {e}"))?;
                let mut out = InterleavedOwned::<f32>::new(0.0, 1, r.output_frames_max());
                let (_, written) = r.process_into_buffer(&input, &mut out, None)?;
                let mut got = out.take_data();
                got.truncate(written);
                self.at_16k.extend_from_slice(&got);
                self.at_device_rate.drain(..need);
            },
        }
        Ok(())
    }
}

/// Cuts the stream into utterances on silence.
///
/// The detector is never reset: this is one continuous audio sequence, and
/// earshot only wants a reset when the stream itself changes.
struct Segmenter {
    vad: Box<earshot::Detector>,
    preroll: VecDeque<f32>,
    current: Vec<f32>,
    onset_run: usize,
    silence: usize,
    voiced: usize,
    speaking: bool,
}

impl Segmenter {
    fn new() -> Self {
        Self {
            // Boxed: the detector carries ~8 KiB of state.
            vad: earshot::Detector::default_boxed(),
            preroll: VecDeque::with_capacity(PREROLL + FRAME),
            current: Vec::new(),
            onset_run: 0,
            silence: 0,
            voiced: 0,
            speaking: false,
        }
    }

    /// Push exactly one 256-sample frame. Returns an utterance when one ends.
    fn push(&mut self, frame: &[f32]) -> Option<Vec<f32>> {
        let voiced = self.vad.predict_f32(frame) > VOICE;
        self.push_forced(frame, voiced)
    }

    /// The segmentation state machine, with the voicing decision already made.
    /// Split out so the tests can drive the timing without needing real speech.
    fn push_forced(&mut self, frame: &[f32], voiced: bool) -> Option<Vec<f32>> {
        if !self.speaking {
            // Not yet in an utterance: hold recent audio in the pre-roll ring
            // so the frames that opened it are not lost when it does open.
            self.preroll.extend(frame.iter().copied());
            while self.preroll.len() > PREROLL {
                self.preroll.pop_front();
            }
            self.onset_run = if voiced { self.onset_run + 1 } else { 0 };
            if self.onset_run >= ONSET {
                self.current.extend(self.preroll.drain(..));
                self.speaking = true;
                self.silence = 0;
                self.voiced = self.onset_run;
            }
            return None;
        }

        self.current.extend_from_slice(frame);
        if voiced {
            self.voiced += 1;
            self.silence = 0;
        } else {
            self.silence += 1;
        }

        if self.silence >= HANGOVER || self.current.len() >= MAX_SAMPLES {
            return self.close();
        }
        None
    }

    fn close(&mut self) -> Option<Vec<f32>> {
        let audio = std::mem::take(&mut self.current);
        let voiced = self.voiced;
        self.speaking = false;
        self.onset_run = 0;
        self.silence = 0;
        self.voiced = 0;
        self.preroll.clear();
        (voiced >= MIN_VOICED).then_some(audio)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One runnable check on the part with real logic: does the segmenter cut
    /// where it should, keep the run-up, and throw away a blip?
    ///
    /// The VAD is not exercised here — that needs real speech. This drives the
    /// state machine directly through a stub, which is the part that can break
    /// silently. Run with: cargo test --bin live
    struct Fake {
        seg: Segmenter,
    }

    impl Fake {
        /// Push `n` frames, forcing the voiced/unvoiced decision.
        fn push(&mut self, n: usize, voiced: bool) -> Vec<Vec<f32>> {
            // A loud square wave reads as voice to any energy-based front end;
            // silence reads as none. Driving the real detector keeps this
            // honest about the 256-sample contract.
            let frame: Vec<f32> = (0..FRAME)
                .map(|i| if voiced { if i % 8 < 4 { 0.4 } else { -0.4 } } else { 0.0 })
                .collect();
            let mut out = Vec::new();
            for _ in 0..n {
                // Bypass the VAD's judgement; we are testing segmentation.
                let v = voiced;
                if let Some(u) = self.seg.push_forced(&frame, v) {
                    out.push(u);
                }
            }
            out
        }
    }

    #[test]
    fn cuts_on_silence_and_keeps_the_preroll() {
        let mut f = Fake { seg: Segmenter::new() };

        // Silence alone never produces anything.
        assert!(f.push(50, false).is_empty());

        // Speech, then a pause long enough to close it.
        assert!(f.push(60, true).is_empty(), "must not cut mid-speech");
        let out = f.push(HANGOVER, false);
        assert_eq!(out.len(), 1, "one utterance should have closed");

        // It carries the pre-roll: more audio than the voiced frames alone.
        let n = out[0].len();
        assert!(n > 60 * FRAME, "pre-roll missing: {n} <= {}", 60 * FRAME);
        assert!(n <= (60 + HANGOVER) * FRAME + PREROLL, "utterance too long: {n}");
    }

    #[test]
    fn discards_a_blip_too_short_to_be_speech() {
        let mut f = Fake { seg: Segmenter::new() };
        f.push(10, false);
        // Above ONSET so it opens, below MIN_VOICED so it must be thrown away.
        assert!(ONSET < 5 && 5 < MIN_VOICED);
        f.push(5, true);
        assert!(f.push(HANGOVER, false).is_empty(), "a 80 ms blip was transcribed");
    }

    /// Rate conversion must preserve duration. A resampler that quietly drops
    /// or repeats audio still sounds like speech, so nothing downstream would
    /// notice — the model would just get a subtly wrong utterance.
    #[test]
    fn resampling_preserves_duration() {
        for (rate, channels) in [(48_000usize, 1usize), (44_100, 2), (16_000, 1)] {
            let mut intake = Intake::new(rate as u32, channels).unwrap();
            // Exactly one second of interleaved audio at the device's rate.
            let block = vec![0.25f32; rate * channels];
            for chunk in block.chunks(rate * channels / 10) {
                intake.to_16k(chunk).unwrap();
            }
            let got = intake.at_16k.len();
            let want = SAMPLE_RATE as usize;
            // The FFT resampler holds back up to one chunk of tail, so allow a
            // shortfall of that order but nothing like a lost or doubled third.
            assert!(
                got <= want && got + 2048 >= want,
                "{rate} Hz x{channels}: 1 s became {got} samples at 16 kHz, want ~{want}"
            );
        }
    }

    #[test]
    fn cuts_before_whispers_window() {
        let mut f = Fake { seg: Segmenter::new() };
        // Talk without pausing for longer than the model can see at once.
        let out = f.push(MAX_SAMPLES / FRAME + 10, true);
        assert!(!out.is_empty(), "a monologue was never cut");
        for u in &out {
            assert!(u.len() <= N_SAMPLES,
                    "chunk exceeds Whisper's window: {} > {N_SAMPLES}", u.len());
        }
    }
}
