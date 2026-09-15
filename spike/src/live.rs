//! Phase 2 — the loop, with no interface.
//!
//! Microphone in, text out, model held warm in between. Everything the finished
//! app does except the parts a user can see: no hotkey, no pill, no paste. This
//! is the first time the pieces run against live audio rather than files, so
//! what it is really testing is whether the *chain* holds together —
//!
//! ```text
//! cpal → downmix → resample → earshot VAD → segment → Engine → clipboard
//! ```
//!
//! Usage:
//!
//! ```bash
//! cargo run --release --bin live -- <model-dir>
//! ```
//!
//! Talk. Each time you pause, that chunk is transcribed, printed, and put on
//! the clipboard. Press Enter to stop.
//!
//! Environment:
//!
//! ```text
//! KOTHA_THREADS   CPU threads (default: physical cores, which measured
//!                 fastest on the Ryzen — 6 beat 12, SMT hurt).
//! ```
//!
//! WHAT IS DELIBERATELY MISSING
//! ---------------------------
//! **The hotkey.** It belongs to this phase on paper, but the acceptance test is
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
use std::time::{Duration, Instant};

use anyhow::{bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use crate::correct::Corrector;
use crate::{suspicious_fusion, Engine, N_SAMPLES, SAMPLE_RATE};
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

/// The chord that means "paste" on this platform.
#[cfg(target_os = "macos")]
const PASTE_MODIFIER: Key = Key::Meta; // ⌘V
#[cfg(not(target_os = "macos"))]
const PASTE_MODIFIER: Key = Key::Control; // Ctrl+V

/// The V in that chord — on Windows as the physical key, not the character.
///
/// `Key::Unicode('v')` asks the *active keyboard layout* which key types a "v".
/// On a Bengali layout there may be no such key, and enigo's Windows backend
/// then falls back to typing the character as text (`win_impl.rs`, "Falling
/// back to entering it as text"). Ctrl held over typed text is not a paste, so
/// the dictation would reach the clipboard and stop there — on exactly the
/// machines Kotha is for. `Key::V` is the virtual key `VK_V`, which is the
/// same key whatever layout is active, and it is what applications read a
/// Ctrl+V shortcut from. Windows only, because enigo only has it there.
#[cfg(target_os = "windows")]
const PASTE_KEY: Key = Key::V;
#[cfg(not(target_os = "windows"))]
const PASTE_KEY: Key = Key::Unicode('v');

/// How long the target application gets to read the clipboard before the
/// previous contents go back.
///
// ponytail: a fixed delay, not a handshake. There is no portable way to be told
// "the paste has been read" — on Wayland our own process serves the selection,
// but arboard does not surface that request. Restoring too early would make the
// target paste the *previous* text, which is worse than never restoring at all,
// so this errs long: 400 ms is invisible next to a multi-second decode. If a
// loaded application ever pastes stale text, raise it.
const PASTE_SETTLE: Duration = Duration::from_millis(400);

/// The command-line loop. `src/bin/live.rs` is a three-line shim over this.
pub fn run() -> Result<()> {
    let model_dir: PathBuf = std::env::args()
        .nth(1)
        .context("usage: live <model-dir> [file.wav]  |  live --correct")?
        .into();

    // The corrector on its own, as a stdin filter — no model, no microphone.
    // This exists to be diffed against the Python prototype it was ported
    // from, which has the identical mode:
    //
    //     live --correct < normalised.txt   |   correct.py < normalised.txt
    //
    // Token-level and whitespace-joined, matching the prototype exactly; the
    // punctuation- and case-preserving path is `correct_text`, which is what
    // the pipeline below uses.
    if model_dir.as_os_str() == "--correct" {
        let c = Corrector::new();
        for line in std::io::stdin().lines() {
            let line = line?;
            let fixed: Vec<&str> =
                line.split_whitespace().map(|t| c.correct_token(t)).collect();
            println!("{}", fixed.join(" "));
        }
        return Ok(());
    }

    let threads = decode_threads();

    // A WAV argument drives the identical Intake → Segmenter → Engine path with
    // the microphone taken out of it. That is how this loop gets verified
    // without anyone having to talk, and how a segmentation change can be
    // re-checked against the same audio twice.
    match std::env::args().nth(2) {
        Some(wav) => run_file(&model_dir, threads, &wav),
        None => run_mic(&model_dir, threads),
    }
}

/// How the user asked for text to be delivered.
///
/// Synthetic input is opt-in twice over. Sending keystrokes into whichever
/// window happens to be focused is not something to do because a program was
/// started, and the portal route additionally raises a permission dialog — so
/// that one has to be named.
///
/// The app now carries this as a setting — the tray's Text output submenu,
/// saved in `settings.json` — and reads `KOTHA_PASTE` only as an override. This
/// function is what the spike binaries still use, and what that override goes
/// through.
///
/// ```text
/// KOTHA_PASTE unset   clipboard only
/// KOTHA_PASTE=1       clipboard + paste, no permission dialog
/// KOTHA_PASTE=portal  clipboard + paste through the desktop portal
/// ```
pub fn paste_mode() -> (bool, bool) {
    match std::env::var("KOTHA_PASTE").as_deref() {
        Ok("portal") => (true, true),
        Ok("1") | Ok("true") => (true, false),
        _ => (false, false),
    }
}

/// Load the model, print what it cost. Shared so both modes prove the same
/// thing: one load, many utterances.
fn warm_up(model_dir: &Path, threads: usize) -> Result<Engine> {
    // Before `Engine::load`, not after: the engine's thread pool inherits the
    // QoS of whichever thread builds it. See `prefer_performance_cores`.
    prefer_performance_cores();
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
    corrector: &Corrector,
    out: &mut Output,
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
    // Phase 1, at the end of the pipeline where it always belonged: a pure
    // String → String that repairs the English and cannot touch the Bengali.
    // Both lines are printed when it fires, because the raw output is what a
    // decode bug shows up in and the corrected one is what the user gets.
    let fixed = corrector.correct_text(text);
    if fixed != text {
        println!("  → {fixed}");
    }

    out.deliver(&fixed);
    println!();
    Ok(())
}

/// Where dictated text goes — Phase 3.
///
/// Two modes, and the fallback matters as much as the mechanism:
///
///   **copy**   The clipboard and nothing else. Needs no permission on any
///              platform and cannot fail for permission reasons; the user
///              presses paste. This is what the app does before anything has
///              been granted, and what it degrades to when injection is
///              refused. Default here, because firing synthetic keystrokes into
///              whatever window happens to be focused should be asked for.
///
///   **paste**  The clipboard plus a synthetic paste, with the previous
///              clipboard contents put back afterwards. `KOTHA_PASTE=1`.
///
/// A paste and not per-character typing, because Bengali conjuncts and
/// combining marks break character-by-character injection in many applications
/// A paste is atomic.
pub struct Output {
    clipboard: Option<arboard::Clipboard>,
    keyboard: Option<Enigo>,

    /// A fresh permission token from the desktop portal, if this connection
    /// got one. Whoever owns the settings file should save it and hand it back
    /// to the next `open`, which is what stops the desktop asking again.
    ///
    /// It changes every time it is used, so saving it once is not enough — the
    /// old one is dead the moment a new one arrives.
    pub restore_token: Option<String>,
}

impl Output {
    /// `restore_token` is the one saved by a previous run, or `None` the first
    /// time.
    ///
    /// It is **not** portal-only, which is what an earlier version of this
    /// comment claimed and what cost a permission dialog on every launch: on
    /// KDE Wayland the plain `paste` route reaches libei too, and asks for the
    /// same grant. Every branch of `connect_keyboard` takes it now.
    pub fn open((paste, portal): (bool, bool), restore_token: Option<String>) -> Self {
        let clipboard = match arboard::Clipboard::new() {
            Ok(c) => Some(c),
            Err(e) => {
                // Not fatal. Printing is what Phase 2 gates, and an unavailable
                // clipboard should not cost you the transcript.
                eprintln!("clipboard unavailable ({e}); printing only");
                None
            }
        };

        // Said out loud, because the two outcomes look identical from here:
        // the desktop either restores the grant silently or puts the dialog up
        // again, and this process is not told which. If a dialog appears on a
        // launch whose log says "offering the saved permission", the token was
        // refused — which is a different bug from never having sent one.
        if paste {
            println!(
                "output  {}",
                match &restore_token {
                    Some(_) => "offering the saved permission back to the desktop",
                    None => "no saved permission yet — the desktop will ask once",
                }
            );
        }

        let mut how = "";
        let keyboard = paste.then(|| connect_keyboard(portal, restore_token)).and_then(|r| match r {
            Ok((k, backend)) => {
                how = backend;
                Some(k)
            }
            Err(e) => {
                eprintln!("synthetic paste unavailable ({e})");
                if let Some(hint) = permission_hint(&e) {
                    eprintln!("{hint}");
                }
                eprintln!("        falling back to clipboard only — press paste \
                           yourself, nothing is lost");
                None
            }
        });

        match (&clipboard, &keyboard) {
            (Some(_), Some(_)) => println!("output  clipboard + synthetic paste ({how})"),
            (Some(_), None) => {
                println!("output  clipboard only — Text output in the tray menu, \
                          or KOTHA_PASTE=1 / =portal, to paste at the cursor")
            }
            (None, _) => println!("output  terminal only"),
        }
        let restore_token = keyboard.as_ref().and_then(fresh_token);
        if restore_token.is_some() {
            println!("output  the desktop granted a lasting permission — saved, \
                      so it should not ask again");
        }
        Self { clipboard, keyboard, restore_token }
    }

    pub fn deliver(&mut self, text: &str) {
        let Some(cb) = self.clipboard.as_mut() else { return };
        if text.is_empty() {
            return;
        }

        let Some(kb) = self.keyboard.as_mut() else {
            // Copy-only: the clipboard IS the delivery, so it is not restored.
            if let Err(e) = cb.set_text(text) {
                eprintln!("    clipboard write failed: {e}");
            }
            return;
        };

        let borrowed = Borrowed::take(cb);
        if let Borrowed::Unknown = borrowed {
            // Pasting has to own the clipboard, so this cannot be avoided —
            // only reported. Never drop data silently.
            eprintln!("    note: the clipboard held something this app cannot \
                       read back (a file list, or an app-specific format).");
            eprintln!("          Pasting will replace it.");
        }
        if let Err(e) = cb.set_text(text) {
            eprintln!("    clipboard write failed: {e}; not pasting");
            return;
        }
        if let Err(e) = paste_chord(kb) {
            // Leave our text on the clipboard rather than restoring: the paste
            // did not happen, so the user still needs something to paste.
            eprintln!("    paste failed: {e}; the text is on the clipboard");
            return;
        }
        std::thread::sleep(PASTE_SETTLE);
        borrowed.give_back(cb, text);
    }
}

/// Whatever was on the clipboard before Kotha borrowed it to paste.
///
/// Pasting means owning the clipboard, so the user's own copied thing has to be
/// displaced for a moment and then put back. Getting that wrong is the kind of
/// bug that makes an app feel untrustworthy — you copy a link, dictate a
/// sentence, and the link is gone.
enum Borrowed {
    Text(String),
    /// A screenshot is a perfectly ordinary thing to have copied, and losing it
    /// to a dictation would be worse than the dictation was worth.
    Image(arboard::ImageData<'static>),
    /// A file list, or an application's private format. It cannot be read back,
    /// and pasting will overwrite it regardless — so this exists to be
    /// *reported* rather than silently destroyed.
    Unknown,
    Nothing,
}

impl Borrowed {
    fn take(cb: &mut arboard::Clipboard) -> Self {
        match cb.get_text() {
            Ok(t) if !t.is_empty() => return Self::Text(t),
            Ok(_) => return Self::Nothing,
            Err(_) => {}
        }
        match cb.get_image() {
            Ok(img) => Self::Image(img),
            Err(arboard::Error::ContentNotAvailable) => Self::Nothing,
            Err(_) => Self::Unknown,
        }
    }

    /// Put it back — but only if our text is still what is on the clipboard.
    ///
    /// If anything changed it in the meantime, that was the user copying
    /// something, and overwriting their fresh copy with a stale backup would be
    /// a worse bug than the one this is fixing.
    fn give_back(self, cb: &mut arboard::Clipboard, ours: &str) {
        // arboard::Error is not PartialEq, so compare the Ok side only.
        if cb.get_text().map(|t| t != ours).unwrap_or(true) {
            return;
        }
        match self {
            Self::Text(t) => drop(cb.set_text(t)),
            Self::Image(img) => drop(cb.set_image(img)),
            // Nothing to put back. Leaving the dictated text is friendlier than
            // clearing: if the paste missed, the user can still paste it again.
            Self::Nothing | Self::Unknown => {}
        }
    }
}

/// Connect to exactly one input backend, and say which.
///
/// enigo sends every keystroke through *all* the connections it managed to
/// open, not the first that works. A Linux session with two live connections
/// therefore pastes **twice** — silently, and only on some compositors, which
/// is the worst way for a bug like this to behave.
///
/// The session is a runtime fact and enigo's backends are compile-time
/// features, and `Settings` has no switch to turn one off. So the choice is
/// made here by pointing the unwanted backends at a display that cannot exist.
///
/// Three routes, in descending order of how much they can reach:
///
///   **portal** — libei through the XDG RemoteDesktop portal. Reaches every
///   window, native Wayland included. Costs a permission dialog, so it is asked
///   for by name (`KOTHA_PASTE=portal`) rather than sprung on anyone.
///
///   **wayland** — `zwp_virtual_keyboard_v1`. Reaches everything, no dialog,
///   but only wlroots compositors (Sway, Hyprland) offer it. Measured on KDE
///   Plasma 6 / kwin_wayland, 2026-08-31: KWin does **not**.
///
///   **x11** — XTEST. Reaches XWayland clients only; a native Wayland window
///   never sees it. This is what KDE falls back to.
#[cfg(target_os = "linux")]
fn connect_keyboard(
    portal: bool,
    restore_token: Option<String>,
) -> Result<(Enigo, &'static str), enigo::NewConError> {
    // A display name no socket will ever have, used to veto a backend.
    let nowhere = || Some("kotha-no-such-display".to_string());

    if portal {
        // Veto both others so libei is the only connection that can open —
        // otherwise a machine where two succeed would paste twice.
        //
        // `restore_token` is the whole reason the dialog is not forever. The
        // portal hands one back after the user says yes; give it back next
        // launch and the portal restores the same grant silently. Without it
        // every launch is a fresh ask, which is what 0.6.1 did.
        return Enigo::new(&Settings {
            x11_display: nowhere(),
            wayland_display: nowhere(),
            restore_token,
            ..Default::default()
        })
        .map(|k| (k, "libei via the desktop portal — reaches every window"));
    }

    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        // The token goes here too, and leaving it out was a bug worth naming:
        // this route asks for a permission it then could not remember.
        //
        // enigo's Wayland backend is not only the virtual-keyboard protocol.
        // Where the compositor does not offer that — KWin does not, which is
        // the measured fact three notes up — it reaches the same libei the
        // portal route does, and the desktop puts up the same "allow remote
        // control?" dialog. `Output::open` then reads a token back off the
        // connection and saves it, because `fresh_token` asks the *connection*
        // and does not care which branch opened it.
        //
        // So the token was written to settings.json on every launch and handed
        // back on none of them, and the dialog came up every single time. The
        // clone is because this attempt may fail and fall through to X11.
        let wayland_only = Settings {
            x11_display: nowhere(),
            restore_token: restore_token.clone(),
            ..Default::default()
        };
        if let Ok(k) = Enigo::new(&wayland_only) {
            return Ok((k, "wayland virtual keyboard — reaches every window"));
        }
    }
    // XTEST does not have permissions to restore, so the token is inert here.
    // Passed anyway rather than dropped: every branch handling it the same way
    // is what stops the next one from forgetting.
    Enigo::new(&Settings {
        wayland_display: nowhere(),
        restore_token,
        ..Default::default()
    })
    .map(|k| {
        (k, "x11/xwayland only — native Wayland windows will NOT receive it; \
             try KOTHA_PASTE=portal")
    })
}

/// The portal's token, after a connection has been made. Rotates on every use.
#[cfg(target_os = "linux")]
fn fresh_token(k: &Enigo) -> Option<String> {
    k.restore_token()
}

#[cfg(not(target_os = "linux"))]
fn fresh_token(_: &Enigo) -> Option<String> {
    None
}

/// macOS has one route and one gate on it.
///
/// The route is Quartz: enigo posts `CGEvent`s, which reach every application
/// including native ones, so there is no equivalent of Linux's three-way choice
/// and the `portal` flag means nothing here.
///
/// The gate is **Accessibility**. Synthetic input is a TCC-protected
/// capability, and an untrusted process may post events all day without one of
/// them arriving — which would be exactly the kind of silent failure this
/// project refuses to ship. enigo checks first: `Enigo::new` calls
/// `AXIsProcessTrustedWithOptions` and returns `NoPermission` rather than
/// handing back a connection that cannot type
/// (enigo 0.6.1, `macos/macos_impl.rs:513`). So the failure is loud, and this
/// function's job is only to make it *useful* — see `permission_hint`.
///
/// `open_prompt_to_get_permissions` is left at its default of `true`, which is
/// deliberate: that is the flag that makes the check raise the system dialog,
/// and the system dialog is how macOS is supposed to ask. It fires only when
/// the permission is actually missing, and only when the user has already
/// chosen a paste mode — which is the second of the two opt-ins described
/// above `paste_mode`.
///
/// **Unverified.** Written 2026-09-03 against enigo's source on a machine that
/// has not built this target.
#[cfg(target_os = "macos")]
fn connect_keyboard(
    _portal: bool,
    _restore_token: Option<String>,
) -> Result<(Enigo, &'static str), enigo::NewConError> {
    Enigo::new(&Settings::default()).map(|k| (k, "quartz — reaches every window"))
}

/// Windows has one route and, unlike macOS, no permission to ask for.
///
/// enigo calls `SendInput`, which reaches every window with one exception that
/// Windows enforces and never reports: a program cannot send input to a window
/// running at a higher integrity level. An editor or terminal started "as
/// administrator" receives nothing, `SendInput` still says it succeeded, and
/// `deliver` then puts the previous clipboard back — so the text is left only
/// in the log. Named in the log line, because nothing else will name it.
///
/// **Unverified.** Written 2026-09-15 on a Mac; no Windows build has run it.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn connect_keyboard(
    _portal: bool,
    _restore_token: Option<String>,
) -> Result<(Enigo, &'static str), enigo::NewConError> {
    Enigo::new(&Settings::default()).map(|k| {
        (k, "SendInput — reaches every window except ones running as administrator")
    })
}

/// The line that turns a refusal into something the user can act on.
///
/// "The application does not have the permission to simulate input" is true and
/// useless: it does not say which permission, where it lives, or what to do
/// afterwards. On macOS all three have specific answers, and the third one is
/// the trap — `Output::open` runs again only when the Text output setting
/// *changes* (`main.rs`, the `chosen != mode` guard, checked at the top of a
/// dictation), so granting Accessibility while Kotha is running changes nothing.
/// Re-picking the same option in the settings window is a no-op, and switching
/// away and back between dictations is too, so the only reliable instruction is
/// a restart. A user who grants the permission, sees no difference, and
/// concludes the app is broken is a user lost to a missing sentence.
#[cfg(target_os = "macos")]
fn permission_hint(e: &enigo::NewConError) -> Option<&'static str> {
    matches!(e, enigo::NewConError::NoPermission).then_some(
        "        macOS calls it Accessibility. If the system did not just ask,          open
        System Settings › Privacy & Security › Accessibility and          switch Kotha on.
        Then quit Kotha from the tray and          open it again — it does not notice
        the permission while it is running.",
    )
}

#[cfg(not(target_os = "macos"))]
fn permission_hint(_: &enigo::NewConError) -> Option<&'static str> {
    None
}

fn paste_chord(kb: &mut Enigo) -> Result<(), enigo::InputError> {
    kb.key(PASTE_MODIFIER, Direction::Press)?;
    let pressed = kb.key(PASTE_KEY, Direction::Click);
    // Release the modifier even if the keystroke failed. A stuck Ctrl would
    // break the user's keyboard until they pressed and released it themselves.
    kb.key(PASTE_MODIFIER, Direction::Release)?;
    pressed
}

/// Offline: a WAV file fed through the loop in microphone-sized blocks.
fn run_file(model_dir: &Path, threads: usize, wav: &str) -> Result<()> {
    let samples = crate::read_wav_16k_mono(Path::new(wav))?;
    println!("input   {wav}  ({:.1}s)", samples.len() as f64 / SAMPLE_RATE as f64);

    let engine = warm_up(model_dir, threads)?;
    let corrector = Corrector::new();
    let mut out = Output::open(paste_mode(), None);
    let mut intake = Intake::new(SAMPLE_RATE, 1)?;
    let mut segmenter = Segmenter::new();
    let mut n = 0;

    // 100 ms at a time, the size a real capture callback delivers, so the
    // buffering and frame-alignment logic is exercised rather than bypassed.
    let block = SAMPLE_RATE as usize / 10;
    for chunk in samples.chunks(block) {
        for utterance in intake.feed(chunk, &mut segmenter)? {
            n += 1;
            emit(&engine, &corrector, &mut out, n, &utterance)?;
        }
    }
    // The file ended without a trailing pause; take whatever is still open.
    if let Some(utterance) = segmenter.close() {
        n += 1;
        emit(&engine, &corrector, &mut out, n, &utterance)?;
    }

    println!("{n} utterance(s) from one model load.");
    Ok(())
}

/// Live: the microphone.
fn run_mic(model_dir: &Path, threads: usize) -> Result<()> {
    let Microphone { stream, blocks: rx, mut intake, .. } = open_microphone()?;

    let engine = warm_up(model_dir, threads)?;
    let corrector = Corrector::new();
    let mut out = Output::open(paste_mode(), None);

    let quit = Arc::new(AtomicBool::new(false));
    {
        let quit = quit.clone();
        std::thread::spawn(move || {
            let mut line = String::new();
            let _ = std::io::stdin().read_line(&mut line);
            quit.store(true, Ordering::Relaxed);
        });
    }

    println!("listening — talk, pause, and it appears. Enter to stop.\n");

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
            emit(&engine, &corrector, &mut out, n, &utterance)?;
        }
    }

    drop(stream);
    println!("stopped after {n} utterance(s), one model load.");
    Ok(())
}

/// How many threads CTranslate2 gets. Override with `KOTHA_THREADS`.
///
/// Physical cores, not logical: on the Ryzen 5600G 6 beat 12, so SMT actively
/// hurt.
///
/// Shared by the CLI and the app deliberately. A decode running at a different
/// thread count depending on which front end started it would make every
/// recorded timing ambiguous.
pub fn decode_threads() -> usize {
    std::env::var("KOTHA_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(default_threads)
}

/// Every core, because on a symmetric machine every core is the same core.
#[cfg(not(target_os = "macos"))]
fn default_threads() -> usize {
    num_cpus::get_physical()
}

/// **Performance cores only**, which on Apple Silicon is not the same number
/// `num_cpus` reports and is the difference this whole module exists for.
///
/// An M2 has 4 performance cores and 4 efficiency cores. `hw.physicalcpu` says
/// 8 and every one of them can run this code, so `num_cpus::get_physical()`
/// hands back 8 — and 8 is the wrong answer, because the eight are not
/// interchangeable. An efficiency core is roughly a third of a performance
/// core.
///
/// That asymmetry costs more than the arithmetic suggests. CTranslate2 splits a
/// layer's work across its threads and then *joins*: the layer is not finished
/// until the slowest thread is. Give it eight equal shares on four fast cores
/// and four slow ones and every fast core finishes its share and then waits,
/// idle, for the slow ones — so eight threads buy roughly what five would, and
/// pay eight threads' worth of synchronisation and memory traffic for it. Four
/// threads on four performance cores is the shape the hardware actually has.
///
/// `hw.perflevel0` is always the fastest level macOS knows about, so this is
/// correct rather than merely Apple-Silicon-specific: on an Intel Mac there is
/// one level, `perflevel0.physicalcpu` equals `hw.physicalcpu`, and the answer
/// is the same one `num_cpus` would have given. Older systems predating the
/// key fall through to `num_cpus`.
///
/// This pairs with `prefer_performance_cores` and is close to useless without
/// it: asking for four threads does not tell macOS *which* four cores to run
/// them on. The thread count says how much parallelism to create; the QoS class
/// says where it is allowed to land. Both, or neither.
///
/// **Unmeasured.** The reasoning is the hardware's, not a benchmark's — this
/// machine has never built the engine. `KOTHA_THREADS` overrides it, and
/// `./setup.sh --bench` sweeps it against every count that makes sense here.
#[cfg(target_os = "macos")]
fn default_threads() -> usize {
    perflevel0_physicalcpu().unwrap_or_else(num_cpus::get_physical)
}

/// `sysctl hw.perflevel0.physicalcpu`, or `None` if the key is not there.
#[cfg(target_os = "macos")]
fn perflevel0_physicalcpu() -> Option<usize> {
    let name = c"hw.perflevel0.physicalcpu";
    let mut out: i32 = 0;
    let mut len = std::mem::size_of::<i32>();

    // Safety: `name` is a NUL-terminated C string, and `out`/`len` are a live
    // i32 and its true size. sysctlbyname writes at most `len` bytes.
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut out as *mut i32).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };

    (rc == 0 && out > 0).then_some(out as usize)
}

/// Ask macOS to schedule this thread on the performance cores.
///
/// There is no core affinity on Apple Silicon — nothing pins a thread to a
/// P-core and Apple does not intend to offer one. What there is instead is
/// **quality of service**, and it is not advisory in the way that word usually
/// implies: the QoS class is the input the scheduler uses to decide which
/// *cluster* a thread is eligible for. A `BACKGROUND` thread runs on the
/// efficiency cores and only there, however idle the machine is. A thread that
/// never declares a class inherits whatever it was given, which for a thread
/// spawned out of a GUI event loop is not something to leave to chance.
///
/// `USER_INITIATED` is the honest description of a dictation decode: the user
/// pressed a key and is sitting there waiting for the words to appear. It is
/// also the right ceiling. `USER_INTERACTIVE` exists for work the next frame
/// depends on — it is the main thread's class — and a multi-second decode is
/// not that; asking for it on a long compute is what Apple's own guidance warns
/// against, and the system may demote it anyway.
///
/// **Call this on the thread that will own the engine, before the engine is
/// built.** That ordering is the whole trick. CTranslate2 and ruy both create
/// their thread pools out of whichever thread constructs and first drives them,
/// and macOS propagates the creating thread's QoS to threads made with
/// `pthread_create` and default attributes. Set it first and the entire pool is
/// born at the right class; set it afterwards and the caller is promoted while
/// the pool that does the actual arithmetic stays wherever it landed. Kotha
/// gets this for free because one worker thread owns the engine for the life of
/// the process, which it does for unrelated reasons.
///
/// Not fatal if it fails, like the window hints: the decode still runs, just
/// possibly on the wrong side of the chip.
#[cfg(target_os = "macos")]
pub fn prefer_performance_cores() {
    // `QOS_CLASS_USER_INITIATED`, from <sys/qos.h>. libc has no binding for
    // any of this, so the constant and the function are both spelled out.
    // Stable since 10.10 and it lives in libSystem, which is always linked.
    const QOS_CLASS_USER_INITIATED: u32 = 0x19;

    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }

    // Safety: no arguments to get wrong. It sets a property of the calling
    // thread and touches nothing else.
    let rc = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) };

    // pthread functions return the error number rather than setting errno.
    if rc == 0 {
        println!("cores   user-initiated QoS — decode belongs on the P-cores");
    } else {
        eprintln!(
            "cores   could not raise this thread's QoS (error {rc}); the decode \
             may be scheduled onto the efficiency cores and run slower"
        );
    }
}

#[cfg(not(target_os = "macos"))]
pub fn prefer_performance_cores() {}

/// A live capture: the stream, the blocks it produces, and the converter that
/// turns them into 16 kHz mono.
///
/// `stream` is a field nobody reads, and it must not be dropped — dropping it
/// stops the capture. It is also not `Send` under ALSA, so a `Microphone` has
/// to be created, used and dropped on one thread.
pub struct Microphone {
    pub stream: cpal::Stream,
    pub blocks: mpsc::Receiver<Vec<f32>>,
    pub intake: Intake,
    /// How many interleaved samples make up 1/30 second at the device's rate.
    ///
    /// The pill's glyph wants a steady 30 levels a second, and a capture block
    /// is whatever size the driver gave us — see BLOCKS_PER_SEC, which asks
    /// for 10 ms and does not always get it. Whatever arrives, it is not
    /// 33 ms, so the app accumulates samples to this size across blocks and
    /// emits one RMS per full window. Slicing per block, which is what this
    /// used to describe, only handles blocks larger than a window; see `meter`
    /// in the app for what the small ones did.
    pub level_chunk: usize,
}

/// Capture blocks per second to ask the driver for.
///
/// A driver picks its own block size unless asked, and what it picks is not
/// chosen with a meter in mind: this Linux box chose 64 ms, which means two
/// whole 1/30 s windows land at once and then nothing arrives for 64 ms. The
/// glyph updates in bursts at 15 Hz, showing audio that ended up to 64 ms ago,
/// and no amount of work in the frontend can recover a level that has not been
/// captured yet.
///
/// 100 blocks a second is 10 ms — small enough that the block stops being the
/// largest term in how late the glyph is, large enough to stay well clear of
/// the xrun territory a 2 ms buffer lives in. It is also roughly what
/// CoreAudio already hands the M2 (10.7 ms), so this changes that machine
/// barely at all.
///
/// Nothing downstream cares what size the blocks are: `Intake` re-blocks to
/// 256-sample VAD frames and `meter` accumulates its own 1/30 s windows, so
/// both were already independent of whatever the driver felt like sending.
const BLOCKS_PER_SEC: u32 = 100;

/// Open the default input device and start capturing.
///
/// Both the CLI loop below and the Tauri app go through here, so there is one
/// answer to "what does Kotha do with a 44.1 kHz stereo microphone" rather
/// than two that can drift apart.
pub fn open_microphone() -> Result<Microphone> {
    // Opened before the model loads: a missing microphone should cost a
    // millisecond, not four seconds and 1.4 GB.
    let device = cpal::default_host()
        .default_input_device()
        .context("no input device — is a microphone connected?")?;
    let supported = pick_config(&device)?;
    let in_rate = supported.sample_rate();
    let channels = supported.channels() as usize;

    println!("input   {device}"); // DeviceTrait: Display gives the name
    println!(
        "format  {in_rate} Hz, {channels} ch, {:?}{}",
        supported.sample_format(),
        if in_rate == SAMPLE_RATE { "  (no resampling needed)" } else { "" }
    );

    let (tx, blocks) = mpsc::channel::<Vec<f32>>();
    let stream = open_stream(&device, &supported, &tx)?;
    // The stream holds the only sender that should stay alive: the receive
    // loop in `meter` ends when every sender is gone, which is how a dictation
    // stops. A clone left behind here would keep it running forever.
    drop(tx);
    let intake = Intake::new(in_rate, channels)?;
    // Interleaved, so a frame is `channels` samples.
    let level_chunk = (in_rate as usize * channels / 30).max(1);

    stream.play().context("could not start the input stream")?;
    Ok(Microphone { stream, blocks, intake, level_chunk })
}

/// Root-mean-square of one captured block, as a 0..1 level.
///
/// This is what the pill's waveform is drawn from. Deliberately unshaped —
/// the curve that makes it look right lives in `app/ui/pill.js`, so tuning the
/// waveform never means recompiling. Interleaved channels are fine here: it is
/// a loudness meter, not a signal.
pub fn rms(block: &[f32]) -> f32 {
    if block.is_empty() {
        return 0.0;
    }
    (block.iter().map(|v| v * v).sum::<f32>() / block.len() as f32).sqrt()
}

/// Pick an input config, preferring one the device can give us at 16 kHz.
///
/// Whisper needs 16 kHz. If the hardware will produce it there is nothing to
/// resample, which is both faster and one less thing to get wrong; most
/// machines will not, so `Intake` still has to handle the general case.
pub fn pick_config(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
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

/// Start the capture stream, asking for a small block and taking the driver's
/// own if it will not give us one.
///
/// A `Fixed` buffer size is a request, not a setting. Devices are free to
/// refuse it, and some refuse it only at the point the stream is built, with
/// an error that says nothing useful — so the fallback is not defensive
/// programming, it is the documented shape of this API. Falling back costs a
/// laggier meter; failing outright costs the whole app.
fn open_stream(
    device: &cpal::Device,
    supported: &cpal::SupportedStreamConfig,
    tx: &mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream> {
    let mut config = supported.config();
    let rate = supported.sample_rate();

    let wanted = match supported.buffer_size() {
        cpal::SupportedBufferSize::Range { min, max } => {
            Some((rate / BLOCKS_PER_SEC).clamp(*min, *max))
        }
        // The device will not say what it supports, so any number is a guess
        // that can fail for no diagnosable reason. Do not guess.
        cpal::SupportedBufferSize::Unknown => None,
    };

    if let Some(frames) = wanted {
        config.buffer_size = cpal::BufferSize::Fixed(frames);
        match build_stream(device, supported.sample_format(), &config, tx.clone()) {
            Ok(stream) => {
                println!(
                    "buffer  {frames} frames ({:.1} ms) requested and granted",
                    frames as f64 / rate as f64 * 1000.0
                );
                return Ok(stream);
            }
            Err(e) => eprintln!(
                "buffer  {frames} frames refused ({e}) — taking the driver's own                  block size, which will make the pill's glyph lag the voice"
            ),
        }
        config.buffer_size = cpal::BufferSize::Default;
    }

    println!("buffer  the driver's own block size");
    build_stream(device, supported.sample_format(), &config, tx.clone())
}

fn build_stream(
    device: &cpal::Device,
    format: cpal::SampleFormat,
    config: &cpal::StreamConfig,
    tx: mpsc::Sender<Vec<f32>>,
) -> Result<cpal::Stream> {
    let on_error = |e| eprintln!("audio stream error: {e}");

    // ponytail: the queue is unbounded. Audio is never dropped, it just arrives
    // late if decoding falls behind, which is the right trade at 1.5x realtime.
    // Bound it if a slower machine ever makes the backlog grow without end.
    Ok(match format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            config.clone(),
            move |data: &[f32], _: &_| {
                let _ = tx.send(data.to_vec());
            },
            on_error,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            config.clone(),
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
pub struct Intake {
    channels: usize,
    resampler: Option<Fft<f32>>,
    at_device_rate: Vec<f32>,
    at_16k: Vec<f32>,
}

impl Intake {
    pub fn new(in_rate: u32, channels: usize) -> Result<Self> {
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
    pub fn feed(&mut self, block: &[f32], seg: &mut Segmenter) -> Result<Vec<Vec<f32>>> {
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
pub struct Segmenter {
    vad: Box<earshot::Detector>,
    preroll: VecDeque<f32>,
    current: Vec<f32>,
    onset_run: usize,
    silence: usize,
    voiced: usize,
    speaking: bool,
}

impl Segmenter {
    pub fn new() -> Self {
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

    /// Is the VAD currently inside an utterance?
    ///
    /// The app stops on its own after a stretch of silence, and "silence"
    /// has to mean the VAD's opinion rather than a level threshold — otherwise
    /// there would be two disagreeing definitions of quiet in one program.
    pub fn is_speaking(&self) -> bool {
        self.speaking
    }

    /// End the utterance in progress and hand it over, if there was one.
    ///
    /// The CLI never needs this — it runs until the stream ends. The app does:
    /// pressing the hotkey to stop is a perfectly ordinary thing to do in the
    /// middle of a sentence, and without this that sentence is thrown away.
    /// The same `MIN_VOICED` floor applies, so a flush of near-silence still
    /// yields nothing.
    pub fn flush(&mut self) -> Option<Vec<f32>> {
        self.close()
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

    /// The clipboard must come back exactly as the user left it, and must NOT
    /// be clobbered if they copied something while we were pasting. This is
    /// the failure that would make the app feel untrustworthy.
    ///
    /// Touches the real system clipboard, so it skips where there is none.
    #[test]
    fn clipboard_is_borrowed_and_returned() {
        let Ok(mut cb) = arboard::Clipboard::new() else {
            eprintln!("no clipboard here; skipping");
            return;
        };
        let theirs = "https://example.com/the-link-the-user-copied";
        let ours = "আমার dictated text";

        // The ordinary case: borrow it, paste, put it back.
        cb.set_text(theirs).unwrap();
        let borrowed = Borrowed::take(&mut cb);
        assert!(matches!(borrowed, Borrowed::Text(ref t) if t == theirs));
        cb.set_text(ours).unwrap();
        borrowed.give_back(&mut cb, ours);
        assert_eq!(cb.get_text().unwrap(), theirs, "the user's clipboard was not restored");

        // The case that matters more: they copied something new while we were
        // mid-paste. Their fresh copy must survive.
        cb.set_text(theirs).unwrap();
        let borrowed = Borrowed::take(&mut cb);
        cb.set_text(ours).unwrap();
        let fresh = "something they copied a moment ago";
        cb.set_text(fresh).unwrap();
        borrowed.give_back(&mut cb, ours);
        assert_eq!(cb.get_text().unwrap(), fresh, "a stale backup overwrote a fresh copy");
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
