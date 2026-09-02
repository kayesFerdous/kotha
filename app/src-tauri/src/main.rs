//! Phase 4, first half — the pill's window, and nothing behind it yet.
//!
//! This is a gate, in the same spirit as Phase 0's. Everything above the
//! window is proven: the engine decodes, the VAD segments, the corrector
//! repairs, the clipboard delivers. What is *not* proven is whether a
//! compositor will give us the window this app needs —
//!
//!   * frameless, transparent and always on top,
//!   * that never takes focus, so the caret keeps blinking where the user
//!     left it,
//!   * placed at the bottom centre of the screen,
//!   * click-through, so it is furniture rather than an obstacle,
//!   * with a global hotkey and a tray icon.
//!
//! On KDE Wayland, at least two of those are open questions — `set_position`
//! is widely a no-op there because the compositor owns placement, and global
//! shortcuts go through X11 in the underlying crate. Finding out costs a
//! two-minute build here and a ten-minute one once the engine is attached, so
//! it happens here first.
//!
//! The chain behind it is the real one. `kotha_spike::live` supplies the
//! microphone, the VAD, the segmenter and the clipboard, and `Engine` and
//! `Corrector` do the rest — the same code the Phase 0 gate measured and the
//! Phase 2 loop ran, driven from here instead of from a terminal.
//!
//! Run it:
//!
//! ```bash
//! cargo run -p kotha --release
//! ```
//!
//! Then press Ctrl+Alt+Space, or use the tray icon.
//!
//! Environment:
//!
//! ```text
//! KOTHA_MODEL    the CTranslate2 model directory. Unset, the app uses
//!                ./models/whisper-medium-bn-en-cs-faster if that exists and
//!                otherwise downloads to the app data directory.
//! KOTHA_THREADS  decode threads (default: physical cores).
//! KOTHA_PASTE    overrides the tray's Text output setting for one run:
//!                unset = use the setting, 1 = synthetic paste, portal = libei,
//!                anything else = clipboard only.
//! ```

use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Context};

use kotha_spike::correct::Corrector;
use kotha_spike::live::{self, Microphone, Output, Segmenter};
use kotha_spike::{suspicious_fusion, Engine};

use tauri::menu::{CheckMenuItem, IsMenuItem, Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, LogicalSize, Manager, PhysicalPosition, WebviewWindow};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

/// Where the model lives.
///
/// Three answers, in order, because three different people are asking:
///
///   1. `KOTHA_MODEL` — someone testing a different checkpoint.
///   2. `./models/...` in the working directory — a developer running from the
///      repo, where that directory is symlinks into the paper-project tree.
///      Only taken if it actually has a `model.bin` in it, so a stale empty
///      directory does not shadow a real download.
///   3. The app data directory — everybody else, and where `fetch_model`
///      downloads to. A shipped app is launched from a menu with the working
///      directory set to `/` or `$HOME`, so a relative path is not an answer
///      for anyone but case 2.
fn model_dir(app: &AppHandle) -> PathBuf {
    if let Some(p) = std::env::var_os("KOTHA_MODEL") {
        return PathBuf::from(p);
    }
    let dev = PathBuf::from("models").join(MODEL_NAME);
    if dev.join("model.bin").is_file() {
        return dev;
    }
    app.path()
        .app_data_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(MODEL_NAME)
}

const MODEL_NAME: &str = "whisper-medium-bn-en-cs-faster";
const HF_REPO: &str = "kayees/whisper-medium-bn-en-cs-faster";

/// Every file `ct2rs::Whisper::new()` wants. Do not prune this list — it loads
/// the published directory as-is, which is the whole reason this app runs the
/// exact int8 weights that were benchmarked (CLAUDE.md §3).
const MODEL_FILES: [&str; 5] = [
    "model.bin",
    "tokenizer.json",
    "vocabulary.json",
    "config.json",
    "preprocessor_config.json",
];

/// Download the model if it is not already there. 778 MB, resumable, verified.
///
/// This is `setup.sh`'s logic moved into the binary, because a shipped app has
/// no shell script next to it. It keeps that script's two hard-won details:
///
///   * **Resume, and retry around a dropped connection.** HuggingFace resets
///     long transfers often enough that a single request is not reliable for a
///     775 MB file — one was lost at 226 MB during Phase 0. Each attempt asks
///     for a byte range starting from whatever is already on disk, so a failure
///     costs the remainder and not the whole thing.
///   * **Verify against HuggingFace's own manifest, not a hard-coded number.**
///     A truncated or badly-resumed `model.bin` fails much later and very
///     confusingly, and eyeballing the size does not catch a corrupt resume.
///     Reading the expected size and hash from the API also means re-uploading
///     the model does not require editing a constant here.
///
/// The manifest is fetched first and drives everything: a local file is
/// complete when its length matches, partial when it is shorter, and junk when
/// it is longer. That is a better completeness test than `curl -C -` had, which
/// could only ask the server and hope.
///
/// `progress` is called with (bytes done, bytes total) for the whole set, often
/// enough to drive a bar and rarely enough not to be the bottleneck.
fn fetch_model(dir: &Path, mut progress: impl FnMut(u64, u64)) -> anyhow::Result<()> {
    // Three separate deadlines rather than one, which is the reason ureq is a
    // better fit here than a total-timeout client: connecting and getting
    // headers back should be quick, and the 775 MB body must not be on a clock
    // at all. A slow link is not an error.
    let http: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(20)))
        .timeout_recv_response(Some(Duration::from_secs(30)))
        .timeout_global(None)
        .build()
        .into();

    let manifest: serde_json::Value = http
        .get(format!("https://huggingface.co/api/models/{HF_REPO}/tree/main"))
        .call()
        .context("could not reach HuggingFace for the model manifest")?
        .body_mut()
        .read_json()
        .context("could not read the model manifest from HuggingFace")?;

    // name -> (size, sha256). Small files are not LFS objects and have no hash,
    // so for those the size is the whole check — which is fine, because what
    // they are exposed to is truncation, not a corrupt resume.
    let mut want: Vec<(&str, u64, Option<String>)> = Vec::new();
    for name in MODEL_FILES {
        let entry = manifest
            .as_array()
            .and_then(|es| es.iter().find(|e| e["path"] == name))
            .ok_or_else(|| anyhow!("{name} is not in the {HF_REPO} manifest"))?;
        let size = entry["size"]
            .as_u64()
            .ok_or_else(|| anyhow!("{name} has no size in the manifest"))?;
        want.push((name, size, entry["lfs"]["oid"].as_str().map(str::to_owned)));
    }

    let total: u64 = want.iter().map(|(_, s, _)| s).sum();
    if want.iter().all(|(n, s, _)| on_disk(&dir.join(n)) == *s) {
        progress(total, total);
        return Ok(());
    }

    std::fs::create_dir_all(dir)?;
    let already: u64 = want.iter().map(|(n, s, _)| on_disk(&dir.join(n)).min(*s)).sum();
    progress(already, total);
    println!(
        "model   {} of {} MB to fetch into {}",
        (total - already) >> 20,
        total >> 20,
        dir.display()
    );

    // Bytes in files already finished. Progress is `finished + <this file so
    // far>`, computed rather than accumulated, so a restart that throws away a
    // partial file cannot leave a running total quietly wrong.
    let mut finished: u64 = 0;

    for (name, size, sha) in &want {
        let path = dir.join(name);

        for attempt in 1..=6 {
            let have = on_disk(&path);
            if have == *size {
                break;
            }
            if have > *size {
                // Longer than the manifest says, so this is not a partial
                // download — it is a different or damaged file. Start over
                // rather than resuming onto garbage.
                std::fs::remove_file(&path)?;
            }
            match stream_to(&http, name, &path, on_disk(&path), &mut |sofar| {
                progress(finished + sofar, total)
            }) {
                Ok(()) => break,
                Err(e) if attempt < 6 => {
                    // Resume rather than restart: HuggingFace resets long
                    // transfers often enough that a single request is not
                    // reliable for a 775 MB file — Phase 0 lost one at 226 MB.
                    // A failure then costs the remainder, not the whole thing.
                    eprintln!("model   attempt {attempt} failed ({e:#}) — resuming");
                    thread::sleep(Duration::from_secs(3));
                }
                Err(e) => return Err(e.context(format!("gave up on {name} after 6 attempts"))),
            }
        }

        let got = on_disk(&path);
        if got != *size {
            bail!("{name} is {got} bytes, expected {size} — the download is incomplete");
        }
        if let Some(sha) = sha {
            println!("model   hashing {name}, this takes a few seconds");
            let actual = sha256(&path)?;
            if &actual != sha {
                // A resume landed on the wrong offset. Nothing here can repair
                // that, and leaving it would make the next run skip it on a
                // matching size — so it goes.
                std::fs::remove_file(&path)?;
                bail!(
                    "{name} sha256 does not match HuggingFace, so it has been deleted.\n    \
                     expected {sha}\n    actual   {actual}\n    \
                     Start a dictation again to re-fetch it."
                );
            }
        }
        finished += size;
    }

    // Say so explicitly rather than relying on the last file's last chunk to
    // land on the total: the files at the end of the list are often already
    // complete, in which case they report nothing at all and the caller is left
    // watching a bar stopped just short of full.
    progress(total, total);
    println!("model   ready in {}", dir.display());
    Ok(())
}

/// Is there a model to load?
///
/// ponytail: "every file is there and not empty", which a download killed
/// halfway would also satisfy. The checksum in `fetch_model` protects the
/// normal path; a machine that lost power mid-download gets a load failure
/// naming the directory to delete. A marker file would close it properly if
/// anyone ever hits it.
fn model_ready(dir: &Path) -> bool {
    MODEL_FILES.iter().all(|f| on_disk(&dir.join(f)) > 0)
}

/// Run the download, reporting to the first-run window as it goes.
///
/// Errors are sent to the window rather than returned: the user is looking
/// straight at it, and a failure they can read and retry is worth more than a
/// line in a terminal they never opened.
fn download(app: &AppHandle) {
    let dir = model_dir(app);
    let mut last_ui = 0u64;
    let mut last_log = 0u64;

    let result = fetch_model(&dir, |done, total| {
        // Two rates: a megabyte for the bar, sixteen for the terminal. The
        // terminal is a log and the bar is an animation, and they want very
        // different amounts of noise.
        if done == total || done.saturating_sub(last_ui) >= 1 << 20 {
            last_ui = done;
            let _ = app.emit("kotha://download", serde_json::json!({ "done": done, "total": total }));
        }
        if done == total || done.saturating_sub(last_log) >= 16 << 20 {
            last_log = done;
            println!("model   {} / {} MB", done >> 20, total >> 20);
        }
    });

    if let Err(e) = result {
        eprintln!("model   download failed: {e:#}");
        let _ = app.emit("kotha://download", serde_json::json!({ "error": format!("{e:#}") }));
    }
}

/// Bring up the first-run window, focused — the one window in this app that
/// is *supposed* to take focus, because it has a button on it.
fn show_setup(app: &AppHandle) {
    let Some(w) = app.get_webview_window("setup") else {
        eprintln!("no window labelled `setup` — check tauri.conf.json");
        return;
    };
    let _ = w.show();
    let _ = w.set_focus();
}

/// The only thing the UI is allowed to ask Rust to do.
///
/// The pill's contract is one-way by design (see the header of `pill.js`) and
/// stays that way. The first-run window is the exception, and it is the exception
/// for one reason: 778 MB should not leave without somebody pressing a button.
#[tauri::command]
fn start_download(app: AppHandle) {
    let session = app.state::<Session>();
    if session.tx.lock().map(|tx| tx.send(Cmd::Fetch).is_ok()).unwrap_or(false) {
        return;
    }
    eprintln!("worker is gone — the download cannot start");
    let _ = app.emit(
        "kotha://download",
        serde_json::json!({ "error": "Kotha's worker thread is not running. Restart the app." }),
    );
}

/// Which key the "Ready" panel should tell the user to press, or `None` if
/// nothing is bound.
///
/// Computed rather than stored: the hotkey can be changed from the tray after
/// this window is already open, and registration can fail — in which case the
/// panel must not name a key that does nothing. That is the whole reason this
/// is not a string in the HTML any more.
#[tauri::command]
fn hotkey_label(app: AppHandle) -> Option<String> {
    let k = hotkey(&app);
    app.global_shortcut().is_registered(k.as_str()).then_some(k)
}

/// Bytes already on disk, or 0 — a missing file and an empty one are the same
/// thing to a resume.
fn on_disk(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

/// One attempt at one file, appending from byte `from`.
///
/// `on_bytes` is called with how many bytes of *this file* are on disk, not a
/// delta — so a restart reports a smaller number and the caller's total goes
/// backwards honestly, instead of finishing at 110%.
fn stream_to(
    http: &ureq::Agent,
    name: &str,
    path: &Path,
    from: u64,
    on_bytes: &mut impl FnMut(u64),
) -> anyhow::Result<()> {
    let mut req = http.get(format!("https://huggingface.co/{HF_REPO}/resolve/main/{name}"));
    if from > 0 {
        req = req.header("Range", format!("bytes={from}-"));
    }
    let mut res = req.call()?;

    // A server that ignores the range header answers 200 with the whole file.
    // Appending that onto what we already have would produce a file of the
    // right length made of the wrong bytes — which only the hash would catch,
    // and only after 775 MB. So truncate and take it from the top instead.
    let resuming = from > 0 && res.status().as_u16() == 206;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(resuming)
        .truncate(!resuming)
        .open(path)?;

    let mut sofar = if resuming { from } else { 0 };
    on_bytes(sofar);

    let mut body = res.body_mut().as_reader();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = std::io::Read::read(&mut body, &mut buf)?;
        if n == 0 {
            break;
        }
        std::io::Write::write_all(&mut file, &buf[..n])?;
        sofar += n as u64;
        on_bytes(sofar);
    }
    std::io::Write::flush(&mut file)?;
    Ok(())
}

fn sha256(path: &Path) -> anyhow::Result<String> {
    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut file, &mut hasher)?;
    Ok(format!("{:x}", hasher.finalize()))
}

/// The pill's window, in logical pixels.
///
/// Rust owns this rather than tauri.conf.json, because placement has to divide
/// by the window's width to centre it and `outer_size()` reads 0×0 until the
/// compositor has mapped the window — which is after the first `show()`, by
/// which time the pill has already appeared in the wrong place.
///
/// It has to stay comfortably larger than the capsule in pill.css: the pill is
/// about 187×44 today, and its shadow spreads 40 px. Bigger than that is only
/// more transparent surface for the compositor to blend every frame.
const PILL_WINDOW: (f64, f64) = (320.0, 120.0);

/// Gap between the bottom of the screen and the bottom of the window, in
/// logical pixels. Override with `KOTHA_BOTTOM`.
///
/// This is a calibration knob, not a layout constant. Where a window actually
/// lands is a negotiation with the window manager, and 72 is what looked right
/// on KDE Plasma 6 through XWayland — measured, not assumed: the pill's bottom
/// edge came out 75 px above the screen. Expect to re-measure per desktop, and
/// on the M2 in particular, where the Dock is in the way.
const BOTTOM_MARGIN: f64 = 72.0;

fn bottom_margin() -> f64 {
    std::env::var("KOTHA_BOTTOM")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(BOTTOM_MARGIN)
}

/// How long after `done` the window is hidden.
///
/// The UI owns the visible timing — see DONE_DWELL in pill.js, currently
/// 1100 ms, plus the fade-out. This only has to be comfortably longer, since
/// hiding a window nothing can see costs nothing but a few frames of
/// compositing. If the pill ever vanishes mid-tick, this is why.
const HIDE_AFTER: Duration = Duration::from_millis(1900);

/// How often the worker checks whether the user has asked it to stop.
const POLL: Duration = Duration::from_millis(200);

/// Stop dictating after this much silence.
///
/// PLAN.md's description of the app has always said "press again, or stay
/// silent for two seconds, and it fades out". The first build only had the
/// first half, which meant that if the hotkey was missed — and on Wayland it
/// can be — there was no way to end a dictation at all.
///
/// Measured from the VAD's opinion, not a level threshold, so there is one
/// definition of quiet in the program rather than two. It has to be comfortably
/// longer than the segmenter's own ~600 ms hangover, or it would fire between
/// sentences.
/// Override in seconds with `KOTHA_IDLE`; 0 disables it.
const IDLE_STOP: Duration = Duration::from_secs(3);

fn idle_stop() -> Option<Duration> {
    let secs = std::env::var("KOTHA_IDLE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(IDLE_STOP.as_secs());
    (secs > 0).then(|| Duration::from_secs(secs))
}

/// What the hotkey and the tray ask the worker to do.
enum Cmd {
    Start,
    Stop,
    /// Download the model. Sent by the first-run window's button, and by
    /// nothing else — the 778 MB is never spent without being asked for.
    Fetch,
}

/// The app's state: whether a dictation is running, and how to reach the
/// worker that runs it.
struct Session {
    listening: AtomicBool,
    /// `mpsc::Sender` is `Send` but not `Sync`, and Tauri state must be both.
    tx: Mutex<mpsc::Sender<Cmd>>,
}

fn main() {
    prefer_x11();

    let (tx, rx) = mpsc::channel::<Cmd>();

    tauri::Builder::default()
        .manage(Session { listening: AtomicBool::new(false), tx: Mutex::new(tx) })
        .invoke_handler(tauri::generate_handler![start_download, hotkey_label])
        .setup(move |app| {
            let pill = app
                .get_webview_window("pill")
                .expect("no window labelled `pill` — check tauri.conf.json");

            // The one thing this app cannot get wrong. A dictation pill that
            // steals focus takes the caret with it and the feature collapses.
            pill.set_focusable(false)?;
            pill.set_size(LogicalSize::new(PILL_WINDOW.0, PILL_WINDOW.1))?;

            // NOTE: click-through is deliberately NOT set here. See the call
            // in `toggle`, and the bug note above it.

            tray(app.handle())?;

            app.handle().plugin(
                tauri_plugin_global_shortcut::Builder::new()
                    .with_handler(|app, _, event| {
                        // Exactly one shortcut is ever registered — the tray
                        // unregisters before it binds another — so whatever
                        // arrives here is it.
                        if event.state() == ShortcutState::Pressed {
                            toggle(app);
                        }
                    })
                    .build(),
            )?;

            // Not fatal. On Wayland this is the call most likely to fail, and
            // the tray icon still works when it does — so say so and carry on
            // rather than refusing to start.
            let chosen = hotkey(app.handle());
            match app.global_shortcut().register(chosen.as_str()) {
                Ok(()) => println!("hotkey  {chosen}"),
                Err(e) => eprintln!(
                    "hotkey  unavailable ({e}) — use the tray icon.\n\
                             On Wayland the underlying crate binds through X11, \
                     which the compositor may not expose."
                ),
            }

            // A first run has no model, and 778 MB is not something to start
            // behind the user's back on a keystroke they pressed expecting to
            // dictate. So the window comes up at launch, says what it needs,
            // and waits to be told.
            if !model_ready(&model_dir(app.handle())) {
                println!("model   missing — first run");
                show_setup(app.handle());
            }

            // One long-lived worker owns the engine, so the model is loaded
            // once and stays warm across dictations — which is exactly what
            // Phase 2's acceptance test was about. It also means the engine
            // never crosses a thread boundary after it is built.
            let handle = app.handle().clone();
            thread::spawn(move || worker(handle, rx));

            // KOTHA_DEMO=1 records a fixed-length dictation on its own, a
            // second after startup. The window questions this gate was built
            // to answer need the pill actually on screen, and the hotkey may
            // be exactly the thing that is broken — so it must not depend on
            // the hotkey.
            if std::env::var("KOTHA_DEMO").is_ok() {
                let app = app.handle().clone();
                thread::spawn(move || {
                    thread::sleep(Duration::from_millis(900));
                    toggle(&app);
                    // Read the position again once the compositor has had time
                    // to answer. Reading it in the same breath as the request
                    // cannot tell a refusal from an asynchronous grant.
                    thread::sleep(Duration::from_millis(1200));
                    if let Some(w) = app.get_webview_window("pill") {
                        println!("window  a moment later, at {:?}", w.outer_position());
                    }
                    thread::sleep(Duration::from_secs(6));
                    // It may already have stopped itself on silence, in which
                    // case this would start a second one. Only stop what is
                    // still running.
                    if app.state::<Session>().listening.load(Ordering::SeqCst) {
                        toggle(&app);
                    }
                    thread::sleep(HIDE_AFTER + Duration::from_millis(600));
                    app.exit(0);
                });
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("Kotha failed to start");
}

/// Ask GTK for the X11 backend on Linux, before it initialises.
///
/// Measured on KDE Plasma 6 / kwin_wayland, 2026-08-31: under native Wayland
/// `set_position` is simply ignored — the window stays at (0, 0) a full second
/// after the request — because the compositor owns placement and there is no
/// protocol for an ordinary client to ask. Through XWayland the same call puts
/// the window exactly where it belongs.
///
/// A dictation pill in the top-left corner of the screen is not a pill, so
/// XWayland is the better trade on Linux today. It costs sharpness under
/// fractional scaling and nothing else — in particular it does **not** affect
/// how text is delivered, because Phase 3 settled on libei through the desktop
/// portal, which goes over D-Bus and does not care which display server this
/// process talks to.
///
/// Override with `KOTHA_GDK_BACKEND=wayland` to see the native behaviour. It
/// has to be Kotha's own variable rather than deferring to an existing
/// `GDK_BACKEND`: a KDE session exports `GDK_BACKEND=wayland` to every GTK
/// application, so "leave it alone if the user set it" would mean never firing
/// at all, on the exact desktop that needs it.
//
// ponytail: an env var instead of the real fix. The proper Wayland answer is
// gtk-layer-shell (`zwlr_layer_shell_v1`, which KWin supports): an overlay
// layer with anchors, which would place the pill natively *and* make it
// structurally incapable of taking keyboard focus. It has to be initialised
// before the GtkWindow is realised, which is inside Tauri's builder, so it is
// real work. Do it when Linux is a first-class target rather than the
// development machine.
#[cfg(target_os = "linux")]
fn prefer_x11() {
    let backend = std::env::var("KOTHA_GDK_BACKEND").unwrap_or_else(|_| "x11".into());
    println!("gdk     backend {backend}");
    std::env::set_var("GDK_BACKEND", backend);
}

#[cfg(not(target_os = "linux"))]
fn prefer_x11() {}

/// Start or stop a dictation. The hotkey and the tray both land here.
///
/// This does no work itself: it flips the flag the worker is watching and
/// pokes the channel. Everything slow — opening the microphone, loading the
/// model, decoding — happens on the worker, because this runs on the UI thread
/// and a frozen pill is worse than no pill.
fn toggle(app: &AppHandle) {
    let session = app.state::<Session>();
    let was_listening = session.listening.swap(true, Ordering::SeqCst);
    if was_listening {
        session.listening.store(false, Ordering::SeqCst);
    }
    let cmd = if was_listening { Cmd::Stop } else { Cmd::Start };
    println!("toggle  {}", if was_listening { "stop" } else { "start" });
    let sent = session.tx.lock().map(|tx| tx.send(cmd).is_ok()).unwrap_or(false);
    if !sent {
        eprintln!("worker is gone — dictation is not available");
    }
}

/// The worker thread: one engine, loaded once, for the life of the process.
fn worker(app: AppHandle, rx: mpsc::Receiver<Cmd>) {
    let threads = live::decode_threads();

    let corrector = Corrector::new();
    let mut mode = paste_setting(&app);
    let mut out = Output::open(paste_flags(&mode));
    let mut engine: Option<Engine> = None;

    while let Ok(cmd) = rx.recv() {
        // A Stop with nothing running is ordinary — the dictation may have
        // already ended on its own. Ignore it rather than treating it as an
        // error.
        match cmd {
            Cmd::Stop => continue,
            Cmd::Fetch => {
                download(&app);
                continue;
            }
            Cmd::Start => {}
        }

        // The hotkey does not spend 778 MB. If the model is not there, the
        // first-run window comes up and asks — which is the whole point of it
        // existing, and the reason `dictate` no longer downloads anything.
        if !model_ready(&model_dir(&app)) {
            println!("model   not downloaded yet — opening the first-run window");
            show_setup(&app);
            app.state::<Session>().listening.store(false, Ordering::SeqCst);
            continue;
        }

        // The tray can change this between dictations. Reopening costs a
        // clipboard handle and, on the portal route, the permission dialog — so
        // only when the choice actually changed, and never mid-dictation.
        let chosen = paste_setting(&app);
        if chosen != mode {
            mode = chosen;
            out = Output::open(paste_flags(&mode));
        }

        if let Err(e) = dictate(&app, &rx, &mut engine, &corrector, &mut out, threads) {
            // Never leave the pill up on a failure: the user pressed a key and
            // deserves to be told, not to be left looking at a frozen pill.
            eprintln!("dictation failed: {e:#}");
            app.state::<Session>().listening.store(false, Ordering::SeqCst);
            let _ = app.emit("kotha://state", "idle");
            hide_soon(&app, Duration::ZERO);
        }
    }
}

/// One dictation, start to finish, on the worker thread.
fn dictate(
    app: &AppHandle,
    rx: &mpsc::Receiver<Cmd>,
    engine: &mut Option<Engine>,
    corrector: &Corrector,
    out: &mut Output,
    threads: usize,
) -> anyhow::Result<()> {
    // The microphone first: a missing one should cost a millisecond, not four
    // seconds and 1.4 GB. `Microphone` is not Send under ALSA, which is the
    // other reason all of this lives on one thread.
    let Microphone { stream, blocks, mut intake, level_chunk } = live::open_microphone()?;
    show(app);
    let _ = app.emit("kotha://state", "listening");

    // Loaded on first use rather than at startup: a tray app that has not
    // dictated yet has no business holding 1.4 GB resident.
    if engine.is_none() {
        let dir = model_dir(app);
        let t = std::time::Instant::now();
        *engine = Some(Engine::load(&dir, threads).map_err(|e| {
            e.context(format!(
                "could not load the model from {}. If a download was interrupted \
                 the files there may be incomplete — delete the directory and \
                 start Kotha again.",
                dir.display()
            ))
        })?);
        println!("model   {} threads, loaded in {:.1}s", threads, t.elapsed().as_secs_f64());
    }
    let engine = engine.as_ref().expect("just loaded");

    let mut segmenter = Segmenter::new();
    let mut n = 0usize;
    let mut last_voice = std::time::Instant::now();
    let mut logged_block = false;
    let idle = idle_stop();

    let stopped_by = loop {
        match rx.try_recv() {
            Ok(Cmd::Stop) => break "hotkey",
            Err(mpsc::TryRecvError::Disconnected) => break "shutdown",
            // A Fetch arriving mid-dictation means the model is already
            // there and somebody pressed the button anyway. Ignoring it beats
            // interrupting a live dictation to re-verify 778 MB.
            Ok(Cmd::Start) | Ok(Cmd::Fetch) | Err(mpsc::TryRecvError::Empty) => {}
        }

        let block = match blocks.recv_timeout(POLL) {
            Ok(b) => b,
            // A silent room produces blocks too, so a timeout means the device
            // stopped rather than that nobody is speaking.
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break "device gone",
        };

        if !logged_block {
            println!(
                "audio   {} samples per block ({:.0} ms), {} levels per block",
                block.len(),
                block.len() as f64 / kotha_spike::SAMPLE_RATE as f64 * 1000.0,
                block.len().div_ceil(level_chunk).max(1)
            );
            logged_block = true;
        }

        // One level per ~33 ms rather than one per capture block, so the
        // waveform moves at the same speed whatever buffer size the driver
        // chose. See `Microphone::level_chunk`.
        for piece in block.chunks(level_chunk) {
            let _ = app.emit("kotha://level", live::rms(piece));
        }

        // The VAD closes a chunk at every pause, so text lands while the user
        // is still talking — which is the whole reason for segmenting at all
        // at 1.5x realtime.
        for utterance in intake.feed(&block, &mut segmenter)? {
            n += 1;
            deliver(app, engine, corrector, out, n, &utterance)?;
            let _ = app.emit("kotha://state", "listening");
            last_voice = std::time::Instant::now();
        }

        if segmenter.is_speaking() {
            last_voice = std::time::Instant::now();
        } else if idle.is_some_and(|d| last_voice.elapsed() > d) {
            break "silence";
        }
    };

    // Stopping mid-sentence is an ordinary thing to do. Whatever is buffered
    // gets transcribed rather than thrown away.
    drop(stream);
    if let Some(tail) = segmenter.flush() {
        n += 1;
        deliver(app, engine, corrector, out, n, &tail)?;
    }

    app.state::<Session>().listening.store(false, Ordering::SeqCst);
    let _ = app.emit("kotha://state", if n > 0 { "done" } else { "idle" });
    hide_soon(app, if n > 0 { HIDE_AFTER } else { Duration::from_millis(400) });
    println!("stopped on {stopped_by} after {n} utterance(s)\n");
    Ok(())
}

/// Transcribe one utterance, repair its English, and put it where the user
/// asked for it.
fn deliver(
    app: &AppHandle,
    engine: &Engine,
    corrector: &Corrector,
    out: &mut Output,
    n: usize,
    utterance: &[f32],
) -> anyhow::Result<()> {
    let secs = utterance.len() as f64 / kotha_spike::SAMPLE_RATE as f64;
    let _ = app.emit("kotha://state", "thinking");

    let t = std::time::Instant::now();
    let text = engine.transcribe(utterance)?;
    let took = t.elapsed().as_secs_f64();
    let text = text.trim();

    println!("[{n}] {secs:.1}s audio → {took:.1}s decode ({:.2}x realtime)", secs / took);
    println!("    {text}");
    if let Some(w) = suspicious_fusion(text) {
        eprintln!("    ⚠ {w} — check suppress_tokens");
    }

    let fixed = corrector.correct_text(text);
    if fixed != text {
        println!("  → {fixed}");
    }
    out.deliver(&fixed);
    println!();
    Ok(())
}

/// Put the pill on screen, wherever the user's screen currently is.
///
/// Placement is re-applied on every show: monitors come and go, and the pill
/// should appear on the one being looked at.
fn show(app: &AppHandle) {
    let Some(w) = app.get_webview_window("pill") else {
        return;
    };
    let _ = place(&w);
    let _ = w.show();

    // Click-through, so the pill is furniture and not an obstacle.
    //
    // This has to happen *after* the first show, not in setup(). tao 0.35.3
    // handles the request with `window.window().unwrap()`
    // (linux/event_loop.rs:457) — the GDK window, which does not exist until
    // GTK realises the widget. A window created `visible: false` has not been
    // realised, so asking in setup() aborts the process from inside the GTK
    // main loop, where it cannot even unwind. Silent until it is fatal, and
    // worth reporting upstream: the code already has the Option in hand.
    let _ = w.set_ignore_cursor_events(true);

    // The property the whole feature rests on. Self-reported by the toolkit,
    // so it is evidence rather than proof, but a `true` here would be
    // conclusive the other way.
    println!("window  shown, focused = {:?}", w.is_focused());
}

/// Hide the pill once the UI has finished saying goodbye.
///
/// Spawned rather than slept inline, so the worker is free to accept the next
/// dictation immediately — pressing the hotkey again during the fade-out
/// should start recording, not queue behind an animation.
fn hide_soon(app: &AppHandle, after: Duration) {
    let app = app.clone();
    thread::spawn(move || {
        thread::sleep(after);
        // Someone may have started dictating again while we waited. If so the
        // pill is theirs now, and hiding it would be wrong.
        if app.state::<Session>().listening.load(Ordering::SeqCst) {
            return;
        }
        if let Some(w) = app.get_webview_window("pill") {
            let _ = w.hide();
        }
    });
}

/// Bottom centre of whichever monitor the window is currently on.
fn place(w: &WebviewWindow) -> tauri::Result<()> {
    let Some(monitor) = w.current_monitor()? else {
        return Ok(());
    };
    let screen = monitor.size();
    let origin = monitor.position();
    let scale = monitor.scale_factor();

    // PILL_WINDOW, not outer_size(): see the constant. Asking the window how
    // big it is before the compositor has mapped it gives 0×0, and centring
    // against that puts the pill at the middle of the screen rather than the
    // middle *minus half a window* — which is exactly the symptom this had.
    let win_w = (PILL_WINDOW.0 * scale) as i32;
    let win_h = (PILL_WINDOW.1 * scale) as i32;

    w.set_position(PhysicalPosition::new(
        origin.x + (screen.width as i32 - win_w) / 2,
        origin.y + screen.height as i32 - win_h - (bottom_margin() * scale) as i32,
    ))
}

/// The three ways text can leave Kotha, as they appear in the tray menu.
///
/// The id is what lands in `settings.json`, so a hand-edited file and a menu
/// click cannot mean different things.
const PASTE_MODES: [(&str, &str); 3] = [
    ("copy", "Clipboard only"),
    ("paste", "Paste at the cursor"),
    ("portal", "Paste at the cursor (portal)"),
];

/// One file, one JSON object. The hotkey and the microphone add keys here.
fn settings_path(app: &AppHandle) -> PathBuf {
    app.path().app_config_dir().unwrap_or_else(|_| PathBuf::from(".")).join("settings.json")
}

/// `KOTHA_PASTE`, if it is set, as a mode id.
///
/// It still wins over the setting: it is the documented way to test the three
/// routes, and every note in PLAN.md Phase 3 is written in terms of it. When it
/// is set the menu shows what it forced and refuses to be clicked, rather than
/// offering a choice that would not take effect.
fn paste_env() -> Option<&'static str> {
    match std::env::var("KOTHA_PASTE").ok()?.as_str() {
        "portal" => Some("portal"),
        "1" | "true" => Some("paste"),
        _ => Some("copy"),
    }
}

/// The chosen mode: the environment, then the file, then clipboard-only.
fn paste_setting(app: &AppHandle) -> String {
    paste_env().map(str::to_string).unwrap_or_else(|| paste_choice(&settings_path(app)))
}

fn settings_at(path: &Path) -> Option<serde_json::Value> {
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    v.is_object().then_some(v)
}

/// One string setting, or `None` if the file is missing, corrupt, or silent
/// about this key. Every caller supplies its own default and its own idea of
/// what is valid, because those differ and the storage does not.
fn setting(path: &Path, key: &str) -> Option<String> {
    settings_at(path)?[key].as_str().map(str::to_string)
}

fn save_setting(path: &Path, key: &str, value: &str) {
    let mut v = settings_at(path).unwrap_or_else(|| serde_json::json!({}));
    v[key] = serde_json::Value::String(value.to_string());
    if let Err(e) = std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))
        .and_then(|()| std::fs::write(path, v.to_string()))
    {
        eprintln!("settings not saved to {}: {e}", path.display());
    }
}

/// An unreadable, corrupt or unknown value means clipboard-only rather than a
/// mode `Output::open` has never heard of.
fn paste_choice(path: &Path) -> String {
    setting(path, "paste")
        .filter(|m| PASTE_MODES.iter().any(|(id, _)| id == m))
        .unwrap_or_else(|| "copy".into())
}

/// The hotkeys the tray offers, in Tauri's accelerator syntax.
///
/// A fixed list and not a key-capture widget: capturing a chord needs a focused
/// window and a page to draw it on, and what this actually has to solve is a
/// *collision*, not a preference. The default is the collision — `Ctrl+Alt+Space`
/// is fcitx's and ibus's input-method switch, which on a Bangladeshi desktop is
/// very likely already bound to Avro.
///
/// The list is what the menu offers, not what is accepted. `hotkey` validates by
/// parsing, so anything Tauri understands can be written into `settings.json` by
/// hand and the menu will show it alongside these.
const HOTKEYS: [&str; 4] = ["Ctrl+Alt+Space", "Ctrl+Shift+Space", "Alt+Shift+D", "F9"];

/// The chosen hotkey. Junk in the file falls back to the default rather than
/// leaving the app with nothing bound.
fn hotkey(app: &AppHandle) -> String {
    hotkey_choice(&settings_path(app))
}

fn hotkey_choice(path: &Path) -> String {
    setting(path, "hotkey")
        .filter(|h| Shortcut::from_str(h).is_ok())
        .unwrap_or_else(|| HOTKEYS[0].to_string())
}

/// A mode id as `Output::open` wants it: (paste, portal).
fn paste_flags(mode: &str) -> (bool, bool) {
    match mode {
        "portal" => (true, true),
        "paste" => (true, false),
        _ => (false, false),
    }
}

fn tray(app: &AppHandle) -> tauri::Result<()> {
    // ponytail: a fixed label rather than one that flips to "Stop". The pill
    // on screen already says which state we are in, and keeping the menu item
    // in sync means holding it in app state. Worth it once there is a real
    // settings window to hang it off — Phase 5.
    let bound = hotkey(app);
    let dictate =
        MenuItem::with_id(app, "dictate", format!("Dictate  {bound}"), true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Kotha", true, None::<&str>)?;

    // The offered list, plus whatever is actually bound if someone wrote a
    // fifth thing into settings.json — an unticked menu is worse than a long
    // one, and hiding a setting the app is obeying is how support tickets start.
    let offered: Vec<String> = HOTKEYS
        .iter()
        .map(|k| k.to_string())
        .chain((!HOTKEYS.contains(&bound.as_str())).then(|| bound.clone()))
        .collect();
    let keys = offered
        .iter()
        .map(|k| {
            CheckMenuItem::with_id(app, format!("key:{k}"), k, true, *k == bound, None::<&str>)
        })
        .collect::<tauri::Result<Vec<_>>>()?;
    let hotkeys = Submenu::with_items(
        app,
        "Hotkey",
        true,
        &keys.iter().map(|m| m as &dyn IsMenuItem<_>).collect::<Vec<_>>(),
    )?;

    // The first real setting, and the one worth having first: getting the paste
    // route wrong is what cost an evening in Phase 3, and until now the only way
    // to say it was an environment variable — which a shipped app has nobody to
    // set. A tray submenu because it is three fixed choices; a settings *window*
    // is Phase 7, and would be a window, a page and a capability entry for what
    // the platform already draws.
    let forced = paste_env().is_some();
    let chosen = paste_setting(app);
    let modes = PASTE_MODES
        .iter()
        .map(|(id, label)| {
            CheckMenuItem::with_id(app, format!("paste:{id}"), label, !forced, *id == chosen, None::<&str>)
        })
        .collect::<tauri::Result<Vec<_>>>()?;
    let output = Submenu::with_items(
        app,
        if forced { "Text output  (KOTHA_PASTE)" } else { "Text output" },
        true,
        &modes.iter().map(|m| m as &dyn IsMenuItem<_>).collect::<Vec<_>>(),
    )?;

    let menu = Menu::with_items(
        app,
        &[
            &dictate,
            &PredefinedMenuItem::separator(app)?,
            &hotkeys,
            &output,
            &PredefinedMenuItem::separator(app)?,
            &quit,
        ],
    )?;

    TrayIconBuilder::new()
        .icon(
            app.default_window_icon()
                .expect("no bundle icon — check tauri.conf.json")
                .clone(),
        )
        .tooltip("Kotha")
        .menu(&menu)
        .show_menu_on_left_click(true)
        .on_menu_event(move |app, event| match event.id.as_ref() {
            "dictate" => toggle(app),
            "quit" => app.exit(0),
            id if id.starts_with("key:") => {
                let want = &id["key:".len()..];
                let previous = hotkey(app);
                if want == previous {
                    return;
                }
                // Rebind before saving. A key another application already owns
                // fails here, which is the whole reason this menu exists — and
                // the one outcome that must not leave Kotha with nothing bound.
                let gs = app.global_shortcut();
                let _ = gs.unregister_all();
                if let Err(e) = gs.register(want) {
                    eprintln!("hotkey  {want} refused ({e}) — something else has it; keeping {previous}");
                    if let Err(e) = gs.register(previous.as_str()) {
                        eprintln!("hotkey  {previous} could not be taken back either ({e}) — use the tray icon");
                    }
                    return;
                }
                save_setting(&settings_path(app), "hotkey", want);
                let _ = dictate.set_text(format!("Dictate  {want}"));
                for (item, offer) in keys.iter().zip(&offered) {
                    let _ = item.set_checked(offer == want);
                }
                println!("hotkey  {want}");
            }
            id => {
                let Some(mode) = id.strip_prefix("paste:") else { return };
                // A check item toggles only itself when clicked, so the other
                // two have to be told or the menu shows two ticks. These are
                // radio buttons drawn as checkboxes; muda has no radio item.
                for (item, (known, _)) in modes.iter().zip(PASTE_MODES) {
                    let _ = item.set_checked(known == mode);
                }
                save_setting(&settings_path(app), "paste", mode);
                println!("output  {mode} — from the next dictation");
            }
        })
        .build(app)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Settings must round-trip, share one file without clobbering each other,
    /// and never hand `Output::open` a mode — or the shortcut plugin a string —
    /// that neither has heard of.
    ///
    /// Both failure modes here are quiet ones: a junk value would silently
    /// disable paste, and a corrupt file would panic on the index-assign in
    /// `save_paste_choice` — inside a tray click handler, where nobody is
    /// watching for a backtrace.
    #[test]
    fn settings_round_trip_and_survive_junk() {
        let path = std::env::temp_dir().join("kotha-settings-test.json");
        std::fs::remove_file(&path).ok();
        assert_eq!(paste_choice(&path), "copy", "no file means clipboard only");

        save_setting(&path, "paste", "portal");
        assert_eq!(paste_choice(&path), "portal");
        assert_eq!(paste_flags("portal"), (true, true));

        std::fs::write(&path, r#"{"paste":"telepathy"}"#).unwrap();
        assert_eq!(paste_choice(&path), "copy", "an unknown mode must fall back");

        // Not an object: `v["paste"] = ...` would panic on this.
        std::fs::write(&path, "3").unwrap();
        save_setting(&path, "paste", "paste");
        assert_eq!(paste_choice(&path), "paste");

        // Two settings share the file, so writing one must not lose the other.
        save_setting(&path, "hotkey", "F9");
        assert_eq!(hotkey_choice(&path), "F9");
        assert_eq!(paste_choice(&path), "paste", "saving the hotkey dropped the paste mode");

        // An unbindable string must fall back to the default rather than leave
        // the app with nothing registered at all.
        std::fs::write(&path, r#"{"hotkey":"Ctrl+Banana"}"#).unwrap();
        assert_eq!(hotkey_choice(&path), HOTKEYS[0]);
        for k in HOTKEYS {
            assert!(Shortcut::from_str(k).is_ok(), "{k} is not a usable accelerator");
        }

        std::fs::remove_file(&path).ok();
    }

    /// Resuming a half-finished file must produce the same bytes as fetching it
    /// whole.
    ///
    /// This is the part of `fetch_model` worth a test: the size and hash checks
    /// fail loudly on their own, but a resume that lands on the wrong offset
    /// produces a file of exactly the right length made of the wrong bytes, and
    /// only the hash would ever notice — after 775 MB. So the append-vs-truncate
    /// decision is checked here, on a 1.4 KB file, against the real server.
    ///
    /// It talks to HuggingFace, like `clipboard_is_borrowed_and_returned` talks
    /// to the real clipboard: the thing being tested is agreement with a system
    /// we do not control, and a fake would only prove we agree with ourselves.
    #[test]
    fn a_resumed_download_matches_a_whole_one() {
        let http: ureq::Agent = ureq::Agent::config_builder()
            .timeout_connect(Some(Duration::from_secs(20)))
            .build()
            .into();
        let dir = std::env::temp_dir().join("kotha-resume-test");
        std::fs::create_dir_all(&dir).unwrap();

        let whole = dir.join("whole");
        stream_to(&http, "config.json", &whole, 0, &mut |_| {}).unwrap();
        let expected = std::fs::read(&whole).unwrap();
        assert!(expected.len() > 200, "config.json came back suspiciously short");

        // Half of it on disk, as if a connection had dropped there.
        let half = expected.len() / 2;
        let partial = dir.join("partial");
        std::fs::write(&partial, &expected[..half]).unwrap();

        let mut reported = Vec::new();
        stream_to(&http, "config.json", &partial, half as u64, &mut |n| reported.push(n)).unwrap();

        assert_eq!(std::fs::read(&partial).unwrap(), expected, "resume produced different bytes");
        assert_eq!(reported.first(), Some(&(half as u64)), "resume did not start from the offset");
        assert_eq!(
            reported.last(),
            Some(&(expected.len() as u64)),
            "progress did not end at the full length"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}
