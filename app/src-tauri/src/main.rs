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
//! Then press F9 — ⌥⇧D on macOS, where F9 is Next Track — or use the tray
//! icon. The key is settable, and the default is per-platform; see `HOTKEYS`.
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
use std::time::{Duration, Instant};

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
/// exact int8 weights that were benchmarked.
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
/// Kotha has always been described as "press again, or stay
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
            // Kotha is a tray application, and on macOS that is a policy, not
            // a style. `Accessory` drops the Dock icon and the ⌘-Tab entry —
            // right on its own for something driven by a hotkey and a menu bar
            // item — but the reason it is load-bearing is focus: a `Regular`
            // app *activates* when one of its windows is ordered front, and
            // activating deactivates whatever the user was typing into. The
            // pill cannot become key (see `set_focusable` below), yet without
            // this the app around it would still take the foreground and the
            // caret would stop.
            //
            // It does not break the first-run window, which is the one window
            // here that *should* take focus: `set_focus` reaches
            // `activateIgnoringOtherApps: YES` in tao
            // (macos/util/async.rs:236), and an accessory app is allowed to
            // activate itself when it asks explicitly. Checked against tao
            // 0.35.3, not yet watched on screen.
            //
            // The one thing given up is the application menu bar, and with it
            // the default ⌘C/⌘V/⌘A inside Kotha's own windows. Nothing in
            // Phase 4 has a text field. Phase 5's settings will, and at that
            // point this needs an edit menu rather than a different policy.
            #[cfg(target_os = "macos")]
            app.set_activation_policy(tauri::ActivationPolicy::Accessory);

            let pill = app
                .get_webview_window("pill")
                .expect("no window labelled `pill` — check tauri.conf.json");

            // The one thing this app cannot get wrong. A dictation pill that
            // steals focus takes the caret with it and the feature collapses.
            //
            // On macOS this is stronger than it looks. tao's window class
            // overrides both
            // `canBecomeKeyWindow` and `canBecomeMainWindow` to return this
            // flag (macos/window.rs:415-425), so `false` here makes the pill
            // *structurally* unable to become the key window — which is the
            // guarantee `NSWindowStyleMaskNonactivatingPanel` exists to give,
            // reached without becoming an NSPanel — no NSPanel needed.
            pill.set_focusable(false)?;
            pill.set_size(LogicalSize::new(PILL_WINDOW.0, PILL_WINDOW.1))?;
            no_activate(&pill);

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
    // First thing on this thread, and before any engine exists. This is the
    // thread that will build the engine and run every decode on it, and on
    // Apple Silicon the QoS set here is what decides whether that work lands
    // on the performance cores or the efficiency ones — CTranslate2's pool
    // inherits it. See `live::prefer_performance_cores`.
    live::prefer_performance_cores();

    let threads = live::decode_threads();

    let corrector = Corrector::new();
    let mut mode = paste_setting(&app);
    let mut out = open_output(&app, &mode);
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
            out = open_output(&app, &mode);
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
    let blocks = meter(app, blocks, level_chunk);
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
    let mut worst_lag = Duration::ZERO;
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

        let (captured, block) = match blocks.recv_timeout(POLL) {
            Ok(b) => b,
            // A silent room produces blocks too, so a timeout means the device
            // stopped rather than that nobody is speaking.
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break "device gone",
        };

        // How far behind the microphone this loop is running. It blocks for
        // the length of every decode, so this climbs to seconds after each
        // sentence — which is fine for the audio, since nothing is dropped,
        // and is exactly why the level meter does not live in here.
        worst_lag = worst_lag.max(captured.elapsed());

        if !logged_block {
            // `level_chunk` is 1/30 s of interleaved samples at the device's
            // own rate, so thirty of them is one second of what the driver
            // delivers. Not SAMPLE_RATE: that is the model's 16 kHz, after
            // resampling, and dividing by it made this line report an 11 ms
            // block at 48 kHz as 32 ms.
            let per_second = (level_chunk * 30) as f64;
            println!(
                "audio   {} samples per block ({:.1} ms) | waveform at 30 levels/s",
                block.len(),
                block.len() as f64 / per_second * 1000.0,
            );
            logged_block = true;
        }

        // The VAD closes a chunk at every pause, so text lands while the user
        // is still talking — which is the whole reason for segmenting at all
        // at 1.5x realtime.
        for utterance in intake.feed(&block, &mut segmenter)? {
            // Only a chunk that survived the engine's silence gate counts as
            // speech. Resetting the idle timer on a discarded one is what kept
            // the pill alive forever: the VAD opens on a breath, the engine
            // throws the decode away, and the timer restarts anyway — so the
            // dictation could never end on its own once it had started.
            if deliver(app, engine, corrector, out, n + 1, &utterance)? {
                n += 1;
                last_voice = std::time::Instant::now();
            }
            let _ = app.emit("kotha://state", "listening");
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
        if deliver(app, engine, corrector, out, n + 1, &tail)? {
            n += 1;
        }
    }

    app.state::<Session>().listening.store(false, Ordering::SeqCst);
    let _ = app.emit("kotha://state", if n > 0 { "done" } else { "idle" });
    hide_soon(app, if n > 0 { HIDE_AFTER } else { Duration::from_millis(400) });
    println!(
        "stopped on {stopped_by} after {n} utterance(s); loop fell up to {:.1}s behind the microphone\n",
        worst_lag.as_secs_f64()
    );
    Ok(())
}

/// Drive the pill's waveform from the microphone directly, and pass the audio
/// on to the dictation loop.
///
/// This used to happen inside the loop, and the waveform lagged the voice by
/// the length of a decode. The loop transcribes on its own thread — the
/// engine is not `Send` and lives there — so every utterance stops it for a
/// few seconds, during which the capture blocks queue up, and the levels for
/// everything said meanwhile arrived in one burst when it came back. The
/// pill would sit still through a sentence and then twitch through it in a
/// frame. The audio is fine with that, because nothing is dropped and text
/// lands where it should; the meter is not, because a level that arrives
/// late is a level that lies.
///
/// So the meter sits between the microphone and the loop. It reads each
/// block the moment the driver delivers it, turns it into levels, and only
/// then forwards the block, stamped with when it was captured so the loop can
/// say how far behind it is.
///
/// **One level per 1/30 s of audio, whatever size the blocks are.** The first
/// version sliced each block into 33 ms pieces, which only does anything to a
/// block longer than 33 ms. CoreAudio delivers the M2's microphone in
/// 512-frame blocks — 10.7 ms at 48 kHz — so every block became one level:
/// about 94 a second instead of the 30 `pill.js` is tuned for. That tripled
/// the script evaluations sent into the web view, scrolled the 21-bar history
/// across the pill in a fifth of a second, made a release tuned per frame
/// decay three times too fast, and restarted every bar's 70 ms height
/// transition every 11 ms, so no bar ever reached the height it was given —
/// the wave looked flat while the user was talking. A window now closes when
/// 1/30 s of samples has accumulated, across as many blocks as that takes. A
/// sound reaches the pill within one window, 33 ms, of the driver handing it
/// over. It ends by itself: dropping the stream
/// closes the microphone's channel, the `for` finishes, and the forwarding
/// end closes behind it, which is what the loop sees as "device gone".
///
/// On the way out it logs how many blocks arrived and the loudest level among
/// them. Without that line a dead microphone and a quiet user leave identical
/// logs — no VAD, no decode, no text — and a flat pill looks the same either
/// way. macOS makes the first case common: a process without microphone
/// permission is not refused, it is handed zeros.
fn meter(
    app: &AppHandle,
    mic: mpsc::Receiver<Vec<f32>>,
    level_chunk: usize,
) -> mpsc::Receiver<(Instant, Vec<f32>)> {
    let app = app.clone();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        let (mut blocks, mut loudest) = (0usize, 0f32);
        // The part-filled window, carried from one block into the next.
        let (mut squares, mut filled) = (0f32, 0usize);
        for block in mic {
            blocks += 1;
            for &sample in &block {
                squares += sample * sample;
                filled += 1;
                if filled == level_chunk {
                    let level = (squares / filled as f32).sqrt();
                    loudest = loudest.max(level);
                    let _ = app.emit("kotha://level", level);
                    (squares, filled) = (0.0, 0);
                }
            }
            if tx.send((Instant::now(), block)).is_err() {
                break;
            }
        }
        println!("meter   {blocks} blocks from the microphone, loudest level {loudest:.4}");
        if blocks == 0 {
            eprintln!("meter   no audio arrived at all — the input stream never delivered");
        } else if loudest < 0.002 {
            eprintln!(
                "meter   the microphone delivered only silence. On macOS that is what a \
                 missing microphone permission looks like: check System Settings → \
                 Privacy & Security → Microphone for the app or terminal that launched Kotha"
            );
        }
    });
    rx
}

/// Transcribe one utterance, repair its English, and put it where the user
/// asked for it.
/// Returns whether anything was actually said.
///
/// `false` means the engine's silence gate threw the chunk away, and the
/// caller must treat that as silence — not as a reason to keep listening. See
/// the loop in `dictate`.
fn deliver(
    app: &AppHandle,
    engine: &Engine,
    corrector: &Corrector,
    out: &mut Output,
    n: usize,
    utterance: &[f32],
) -> anyhow::Result<bool> {
    let secs = utterance.len() as f64 / kotha_spike::SAMPLE_RATE as f64;
    let _ = app.emit("kotha://state", "thinking");

    let t = std::time::Instant::now();
    let text = engine.transcribe(utterance)?;
    let took = t.elapsed().as_secs_f64();
    let text = text.trim();

    println!("[{n}] {secs:.1}s audio → {took:.1}s decode ({:.2}x realtime)", secs / took);
    if text.is_empty() {
        println!("    (nothing said)\n");
        return Ok(false);
    }
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
    Ok(true)
}

/// Tell the window manager this window is furniture, not an application.
///
/// `set_focusable(false)` is not enough on X11. It maps to GTK's `accept_focus`,
/// which stops the pill taking *keyboard* focus — and `is_focused()` duly
/// reports `false`. But KWin still makes it the **active window**, which is a
/// different thing, and the consequences are the ones the user actually sees:
/// the window behind it stops drawing its caret, and if the "Dim Inactive"
/// desktop effect is on, that window visibly darkens. Measured 2026-09-02 with
/// `xdotool getactivewindow`: the active window became `Kotha` for ~1.8 s every
/// time the pill appeared.
///
/// The fix is the X11 window type hint. `Notification` is what this window
/// actually is — transient, informational, never interacted with — and no
/// window manager promotes a notification to active. `tao` does not expose type
/// hints, so this reaches through Tauri's `gtk_window()`.
///
/// Must be called before the window is first shown: the hint is read when the
/// window is mapped. The pill is created `visible: false`, so `setup()` is the
/// right place.
///
/// Not fatal if it fails. Without it the pill still works; it just steals the
/// active-window title on the way past.
#[cfg(target_os = "linux")]
fn no_activate(w: &WebviewWindow) {
    use gtk::prelude::GtkWindowExt;
    match w.gtk_window() {
        Ok(g) => g.set_type_hint(gtk::gdk::WindowTypeHint::Notification),
        Err(e) => eprintln!("window  could not set the type hint ({e}); \
                             the pill may dim the window behind it"),
    }
}

/// The same job on macOS, where the mechanism is different in every respect
/// except the consequence.
///
/// AppKit has no window type hints. What it has instead is a *level* and a
/// *collection behaviour*, and the pill needs both changed for reasons the X11
/// path never had to think about:
///
///   * **Level.** Tauri's `alwaysOnTop` maps to `NSFloatingWindowLevel` (3),
///     which floats above ordinary windows and below almost everything
///     interesting — including a full-screen application, which owns its own
///     Space and covers every window below the status level. A dictation pill
///     that vanishes the moment the user goes full-screen is a pill that is
///     absent exactly when someone is writing. `NSStatusWindowLevel` (25) is
///     where the menu bar's own furniture lives and is the right neighbourhood
///     for this.
///
///   * **Collection behaviour.** By default a window belongs to the Space it
///     was created on. The hotkey is global, so the pill has to be able to
///     appear on whichever Space the user is on when they press it, over a
///     full-screen window, without dragging them somewhere else.
///     `CanJoinAllSpaces` says "follow the user, do not move them";
///     `FullScreenAuxiliary` is what permits it over a full-screen app at
///     all. The pair lives in `PILL_BEHAVIOUR`, with the story of the flag
///     that used to be there too.
///
/// Focus itself is handled a layer up, by the accessory activation policy in
/// `setup` — see the note there. This function is only about *where* the pill
/// is allowed to draw. The two are separable and both are required: an
/// accessory app whose window sits at the floating level still disappears
/// under full screen, and a status-level window in a regular app still steals
/// activation.
///
/// **Unverified.** Written 2026-09-03 on a machine that has never built this
/// target; the reasoning is from AppKit's documented semantics, not from
/// watching it. What has to be checked on the first real run is the Phase 4
/// acceptance test — the caret keeps blinking in the window behind — and then
/// the same thing again with that window full-screen.
///
/// Safety: `NSWindow` is `MainThreadOnly`, and this is called from `setup`,
/// which Tauri runs on the main thread. Both setters are safe wrappers; the
/// only unsafe step is trusting `ns_window()` to hand back a live `NSWindow`,
/// which it does for a window that exists — and it is fetched by label
/// immediately above the call.
///
/// Not fatal if it fails, for the same reason the X11 hint is not: without it
/// the pill still works, it is just in the wrong place in the stack.
#[cfg(target_os = "macos")]
fn no_activate(w: &WebviewWindow) {
    use objc2_app_kit::{NSStatusWindowLevel, NSWindow};

    let ptr = match w.ns_window() {
        Ok(p) => p.cast::<NSWindow>(),
        Err(e) => {
            eprintln!("window  no NSWindow ({e}); the pill may hide under \
                       full-screen apps");
            return;
        }
    };

    let Some(win) = (unsafe { ptr.as_ref() }) else {
        eprintln!("window  NSWindow pointer was null; the pill may hide under \
                   full-screen apps");
        return;
    };

    win.setCollectionBehavior(PILL_BEHAVIOUR);
    win.setLevel(NSStatusWindowLevel);
    println!("window  status level, all spaces, over full screen");
}

/// Where the pill may appear: on every Space, and over a full-screen app.
///
/// `Stationary` used to be the third flag here, and it is gone on purpose.
/// AppKit documents it as "unaffected by Exposé; stays visible and stationary,
/// like the desktop window" — and the desktop is the one window that belongs
/// to a single Space. The first macOS build showed the pill on the Space Kotha
/// was launched from rather than the one the hotkey was pressed on, which is
/// exactly what a window pinned to one Space does. Nothing here needs to
/// survive Mission Control, so the flag whose only job was that is the one to
/// drop. Without it the window gets `Transient` by default — "floats in
/// spaces, hidden by Exposé" — which is what a HUD wants anyway.
///
/// One place, because it is applied twice: once at setup, and again on every
/// show — see `raise_regardless` for why the second time is not redundant.
#[cfg(target_os = "macos")]
const PILL_BEHAVIOUR: objc2_app_kit::NSWindowCollectionBehavior =
    objc2_app_kit::NSWindowCollectionBehavior::CanJoinAllSpaces
        .union(objc2_app_kit::NSWindowCollectionBehavior::FullScreenAuxiliary);

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn no_activate(_: &WebviewWindow) {}

/// Bring the pill forward over whatever application is currently active.
///
/// `show()` is not enough on macOS, and this is the bug that hid the pill for
/// the whole of the first build. tao's `set_visible(true)` reaches
/// `orderFront:`, and **`orderFront:` orders within the calling application's
/// own window list**. When Kotha is not the active app — which is always, since
/// the entire point is that the user is typing into something else — that puts
/// the pill behind the active app's windows. It was on screen, right size,
/// right place, `is_visible()` answering `true`, and underneath whatever the
/// user was looking at. Over an empty desktop it appeared perfectly, which is
/// what made it look like a rendering fault for so long.
///
/// `orderFrontRegardless:` is AppKit's answer to exactly this case: order front
/// even though the application is not active, without activating it. It is what
/// HUD and overlay windows are meant to use.
///
/// The window level from `no_activate` is still required and does a different
/// job. Level picks the *band* the window lives in — above full-screen apps,
/// alongside the menu bar. This decides whether it is brought forward within
/// that band at all. Neither substitutes for the other, which is why the level
/// being right was not enough to make the pill visible.
///
/// Re-applied on every show rather than set once, because it is an action, not
/// a property: there is nothing to stay set.
///
/// The level and the collection behaviour are re-asserted here too, just
/// before the window is ordered in, even though `no_activate` set them at
/// setup. Setup runs on a window that has never been on screen — AppKit has
/// not yet given it a backing window, and the behaviour it carries is only
/// handed to the window server when it first orders in. Setting it again with
/// the window about to appear costs two calls and removes one untested
/// assumption. `isOnActiveSpace` is logged on both sides of the ordering: for
/// a hidden window it answers whether ordering it in *would* land on the
/// Space the user is looking at, which is exactly the question. A `false`
/// there with `CanJoinAllSpaces` set is AppKit ignoring the flag, and the log
/// says so rather than leaving it to be noticed.
///
/// Safety: `NSWindow` is main-thread-only and the caller dispatches this
/// through `run_on_main_thread`. The only unsafe step is trusting `ns_window()`
/// for a window that exists, which it does — it was fetched by label a moment
/// ago.
#[cfg(target_os = "macos")]
fn raise_regardless(w: &WebviewWindow) {
    use objc2_app_kit::{NSStatusWindowLevel, NSWindow, NSWindowCollectionBehavior};

    let Ok(ptr) = w.ns_window() else {
        eprintln!("window  no NSWindow to raise; the pill may stay behind the active app");
        return;
    };
    let Some(win) = (unsafe { ptr.cast::<NSWindow>().as_ref() }) else {
        eprintln!("window  NSWindow pointer was null; the pill may stay behind the active app");
        return;
    };

    let before = win.isOnActiveSpace();

    // `CanJoinAllSpaces` alone was measured not to be enough. Kotha was
    // launched from a terminal inside a full-screen app, the first dictation
    // showed the pill there, and every dictation after that reported
    // `on active Space: false` from another Space — before and after ordering
    // in, with `CanJoinAllSpaces` set. A `FullScreenAuxiliary` window that has
    // once been shown over a full-screen app stays attached to that Space
    // when it is ordered out, and the all-Spaces flag does not detach it.
    //
    // `MoveToActiveSpace` is AppKit's instruction for exactly this: when the
    // window is ordered in, bring it to the Space the user is on. It cannot
    // be combined with `CanJoinAllSpaces` — they are alternatives — so it is
    // held only across the ordering, and the all-Spaces behaviour goes back
    // straight afterwards so that switching Space mid-dictation still takes
    // the pill along.
    win.setCollectionBehavior(
        NSWindowCollectionBehavior::MoveToActiveSpace
            | NSWindowCollectionBehavior::FullScreenAuxiliary,
    );
    win.setLevel(NSStatusWindowLevel);
    win.orderFrontRegardless();
    let after = win.isOnActiveSpace();
    win.setCollectionBehavior(PILL_BEHAVIOUR);
    println!(
        "window  ordered front regardless | level {} | behaviour {:#x} | on active Space: {before} before, {after} after",
        win.level(),
        win.collectionBehavior().bits()
    );
    if !after {
        eprintln!(
            "window  the pill is NOT on the active Space — it is showing on another desktop, \
             even ordered in with MoveToActiveSpace; see raise_regardless"
        );
    }
}

/// Put the pill on screen, wherever the user's screen currently is.
///
/// Placement is re-applied on every show: monitors come and go, and the pill
/// should appear on the one being looked at.
fn show(app: &AppHandle) {
    let Some(w) = app.get_webview_window("pill") else {
        return;
    };
    let _ = place(app, &w);

    // `show()` alone leaves the pill behind the active application, and on a
    // window that has never been on screen it is also the moment AppKit
    // learns which Spaces the window belongs to. So the macOS ordering goes
    // first, with the behaviour re-asserted in the same breath — see
    // `raise_regardless`. Dispatched to the main thread because this runs on
    // the worker, and `NSWindow` may only be touched from the main one; the
    // `show()` below queues behind it on the same event loop, so the order
    // holds.
    #[cfg(target_os = "macos")]
    {
        let w2 = w.clone();
        if let Err(e) = w.run_on_main_thread(move || raise_regardless(&w2)) {
            eprintln!("window  could not reach the main thread to raise: {e}");
        }
    }

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
    println!(
        "window  visible = {:?} | outer_position = {:?} | outer_size = {:?}",
        w.is_visible(), w.outer_position(), w.outer_size()
    );

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

/// Bottom centre of the monitor the user is looking at.
///
/// "Looking at" is approximated by the mouse pointer. The honest answer would
/// be the display holding the focused window's caret, and no toolkit will say
/// which that is without the Accessibility API; the pointer is where the user
/// last did something, and on one screen the two are the same screen. The
/// window's own monitor is the fallback, and it is the wrong answer more often
/// than it looks: a hidden window reports the display it was *last* on, which
/// is wherever the previous dictation happened, not where this one is.
fn place(app: &AppHandle, w: &WebviewWindow) -> tauri::Result<()> {
    let monitor = match monitor_under_pointer(app) {
        Some(m) => m,
        None => match w.current_monitor()? {
            Some(m) => m,
            None => return Ok(()),
        },
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

    let x = origin.x + (screen.width as i32 - win_w) / 2;
    let y = origin.y + screen.height as i32 - win_h - (bottom_margin() * scale) as i32;

    println!(
        "place   monitor {}x{} at ({},{}) scale {scale} | pill {win_w}x{win_h} -> ({x},{y})",
        screen.width, screen.height, origin.x, origin.y
    );

    w.set_position(PhysicalPosition::new(x, y))
}

/// The monitor the mouse pointer is on, if the toolkit can say.
///
/// tao hands the pointer back in physical pixels scaled by the *primary*
/// display (`macos/util/mod.rs` `cursor_position`, and the same on X11), while
/// `monitor_from_point` compares against display bounds in logical points
/// (`macos/monitor.rs` `from_point`). Dividing by the primary scale is what
/// makes the two agree; on a mixed-scale desk it is an approximation, but the
/// primary is the one the pointer's units came from. Wayland reports (0, 0)
/// for the pointer and cannot place windows anyway, so that exact answer is
/// treated as "don't know".
fn monitor_under_pointer(app: &AppHandle) -> Option<tauri::Monitor> {
    let cursor = app.cursor_position().ok()?;
    if cursor.x == 0.0 && cursor.y == 0.0 {
        return None;
    }
    let scale = app
        .primary_monitor()
        .ok()
        .flatten()
        .map(|m| m.scale_factor())
        .unwrap_or(1.0);
    let monitor = app.monitor_from_point(cursor.x / scale, cursor.y / scale).ok().flatten();
    match &monitor {
        Some(m) => println!(
            "place   pointer at ({:.0},{:.0}) -> monitor {}",
            cursor.x,
            cursor.y,
            m.name().map(String::as_str).unwrap_or("(unnamed)")
        ),
        None => println!(
            "place   pointer at ({:.0},{:.0}) is on no known monitor; using the window's own",
            cursor.x, cursor.y
        ),
    }
    monitor
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
/// routes, and the notes on text output are written in terms of it. When it
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

/// The hotkeys the tray offers, in Tauri's accelerator syntax. The first is the
/// default.
///
/// A fixed list and not a key-capture widget: capturing a chord needs a focused
/// window and a page to draw it on, and what this actually has to solve is a
/// *collision*, not a preference.
///
/// **F9 is the default, and `Ctrl+Alt+Space` is not, for two measured reasons.**
/// That chord is fcitx's and ibus's input-method switch, so on a Bangladeshi
/// desktop it is very likely already bound to Avro. And on Wayland the key
/// reaches the focused application as well as us — measured on KDE Wayland —
/// where `Ctrl+Alt+Space` inserts a stray `^[^@` into whatever you were about to
/// dictate into. F9 is delivered twice as well, but inserts nothing — so the
/// defect is invisible on it. That is a dodge, not a fix; the portal route is
/// still the fix.
///
/// The trade F9 makes is that a bare function key is easier for another
/// application to claim — an IDE's build key, a browser extension — which is
/// exactly what the rest of this list is for.
///
/// **On macOS the default is `Alt+Shift+D`, because F9 is not a function key
/// there.** On every current Apple keyboard the top row is media controls
/// unless the user has turned on "Use F1, F2, etc. keys as standard function
/// keys" — and F9 specifically is Next Track. Left as the default, the shipped
/// hotkey would not start a dictation; it would skip the user's music. Both of
/// F9's reasons are Linux's anyway: fcitx and ibus are not what a Mac switches
/// input sources with, and the double-delivery defect is KWin's.
///
/// `Alt+Shift+D` — ⌥⇧D — is a chord rather than a bare key, so it is harder for
/// another application to claim; it is nowhere near ⌃Space and ⌃⌥Space, which
/// is where macOS puts input-source switching and therefore where a Bengali
/// keyboard layout lives; and D is for dictate. Option is the special-character
/// modifier, so ⌥⇧D would type `Î` if it were not grabbed — it is grabbed,
/// because Carbon's `RegisterEventHotKey` consumes the key.
///
/// Kayes's to overrule: it is a default, changeable from the tray without
/// touching this list.
///
/// The list is what the menu offers, not what is accepted. `hotkey` validates by
/// parsing, so anything Tauri understands can be written into `settings.json` by
/// hand and the menu will show it alongside these.
#[cfg(target_os = "macos")]
const HOTKEYS: [&str; 4] = ["Alt+Shift+D", "Ctrl+Shift+Space", "Ctrl+Alt+Space", "F9"];

#[cfg(not(target_os = "macos"))]
const HOTKEYS: [&str; 4] = ["F9", "Ctrl+Shift+Space", "Alt+Shift+D", "Ctrl+Alt+Space"];

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

/// Open the text output, carrying the portal's permission across restarts.
///
/// On the portal route the desktop shows an "allow remote control?" dialog the
/// first time. Saying yes hands back a token; giving that token back on the
/// next launch restores the same grant with no dialog at all. The token is
/// single-use — the portal issues a new one each time — so it is written back
/// on every open, not just the first.
///
/// It lives in `settings.json` next to `hotkey` and `paste`, because settings
/// are one file. It is not a secret: it identifies a grant this
/// user already made to this app, and it is worthless to anyone else.
fn open_output(app: &AppHandle, mode: &str) -> Output {
    let path = settings_path(app);
    let out = Output::open(paste_flags(mode), setting(&path, "restore_token"));
    if let Some(token) = out.restore_token.as_deref() {
        save_setting(&path, "restore_token", token);
    }
    out
}

/// A mode id as `Output::open` wants it: (paste, portal).
fn paste_flags(mode: &str) -> (bool, bool) {
    match mode {
        "portal" => (true, true),
        "paste" => (true, false),
        _ => (false, false),
    }
}

/// The picture in the tray, which is not the same picture on macOS.
///
/// Everywhere else the application icon is the right answer: a menu bar or a
/// system tray that draws colour should be given the same object the user sees
/// in a launcher.
///
/// macOS draws a **template image**. It reads the alpha channel, discards the
/// colour, and tints the shape itself — dark on a light menu bar, light on a
/// dark one, and the highlight colour while the menu is open. That is what
/// makes a menu bar look like one thing rather than a row of stickers, and it
/// is not optional in the sense that matters: handed `kotha.svg`, macOS would
/// take the alpha of a *filled rounded square* and draw a solid blob, with the
/// waveform inside it invisible. The icon is also `#18181b`, which on a dark
/// menu bar is nearly the background colour, so the untemplated version fails
/// twice over.
///
/// So macOS gets `tray-macos.png`: the same seven bars, no capsule, pure black
/// on transparent, 36 px for an 18 pt slot (tray-icon 0.24.2 scales to 18 pt;
/// 2x is crisp on this display). It is compiled in rather than bundled,
/// because a tray icon that depends on a file being found is a tray icon that
/// is sometimes missing.
///
/// **Unverified**, like the rest of 2026-09-03: the pixels were checked (seven
/// bars, every opaque pixel `#000000`, corners fully transparent), but nothing
/// has drawn them into a menu bar yet.
#[cfg(target_os = "macos")]
fn tray_icon(_app: &AppHandle) -> tauri::Result<tauri::image::Image<'_>> {
    tauri::image::Image::from_bytes(include_bytes!("../icons/tray-macos.png"))
}

#[cfg(not(target_os = "macos"))]
fn tray_icon(app: &AppHandle) -> tauri::Result<tauri::image::Image<'_>> {
    Ok(app
        .default_window_icon()
        .expect("no bundle icon — check tauri.conf.json")
        .clone())
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
        .icon(tray_icon(app)?)
        // A no-op everywhere but macOS, where it is the difference between an
        // icon and a smudge. See `tray_icon`.
        .icon_as_template(cfg!(target_os = "macos"))
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
