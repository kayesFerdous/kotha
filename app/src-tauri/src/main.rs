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
//! KOTHA_MODEL    the CTranslate2 model directory. Until Phase 5 downloads it,
//!                this defaults to ./models/whisper-medium-bn-en-cs-faster.
//! KOTHA_THREADS  decode threads (default: physical cores).
//! KOTHA_PASTE    unset = clipboard only, 1 = synthetic paste, portal = libei.
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use kotha_spike::correct::Corrector;
use kotha_spike::live::{self, Microphone, Output, Segmenter};
use kotha_spike::{suspicious_fusion, Engine};

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, LogicalSize, Manager, PhysicalPosition, WebviewWindow};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

/// PLAN.md's proposal, not yet final.
const HOTKEY: (Modifiers, Code) = (Modifiers::CONTROL.union(Modifiers::ALT), Code::Space);

/// Where the model lives until Phase 5 downloads it for the user.
fn model_dir() -> PathBuf {
    std::env::var_os("KOTHA_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| "models/whisper-medium-bn-en-cs-faster".into())
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

/// What the hotkey and the tray ask the worker to do.
enum Cmd {
    Start,
    Stop,
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

            let shortcut = Shortcut::new(Some(HOTKEY.0), HOTKEY.1);
            let pressed = shortcut;
            app.handle().plugin(
                tauri_plugin_global_shortcut::Builder::new()
                    .with_handler(move |app, sc, event| {
                        if event.state() == ShortcutState::Pressed && *sc == pressed {
                            toggle(app);
                        }
                    })
                    .build(),
            )?;

            // Not fatal. On Wayland this is the call most likely to fail, and
            // the tray icon still works when it does — so say so and carry on
            // rather than refusing to start.
            match app.global_shortcut().register(shortcut) {
                Ok(()) => println!("hotkey  Ctrl+Alt+Space"),
                Err(e) => eprintln!(
                    "hotkey  unavailable ({e}) — use the tray icon.\n\
                             On Wayland the underlying crate binds through X11, \
                     which the compositor may not expose."
                ),
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
                    toggle(&app);
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
    let sent = session.tx.lock().map(|tx| tx.send(cmd).is_ok()).unwrap_or(false);
    if !sent {
        eprintln!("worker is gone — dictation is not available");
    }
}

/// The worker thread: one engine, loaded once, for the life of the process.
fn worker(app: AppHandle, rx: mpsc::Receiver<Cmd>) {
    let threads = live::decode_threads();

    let corrector = Corrector::new();
    let mut out = Output::open(live::paste_mode());
    let mut engine: Option<Engine> = None;

    while let Ok(cmd) = rx.recv() {
        // A Stop with nothing running is ordinary — the dictation may have
        // already ended on its own. Ignore it rather than treating it as an
        // error.
        if !matches!(cmd, Cmd::Start) {
            continue;
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
    let Microphone { stream, blocks, mut intake } = live::open_microphone()?;
    show(app);
    let _ = app.emit("kotha://state", "listening");

    // Loaded on first use rather than at startup: a tray app that has not
    // dictated yet has no business holding 1.4 GB resident.
    if engine.is_none() {
        let t = std::time::Instant::now();
        let dir = model_dir();
        *engine = Some(Engine::load(&dir, threads).map_err(|e| {
            e.context(format!("could not load the model from {}", dir.display()))
        })?);
        println!("model   {} threads, loaded in {:.1}s", threads, t.elapsed().as_secs_f64());
    }
    let engine = engine.as_ref().expect("just loaded");

    let mut segmenter = Segmenter::new();
    let mut n = 0usize;

    loop {
        match rx.try_recv() {
            Ok(Cmd::Stop) | Err(mpsc::TryRecvError::Disconnected) => break,
            Ok(Cmd::Start) | Err(mpsc::TryRecvError::Empty) => {}
        }

        let block = match blocks.recv_timeout(POLL) {
            Ok(b) => b,
            // A silent room produces blocks too, so a timeout means the device
            // stopped rather than that nobody is speaking.
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        };

        let _ = app.emit("kotha://level", live::rms(&block));

        // The VAD closes a chunk at every pause, so text lands while the user
        // is still talking — which is the whole reason for segmenting at all
        // at 1.5x realtime.
        for utterance in intake.feed(&block, &mut segmenter)? {
            n += 1;
            deliver(app, engine, corrector, out, n, &utterance)?;
            let _ = app.emit("kotha://state", "listening");
        }
    }

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
    println!("stopped after {n} utterance(s)");
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

fn tray(app: &AppHandle) -> tauri::Result<()> {
    // ponytail: a fixed label rather than one that flips to "Stop". The pill
    // on screen already says which state we are in, and keeping the menu item
    // in sync means holding it in app state. Worth it once there is a real
    // settings window to hang it off — Phase 5.
    let dictate = MenuItem::with_id(app, "dictate", "Dictate  ⌃⌥Space", true, None::<&str>)?;
    let settings = MenuItem::with_id(app, "settings", "Settings…", false, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Kotha", true, None::<&str>)?;
    let menu = Menu::with_items(
        app,
        &[
            &dictate,
            &PredefinedMenuItem::separator(app)?,
            &settings,
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
        .on_menu_event(|app, event| match event.id.as_ref() {
            "dictate" => toggle(app),
            "quit" => app.exit(0),
            _ => {}
        })
        .build(app)?;

    Ok(())
}
