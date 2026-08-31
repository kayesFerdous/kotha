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
//! **The audio is fake.** `fake_level` stands in for the microphone so this
//! binary does not pull in `ct2rs`. The next commit replaces it with the real
//! cpal → VAD → Engine → corrector → clipboard chain out of `spike/`, which is
//! already written and already measured; nothing about that chain is in doubt.
//!
//! Run it:
//!
//! ```bash
//! cargo run -p kotha --release
//! ```
//!
//! Then press Ctrl+Alt+Space, or use the tray icon.

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, LogicalSize, Manager, PhysicalPosition, WebviewWindow};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

/// PLAN.md's proposal, not yet final.
const HOTKEY: (Modifiers, Code) = (Modifiers::CONTROL.union(Modifiers::ALT), Code::Space);

/// Microphone frames per second sent to the UI. Fast enough that the waveform
/// reads as continuous, slow enough to be free.
const FPS: u64 = 30;

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

/// Stands in for a decode, so the `thinking` state lasts long enough to look at.
const FAKE_DECODE: Duration = Duration::from_millis(1600);

/// Is a dictation in progress? The whole of the app's state, for now.
#[derive(Default)]
struct Session {
    listening: AtomicBool,
}

fn main() {
    prefer_x11();

    tauri::Builder::default()
        .manage(Session::default())
        .setup(|app| {
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

            // KOTHA_DEMO=1 runs one dictation on its own, a second after
            // startup. The window questions this gate exists to answer need
            // the pill actually on screen, and the hotkey may be exactly the
            // thing that is broken — so the gate must not depend on it.
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
                    thread::sleep(Duration::from_secs(4));
                    toggle(&app);
                    thread::sleep(FAKE_DECODE + HIDE_AFTER + Duration::from_millis(600));
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
fn toggle(app: &AppHandle) {
    let session = app.state::<Session>();

    if session.listening.swap(false, Ordering::SeqCst) {
        // Stopping: the model would now be decoding.
        let _ = app.emit("kotha://state", "thinking");
        let app = app.clone();
        thread::spawn(move || {
            // ponytail: a sleep where the decode goes. The real chain — VAD,
            // Engine, corrector, clipboard — is written and measured in
            // spike/src/bin/live.rs and drops in here next.
            thread::sleep(FAKE_DECODE);
            let _ = app.emit("kotha://state", "done");
            thread::sleep(HIDE_AFTER);
            if let Some(w) = app.get_webview_window("pill") {
                let _ = w.hide();
            }
        });
        return;
    }

    session.listening.store(true, Ordering::SeqCst);
    if let Some(w) = app.get_webview_window("pill") {
        // Placement is re-applied on every show: the pill should follow the
        // screen the user is actually on, and monitors come and go.
        let _ = place(&w);
        let _ = w.show();

        // Click-through, so the pill is furniture and not an obstacle.
        //
        // This has to happen *after* the first show, not in setup(). tao
        // 0.35.3 handles the request with `window.window().unwrap()`
        // (linux/event_loop.rs:457) — the GDK window, which does not exist
        // until GTK realises the widget. A window created `visible: false`
        // has not been realised, so asking in setup() aborts the process from
        // inside the GTK main loop, where it cannot even unwind. Silent until
        // it is fatal, and worth reporting upstream: the code already has the
        // Option in hand.
        let _ = w.set_ignore_cursor_events(true);
        // The property the whole feature rests on. Self-reported by the
        // toolkit, so it is evidence rather than proof, but a `true` here
        // would be conclusive the other way.
        println!(
            "window  shown, focused = {:?}, at {:?}",
            w.is_focused(),
            w.outer_position()
        );
    }
    let _ = app.emit("kotha://state", "listening");

    let app = app.clone();
    thread::spawn(move || {
        let start = Instant::now();
        while app.state::<Session>().listening.load(Ordering::SeqCst) {
            let _ = app.emit("kotha://level", fake_level(start.elapsed().as_secs_f32()));
            thread::sleep(Duration::from_millis(1000 / FPS));
        }
    });
}

/// A plausible speech envelope, so the waveform can be judged before the
/// microphone is wired up. Replaced by the RMS of each real audio frame.
fn fake_level(t: f32) -> f32 {
    let syllable = (t * 5.5).sin().powi(2) - 0.08;
    let tremor = 0.75 + 0.25 * (t * 21.0).sin();
    let breath = if t % 4.2 < 0.55 { 0.06 } else { 1.0 };
    (syllable.max(0.0) * tremor * breath * 0.22).max(0.0)
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
