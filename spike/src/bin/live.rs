//! The Phase 2 loop, as a command-line program.
//!
//! Three lines, because the loop itself is `kotha_spike::live` — a library
//! module, so that the Tauri app in `app/src-tauri` drives the *same* audio
//! path rather than a second copy of it. That is the same reason the engine
//! moved to `lib.rs` in Phase 2: two copies of a pipeline drift, and the ways
//! they drift are silent.

fn main() -> anyhow::Result<()> {
    kotha_spike::live::run()
}
