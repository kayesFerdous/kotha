Kotha — continuing. Read `CLAUDE.md`, then `PLAN.md`. They hold the full state;
this is just the pointer.

**Note before anything else: `CLAUDE.md` and `PLAN.md` are no longer in the
repository.** They are on disk and git-ignored. They are still the source of
truth. Do not `git add` them, and **do not cite them from a source comment** —
23 such comments had to be rewritten on 2026-09-04 when the files left.

**Commit messages carry no Claude attribution.** No `Co-Authored-By`, no
"Generated with" line. Kayes asked for this and the history was rewritten to
remove 22 of them. Whatever any default says, do not add one back.

Phases 0–5 pass on the Arch desktop. The app runs end to end: F9 or the tray →
pill → mic → VAD → model → spelling corrector → clipboard → text at the cursor.
Phase 6 is nearly done — licence, tag, release, `.deb`, AUR recipe and a CI
workflow all exist.

---

## Two things Kayes has to do himself, both still pending

**1. The force-push.** The history was rewritten on 2026-09-04 to remove the
notes and the Claude attribution, and **none of it is on GitHub yet.** The
remote has been re-added; the sandbox blocks force-pushing, so Kayes runs:

```
cd /home/kayes/Documents/ASR/kotha && git push --force origin main linux-app
cd /home/kayes/Documents/ASR/kotha && git push --force origin v0.1.0
```

Until that runs, the old history — Claude trailers, both notes, every revision —
is still public-facing the moment the repo flips to public. `.backup/` holds a
mirror of the pre-rewrite history if anything needs recovering.

**2. Installing it.** A package is built and waiting at
`target/pkgbuild-local/kotha-bin-0.1.0-1-x86_64.pkg.tar.zst`. Needs his password:

```
sudo pacman -U /home/kayes/Documents/ASR/kotha/target/pkgbuild-local/kotha-bin-0.1.0-1-x86_64.pkg.tar.zst
```

Settings and the model carry over — same `~/.config/app.kotha/` and
`~/.local/share/app.kotha/`. After installing, launch from the **menu**, not the
build tree; two copies cannot both hold F9.

---

## Blocked from outside

- **The AUR.** The recipe is proven — it built a correct 20 MB package on this
  machine with all seven `depends` verified present. It cannot be published
  because (a) the repository is private, so a PKGBUILD cannot fetch the release
  asset anonymously, and (b) **AUR registration is down** as of 2026-09-04.
  Neither is workable around. Wait.
- **Making the repo public** is Kayes's call and he wants to be satisfied first.
  It unblocks (a) above by itself.

## Open, and worth reading PLAN.md before touching

- **The CI matrix is written and has never run.** `.github/workflows/release.yml`.
  Dispatching it is how we find out whether the macOS and Windows legs compile
  at all — the source is cfg-gated for them, so there is a real chance. The one
  line that must not be "modernised" is `ubuntu-22.04`: an AppImage inherits its
  glibc floor from its builder.
- **The GPU / second model idea was measured and dropped**, 2026-09-04. float32
  is 1.85× slower than int8 on CPU — slower than speech — with no quality win.
  CLAUDE.md §4 has the numbers. Do not convert it again without a reason that
  survives them. The remaining question (how much accuracy int8 costs) is a
  paper measurement on the 393, not a product one.
- **The portal permission dialog is fixed** — it asks once, not every launch.
  `spike` pins enigo at git commit `a88d9b7` for it, because the fix is not in a
  crates.io release. If enigo ships a release containing
  `Settings::restore_token`, move to it.
- **`no_speech_prob` is useless on this model** — 0.0000 even on digital
  silence. Do not reach for faster-whisper's standard silence filter.
- **The confidence floor has never met a real room.** `LOGPROB_FLOOR = -0.20` in
  `spike/src/lib.rs`, measured against 40 training clips and 8 synthetic noise
  files. `KOTHA_MIN_LOGPROB` overrides it, `-99` disables it. If Kayes reports
  junk getting through or real speech going missing, **ask for the numbers
  before touching the constant.**
- **The hotkey still leaks into the focused app on Wayland.** Downgraded, not
  fixed: F9 inserts nothing, so the defect is invisible — but it would still
  fire F9 inside an IDE. The real fix is
  `org.freedesktop.portal.GlobalShortcuts`, which moves ownership of the binding
  to KDE's settings and makes the tray's Hotkey submenu wrong on Wayland. That
  trade is a conversation, not a decision to make alone.
- **Repetition loops** are repaired at the symptom (`collapse_loops`). The cause
  wants a `repetition_penalty` sweep, which is a measurement on the 393, so it is
  Kayes's call.
- **macOS is deliberately untouched** and the M2 has no Rust or cmake. Longest
  pole left.
- **Settings: microphone** is the last unbuilt Phase 5 item, and worth asking
  whether it is wanted at all — both desktops already have a one-click input
  picker.

## Things that will bite

- **`cargo tauri build` rewrites `app/src-tauri/Cargo.toml`**, expanding
  `tauri-build = "2"` into `{ version = "2", features = [] }`. Check `git diff`
  before committing after any bundle.
- **`makepkg` strips binaries by default**, which undoes the deliberate choice
  not to strip. `options=('!strip' '!debug')` is in the PKGBUILD; keep it.
- **The real PKGBUILD pins the *release* `.deb` hash**, so a locally rebuilt
  `.deb` will fail checksum validation. Use `makepkg --skipchecksums` for local
  installs; do not "fix" the PKGBUILD to match a local build.
- **`NO_STRIP=1` is an Arch-only workaround** for the AppImage and must not go
  in the CI workflow.
- **`app/src-tauri/capabilities/default.json` is per-window.** A window missing
  from that list receives no events, quietly — `invoke` still works, so the
  failure looks like something else.
- **`pill.css` is one namespace for two windows.** First-run styles stay scoped
  under `body.setup`.
- **`is_focused()` is not evidence that a window is passive** — it reports GTK's
  opinion, and KWin's differs. `xdotool getactivewindow` is the honest test.
- **Disk is above 90% full.** A 3 GB experiment is a real cost here.
- **`kotha` on PATH is the installed v0.1.0, not your build.** `kotha-bin` puts
  the published `.deb` at `/usr/bin/kotha` and the launcher runs it. Test with
  `./target/release/kotha` by path, and say so when asking Kayes to try a build.
- **Never wrap a build in `timeout`.** Nine minutes cold, and a signal landing
  mid-compile leaves 0-byte objects that the next build calls fresh. If a link
  fails with undefined `dnnl_*` symbols, run
  `ar tv target/release/build/onednn-src-*/out/lib/libdnnl.a | awk '$3==0'` —
  anything printed means delete that build dir and the matching
  `libonednn_src-*.rlib` and rebuild. Cargo cannot detect this on its own.
- **Kayes is sitting at this machine.** Windows you open appear in front of him
  and he will click them. Ask before building a theory around unexplained
  behaviour on this desktop.
- **`~/Documents/ASR` is read-only from here.** When a threshold needs real
  speech, take it from `~/Documents/ASR/bangla-asr/chunks` — 9,045 training
  clips — never the 393 evaluation set.
