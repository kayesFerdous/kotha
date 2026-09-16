#!/usr/bin/env bash
#
# Kotha — install the prebuilt app.
#
#   curl -fsSL https://raw.githubusercontent.com/kayesFerdous/kotha/v0.1.0/packaging/install.sh | bash
#
# Works on macOS (Apple Silicon and Intel) and on Linux (Arch, Debian and
# everything else, in that order of comfort). Re-running it upgrades in place.
#
#   --appimage   Linux: skip pacman and apt, install the AppImage under $HOME
#   --help
#
# --------------------------------------------------------------------------
# Why this exists, when the release page already has installers
#
# On macOS, because quarantine comes from the *browser*, not from the file. A
# dmg downloaded in Safari or Chrome gets a com.apple.quarantine attribute, and
# Kotha is signed ad-hoc rather than notarised, so Gatekeeper refuses it —
# "Apple could not verify Kotha is free of malware" — and the only way through
# is a trip to System Settings › Privacy & Security › Open Anyway. The same dmg
# fetched with curl carries no such attribute and simply opens.
#
# That is not a bypass. The dialog stands in for a check, and this script does
# the check instead, twice and more precisely: the download is compared against
# a SHA-256 pinned below, and the app bundle is put through `codesign --verify
# --deep --strict` before it is copied anywhere.
#
# On Arch, because there is no package. `kotha-bin` is not in the AUR, so this
# script stands in for `yay -S kotha-bin`: it hands the release .deb to the
# same PKGBUILD the AUR would have carried, and pacman ends up owning the files
# — `pacman -Qi kotha-bin` describes them, `pacman -R kotha-bin` removes them.
#
# On Debian and Ubuntu it saves nothing but a page visit; `sudo apt install
# ./Kotha_0.1.0_amd64.deb` does the same job.
#
# --------------------------------------------------------------------------
# To update for a release: bump VERSION, then replace the four sums with the
# output of `shasum -a 256 Kotha_*` over the files from the release, and fix
# packaging/PKGBUILD's sha256sums to match SHA_DEB — the Arch route refuses to
# run when those two disagree.
#
# The sums are pinned in the repository rather than fetched alongside the
# download on purpose. A checksum that travels with the file it describes
# proves only that the file arrived whole; one that lives in git is the same
# for everybody, shows up in a diff when it changes, and is the reason a
# swapped asset would be caught here rather than installed.

set -euo pipefail

VERSION=0.1.0
REPO=kayesFerdous/kotha
BASE="https://github.com/$REPO/releases/download/v$VERSION"

SHA_DMG_ARM64=b8133305f98b201f4653a9c9c533d9d75196657c87da3e09696f23ba21de4550
SHA_DMG_X64=c76dcfeea697499d0d0ebbaef8be24909d8d6440d1afa36943b7899a857d9742
SHA_DEB=48bdba9ddbe3dbb0d94ee775305ec849e832a90a965ca4396b3b996134c253fb
SHA_APPIMAGE=773c09967a0f8a2ca72b73cb8578d9c7e4373cff60d7a8dfc145543ef6e134bd

# Spelled out rather than sed'd out of the header comment the way setup.sh
# does it: this script's normal home is a pipe, where there is no
# ${BASH_SOURCE[0]} to read back.
usage() {
  cat <<'EOF'
Kotha — install the prebuilt app.

  curl -fsSL https://raw.githubusercontent.com/kayesFerdous/kotha/v0.1.0/packaging/install.sh | bash

Works on macOS (Apple Silicon and Intel) and on Linux (Arch, Debian and
everything else, in that order of comfort). Re-running it upgrades in place.

  --appimage   Linux: skip pacman and apt, install the AppImage under $HOME
  --help
EOF
}

APPIMAGE=0
for arg in "$@"; do
  case "$arg" in
    --appimage) APPIMAGE=1 ;;
    -h|--help)  usage; exit 0 ;;
    *)          echo "unknown flag: $arg" >&2; usage >&2; exit 2 ;;
  esac
done

say()  { printf '\n\033[1m▸ %s\033[0m\n' "$*"; }
ok()   { printf '  \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

TMP="$(mktemp -d)"
MOUNT=""
cleanup() {
  [[ -n $MOUNT ]] && hdiutil detach "$MOUNT" -quiet 2>/dev/null || true
  rm -rf "$TMP"
}
trap cleanup EXIT

# -------------------------------------------------------------- the download

fetch() { # url dest
  if command -v curl >/dev/null; then
    curl -fL --progress-bar "$1" -o "$2"
  elif command -v wget >/dev/null; then
    wget -q --show-progress -O "$2" "$1"
  else
    die "Neither curl nor wget is installed."
  fi
}

sha256_of() {
  if command -v shasum >/dev/null; then shasum -a 256 "$1" | cut -d' ' -f1
  elif command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else die "No shasum or sha256sum on this machine — cannot check the download."
  fi
}

# Loudly, with both hashes shown. A mismatch here is either a bad download or
# something worse, and the user is the one who has to decide which.
verify() { # file expected
  local got
  got="$(sha256_of "$1")"
  [[ $got == "$2" ]] || die "Checksum mismatch on $(basename "$1").

    expected  $2
    got       $got

    Do not install this. Try again on a different network, and if it still
    fails, say so at https://github.com/$REPO/issues — do not work around it."
  ok "SHA-256 verified: $got"
}

# ------------------------------------------------------------------- macOS

install_macos() {
  local file sum
  # Not `uname -m`: under Rosetta that answers x86_64 on an M-series Mac, and
  # this would then install the Intel build on Apple Silicon and run it
  # translated, at a fraction of the speed, for the life of the machine.
  if [[ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" == 1 ]]; then
    file="Kotha_${VERSION}_aarch64.dmg"; sum=$SHA_DMG_ARM64
    ok "Apple Silicon"
  else
    file="Kotha_${VERSION}_x64.dmg"; sum=$SHA_DMG_X64
    ok "Intel"
  fi

  say "Downloading $file"
  fetch "$BASE/$file" "$TMP/$file"
  verify "$TMP/$file" "$sum"

  say "Checking the signature"
  MOUNT="$TMP/mnt"
  hdiutil attach -nobrowse -quiet -mountpoint "$MOUNT" "$TMP/$file"
  [[ -d "$MOUNT/Kotha.app" ]] || die "No Kotha.app inside $file."
  codesign --verify --deep --strict "$MOUNT/Kotha.app" ||
    die "The app bundle's signature does not verify. Not installing it."
  ok "codesign: valid on disk"

  local dest=/Applications
  if [[ ! -w $dest ]]; then
    dest="$HOME/Applications"
    mkdir -p "$dest"
    warn "/Applications is not writable — installing to $dest instead"
  fi

  # An app being replaced underneath itself is how a half-written bundle
  # happens. Kotha lives in the menu bar with no window to close, so quitting
  # it is not something the user can be asked to do reliably.
  if pgrep -x kotha >/dev/null; then
    pkill -x kotha && ok "quit the running Kotha"
    sleep 1
  fi

  say "Installing to $dest/Kotha.app"
  if [[ -d "$dest/Kotha.app" ]]; then
    [[ -x "$dest/Kotha.app/Contents/MacOS/kotha" ]] ||
      die "$dest/Kotha.app exists but does not look like Kotha. Move it aside yourself."
    rm -rf "$dest/Kotha.app"
    ok "replaced the previous install"
  fi
  # ditto, not cp: it carries the extended attributes and the signed resource
  # layout across intact, which cp -R does not promise.
  ditto "$MOUNT/Kotha.app" "$dest/Kotha.app"
  hdiutil detach "$MOUNT" -quiet && MOUNT=""
  ok "installed"

  say "Next"
  cat <<EOF
  Launch it:   open -a "$dest/Kotha.app"
               or Spotlight → Kotha. It lives in the menu bar, not the Dock.

  First launch offers the speech model — 778 MB, once. Nothing works until
  that finishes, and nothing needs the network afterwards.

  Then press ⌥⇧D — Option-Shift-D — talk, and press it again. F9 is the
  default everywhere else, but on a Mac F9 is Next Track.

  macOS asks for the microphone the first time you dictate. Say yes; Kotha
  hears silence without it and cannot tell you why.

  Only if you switch Text output to "Paste at the cursor": macOS calls that
  permission Accessibility, and it has to be granted by hand.

    open "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"

  Switch Kotha on there and dictate again — it notices without a restart.
EOF
}

# ------------------------------------------------------------------- Linux

sudo_prefix() {
  if [[ $EUID -eq 0 ]]; then echo ""
  elif command -v sudo >/dev/null; then echo "sudo"
  else die "This needs root and sudo is not installed. Re-run it as root."
  fi
}

install_arch() {
  if ! command -v makepkg >/dev/null; then
    warn "makepkg is missing, so pacman cannot own this install."
    warn "For the packaged route: sudo pacman -S --needed base-devel"
    warn "Falling back to the AppImage."
    install_appimage
    return
  fi
  [[ $EUID -ne 0 ]] ||
    die "makepkg refuses to run as root. Run this as your normal user — it
    asks for sudo itself, at the one step that needs it."

  local dir="$TMP/pkg"
  mkdir -p "$dir"
  # A checkout next to this script is the same recipe; use it rather than
  # fetching, so someone testing a change tests the change.
  if [[ -f "${BASH_SOURCE[0]:-}" && -f "$(dirname "${BASH_SOURCE[0]}")/PKGBUILD" ]]; then
    cp "$(dirname "${BASH_SOURCE[0]}")/PKGBUILD" "$dir/PKGBUILD"
    ok "using the PKGBUILD beside this script"
  else
    say "Fetching the PKGBUILD"
    fetch "https://raw.githubusercontent.com/$REPO/v$VERSION/packaging/PKGBUILD" "$dir/PKGBUILD"
  fi

  # The two files pin the same .deb and are edited by hand at different times.
  # When they disagree, makepkg's own failure says "validity check failed" and
  # names neither file, so it gets named here instead.
  grep -q "$SHA_DEB" "$dir/PKGBUILD" ||
    die "The PKGBUILD does not carry the checksum this script expects:

      $SHA_DEB

    One of the two is stale. Until they agree, install the AppImage instead:
      bash install.sh --appimage"
  grep -q "pkgver=$VERSION" "$dir/PKGBUILD" ||
    die "The PKGBUILD is not for version $VERSION. One of the two is stale."

  say "Building the package (makepkg will ask for sudo to install it)"
  (cd "$dir" && makepkg -si --noconfirm)
  ok "pacman owns it now — pacman -R kotha-bin removes it"
  linux_next "Kotha"
}

install_deb() {
  local file="Kotha_${VERSION}_amd64.deb"
  local sudo; sudo="$(sudo_prefix)"

  say "Downloading $file"
  fetch "$BASE/$file" "$TMP/$file"
  verify "$TMP/$file" "$SHA_DEB"

  say "Installing"
  # apt, not dpkg, so that libgomp1 and the webkit and tray libraries come
  # along. dpkg is the fallback for a Debian-ish system without apt-get.
  if command -v apt-get >/dev/null; then
    $sudo apt-get install -y "$TMP/$file"
  else
    $sudo dpkg -i "$TMP/$file" || die "dpkg could not satisfy the dependencies.
    Install them and re-run: libgomp1 libayatana-appindicator3-1
    libwebkit2gtk-4.1-0 libgtk-3-0"
  fi
  ok "installed — apt remove kotha removes it"
  linux_next "Kotha"
}

# The route for everything else: Fedora, openSUSE, Void, an Arch without
# base-devel. No package manager is involved, so it all goes under $HOME and
# comes out again with one rm -rf.
install_appimage() {
  local file="Kotha_${VERSION}_amd64.AppImage"
  local home="$HOME/.local/lib/kotha"

  say "Downloading $file"
  fetch "$BASE/$file" "$TMP/$file"
  verify "$TMP/$file" "$SHA_APPIMAGE"
  chmod +x "$TMP/$file"

  # Unpacked rather than left as one file, and not because unpacking is
  # tidier: a packed AppImage mounts itself with FUSE 2, which Arch and recent
  # Fedora do not install by default, and the failure is a silent non-start.
  # Extraction is the AppImage runtime's own and needs nothing.
  say "Unpacking to $home"
  (cd "$TMP" && "./$file" --appimage-extract >/dev/null)
  [[ -x "$TMP/squashfs-root/AppRun" ]] || die "The AppImage did not unpack."
  rm -rf "$home"
  mkdir -p "$(dirname "$home")"
  mv "$TMP/squashfs-root" "$home"
  ok "unpacked"

  mkdir -p "$HOME/.local/bin" "$HOME/.local/share/applications"
  ln -sf "$home/AppRun" "$HOME/.local/bin/kotha"

  # Its own file rather than the bundled one, because Exec has to be an
  # absolute path here: nothing has put AppRun on the system PATH.
  cat > "$HOME/.local/share/applications/kotha.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Kotha
Comment=Offline Bengali-English dictation
Exec=$home/AppRun
Icon=kotha
Categories=Utility;
StartupWMClass=kotha
Terminal=false
EOF

  # The same sizes the .deb ships, in the same places, so the menu and the
  # window manager both find the icon.
  if [[ -d "$home/usr/share/icons/hicolor" ]]; then
    mkdir -p "$HOME/.local/share/icons"
    cp -r "$home/usr/share/icons/hicolor" "$HOME/.local/share/icons/"
  fi
  command -v update-desktop-database >/dev/null &&
    update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
  ok "menu entry added"

  case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) warn "$HOME/.local/bin is not on your PATH, so the 'kotha' command will
    not be found. The menu entry works regardless." ;;
  esac

  say "Uninstall, when you want to"
  echo "  rm -rf $home $HOME/.local/bin/kotha \\"
  echo "         $HOME/.local/share/applications/kotha.desktop"
  linux_next "Kotha from your application menu"
}

linux_next() {
  say "Next"
  cat <<EOF
  Launch $1.

  First launch offers the speech model — 778 MB, once. Nothing works until
  that finishes, and nothing needs the network afterwards.

  Then press F9, talk, press F9 again. The tray icon has Settings.

  On KDE or GNOME Wayland, set Text output to "Paste at the cursor (portal)"
  and say yes to the dialog. It asks once. The other paste mode reaches
  older-style windows only there.
EOF
}

# ------------------------------------------------------------------- picking

case "$(uname -s)" in
  Darwin)
    say "Kotha $VERSION for macOS"
    [[ $APPIMAGE -eq 0 ]] || die "--appimage is a Linux option."
    install_macos
    ;;
  Linux)
    say "Kotha $VERSION for Linux"
    [[ "$(uname -m)" == x86_64 ]] ||
      die "Only x86_64 Linux is built. You are on $(uname -m); build from
    source with ./setup.sh — expect 15 to 30 minutes of compiling."

    id=""; like=""
    if [[ -r /etc/os-release ]]; then
      # shellcheck disable=SC1091
      . /etc/os-release
      id="${ID:-}"; like="${ID_LIKE:-}"
    fi
    if [[ $APPIMAGE -eq 1 ]]; then
      install_appimage
    else
      case " $id $like " in
        *" arch "*|*" archlinux "*) ok "${PRETTY_NAME:-Arch} — the pacman route"; install_arch ;;
        *" debian "*|*" ubuntu "*)  ok "${PRETTY_NAME:-Debian} — the apt route";  install_deb ;;
        *)                          ok "${PRETTY_NAME:-unknown distribution} — no package for it, using the AppImage"
                                    install_appimage ;;
      esac
    fi
    ;;
  *)
    die "This script handles macOS and Linux. On Windows, download
    Kotha_${VERSION}_x64_en-US.msi from
    https://github.com/$REPO/releases/latest — SmartScreen will warn once,
    choose More info, then Run anyway."
    ;;
esac
