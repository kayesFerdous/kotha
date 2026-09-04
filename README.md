# Kotha  কথা

**Offline dictation for the way Bangladeshis actually talk.**

Press a key anywhere, speak, and the text lands at your cursor — Bangla in
Bengali script, English in **Latin script**. `meeting`, not `মিটিং`.

Everything runs on your own machine. No account, no API key, no subscription.
After the first launch it never touches the network again.

---

## Why it exists

Every other Bengali speech recogniser writes English words phonetically, in
Bengali letters. `মিটিং`, `প্রেজেন্টেশন`, `ডেডলাইন`.

That text is a problem. You cannot search it. You cannot paste it into code or
a spreadsheet column. And it reads slowly, because your eye has to sound out a
word it already knows how to spell.

Kotha runs a model trained to keep English in Latin script, then fixes the
spelling before the words reach your cursor.

> **আজকে meeting আছে, deadline টা extend করতে হবে**

That is one sentence, written the way you would write it yourself.

## What it does

```
press F9  →  speak  →  press F9 again  →  the text is at your cursor
```

A small pill appears while you talk, showing that it is listening. Each pause
in your speech closes a chunk, and that chunk is transcribed while you keep
going — so the text arrives as you speak, not all at once at the end.

Everything else lives in the tray icon.

## Installing

**Arch Linux** and derivatives (Manjaro, EndeavourOS, CachyOS):

```bash
yay -S kotha-bin
```

**Debian, Ubuntu, Mint, Pop!_OS** — download `Kotha_0.1.0_amd64.deb` from the
[latest release](https://github.com/kayesFerdous/kotha/releases/latest), then:

```bash
sudo apt install ./Kotha_0.1.0_amd64.deb
```

Then launch **Kotha** from your application menu.

The first time it runs it asks before downloading the speech model — 778 MB,
once, and it shows you the progress. Nothing works until that finishes, and
nothing needs the network afterwards.

## Using it

**Press F9 to start talking. Press it again to stop.**

The tray icon holds everything else:

| Menu | What it does |
|---|---|
| **Dictate** | Same as pressing F9 |
| **Hotkey** | Change the key, if F9 clashes with something you use |
| **Text output** | How the text reaches you — see below |
| **Quit** | Closes it |

### Text output

Three choices, because desktops differ in what they allow:

- **Clipboard only** — Kotha copies the text and you press paste. Works
  everywhere, asks for no permissions. This is the default.
- **Paste at the cursor** — pastes it for you. On X11 this works everywhere. On
  Wayland it reaches older-style windows only.
- **Paste at the cursor (portal)** — pastes it for you in *every* window,
  including modern Wayland ones. Your desktop asks permission the first time.
  It only asks once.

**On KDE or GNOME Wayland, pick the portal one.** Say yes to the dialog and you
will not see it again.

## Requirements

- **Linux**, 64-bit. X11 or Wayland, KDE or GNOME
- **~900 MB of disk** — 100 MB for the app, 778 MB for the model
- **2 GB of free RAM** while dictating
- **A CPU from roughly 2017 or later.** No graphics card needed

macOS and Windows are not supported yet.

### How fast is it

On a six-core desktop CPU (Ryzen 5 5600G) Kotha transcribes at about **1.5×
real time** — ten seconds of speech takes about seven seconds to turn into
text — using around 1.4 GB of RAM. The model loads once, in about four seconds,
the first time you dictate after launching.

## The model

[`kayees/whisper-medium-bn-en-cs-faster`](https://huggingface.co/kayees/whisper-medium-bn-en-cs-faster)
— Whisper-medium, fine-tuned on around 28 hours of casual Bangladeshi speech
and quantised to int8 so it runs well on a CPU.

Kotha also carries a 50,000-word English dictionary and repairs the model's
English spelling before the text reaches you. It never touches Bengali text —
that is a guarantee built into how it works, not a rule it tries to follow. And
when it is not confident about a correction, it leaves the word alone: a
visible misspelling is easier to fix than a confident wrong word.

## Where your settings live

```
~/.config/app.kotha/settings.json      your hotkey and paste choice
~/.local/share/app.kotha/              the downloaded model
```

Deleting the model directory makes Kotha offer to download it again.

## Building from source

You do not need this to use Kotha — the packages above are prebuilt. But if you
want to:

```bash
git clone https://github.com/kayesFerdous/kotha.git
cd kotha
./setup.sh          # installs Rust and cmake, fetches the model
cargo tauri build
```

**Expect 15 to 30 minutes.** Kotha compiles its inference engine (CTranslate2
and oneDNN) from source, which is most of that time. Plug the laptop in —
`setup.sh` will refuse the heavy steps on battery.

Build dependencies on Arch:

```bash
sudo pacman -S rust cmake webkit2gtk-4.1 libayatana-appindicator alsa-lib
```

## Licence

MIT. See [LICENSE](LICENSE).
