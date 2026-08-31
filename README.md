# Kotha  কথা

Offline dictation for the way Bangladeshis actually talk.

Press a hotkey anywhere, speak, and the text lands at your cursor — Bangla in
Bengali script, English in **Latin script**. `meeting`, not `মিটিং`.

Everything runs on your own CPU. Nothing is uploaded, there is no account and no
API key, and after the first launch it never needs the network.

> **Status: early.** The inference engine is proven — the Rust build decodes at
> 1.57x real time on a Ryzen 5600G, matching faster-whisper's speed, with
> feature extraction verified bit-exact against Whisper's reference. Nothing
> above that layer exists yet. See [`PLAN.md`](PLAN.md) for where things stand.

---

## Why it exists

Every Bengali speech recogniser writes English phonetically in Bengali script.
`মিটিং`, `প্রেজেন্টেশন`, `ডেডলাইন`. That text is unsearchable, awkward to copy,
and slow to read. Kotha runs a model fine-tuned to keep English in Latin script,
then repairs its spelling before the words reach your cursor.

## How it works

```
hotkey  →  mic 16 kHz  →  split on silence  →  transcribe  →  fix English  →  paste
```

Each pause in your speech closes a chunk, and that chunk is transcribed while
you keep talking — so text appears as you speak rather than all at once at the
end.

## Requirements

- macOS, Windows or Linux
- ~800 MB of disk for the model, downloaded on first launch
- 2 GB of free RAM while dictating

## Building it

```bash
./setup.sh
```

Installs Rust and cmake and fetches the model. It refuses the heavy steps while
the machine is on battery — plug in first.

Then run the Phase 0 check, which proves the inference engine works before
anything else gets built:

```bash
cd spike && cargo run --release -- ../models/whisper-medium-bn-en-cs-faster ../samples/*.wav
```

## The model

[`kayees/whisper-medium-bn-en-cs-faster`](https://huggingface.co/kayees/whisper-medium-bn-en-cs-faster)
— Whisper-medium fine-tuned on ~28 hours of casual Bangladeshi YouTube speech,
quantised to int8 for CPU. It runs at about 1.5× real time on a six-core desktop
CPU in roughly 1.4 GB of RAM.

## Layout

| Path | What |
|---|---|
| `CLAUDE.md` | Project rules and context, for humans and AI sessions alike |
| `PLAN.md` | The eight phases, current state, and open decisions |
| `setup.sh` | One-time machine setup |
| `spike/` | Phase 0 — the smallest program that proves the engine works |
| `models/` | Model weights. Downloaded, never committed |
