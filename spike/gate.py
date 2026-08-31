#!/usr/bin/env python3
"""Phase 0's acceptance test: does the Rust spike agree with faster-whisper?

Two baselines, and the difference between them is the whole point.

--baseline stored (default)
    The decode already in results/cpu_bench.json. It was made with
    `transcribe(language="bn", beam_size=1, suppress_tokens=[])` and
    faster-whisper's defaults for everything else — which include a temperature
    fallback ladder that SAMPLES when greedy output trips the compression-ratio
    check, and without_timestamps=False. So it is a stochastic multi-temperature
    decode using a different prompt than the spike's single greedy pass. The two
    cannot agree byte-for-byte, and this baseline is not even reproducible
    against itself. Useful only as a rough accuracy reference.

--baseline matched
    Re-decodes the same files with settings that actually match the spike:
    temperature=0.0, without_timestamps=True, condition_on_previous_text=False.
    This is the honest byte-level comparison. Needs faster-whisper, so run it
    with the venv that has it:

        ~/Documents/ASR/bangla-asr-test/.venv/bin/python gate.py --baseline matched

    The decode is cached in matched_baseline.json; delete that to redo it.

Reports how many outputs are byte-identical, each engine's CER against the
human references, and the spike's RTF.

Read-only with respect to ~/Documents/ASR.

Usage:  python3 gate.py [--limit N] [--baseline stored|matched]
"""
import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

ASR = Path.home() / "Documents/ASR"
BENCH = ASR / "results/cpu_bench.json"
CHUNKS = ASR / "bangla-asr-test/chunks/test"
# The workspace shares one target directory at the repo root, so
# CTranslate2 is compiled once rather than once per crate.
SPIKE = Path(__file__).parent.parent / "target/release/kotha-spike"
MODEL = Path(__file__).parent.parent / "models/whisper-medium-bn-en-cs-faster"
CONFIG = "int8, 6 threads"   # the stored-baseline config to diff against
THREADS = "6"
MATCHED_CACHE = Path(__file__).parent / "matched_baseline.json"

# The spike's decode policy, mirrored exactly. Anything left to faster-whisper's
# defaults here would reintroduce the mismatch this mode exists to remove.
MATCHED_ARGS = dict(
    language="bn",
    beam_size=1,
    suppress_tokens=[],
    temperature=0.0,
    without_timestamps=True,
    condition_on_previous_text=False,
)

norm = lambda s: unicodedata.normalize("NFC", s).strip()


def cer(hyp, ref):
    """Levenshtein distance in characters, plus reference length."""
    hyp, ref = norm(hyp), norm(ref)
    prev_row = list(range(len(ref) + 1))
    for i, c in enumerate(hyp, 1):
        row = [i]
        for j, rc in enumerate(ref, 1):
            row.append(min(prev_row[j] + 1, row[j - 1] + 1, prev_row[j - 1] + (c != rc)))
        prev_row = row
    return prev_row[-1], len(ref)


def matched_baseline(utt_ids):
    """Decode with faster-whisper using the spike's exact settings."""
    if MATCHED_CACHE.exists():
        cached = json.loads(MATCHED_CACHE.read_text())
        if set(cached) >= set(utt_ids):
            print(f"using cached matched baseline ({MATCHED_CACHE.name})")
            return {u: cached[u] for u in utt_ids}

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit(
            "--baseline matched needs faster-whisper. Run this script with the "
            "venv that has it:\n"
            "  ~/Documents/ASR/bangla-asr-test/.venv/bin/python gate.py "
            "--baseline matched"
        )

    print(f"decoding {len(utt_ids)} utterances with faster-whisper ...")
    model = WhisperModel(str(MODEL), device="cpu", compute_type="int8",
                         cpu_threads=int(THREADS))
    out = {}
    for i, u in enumerate(utt_ids, 1):
        seg, _ = model.transcribe(str(CHUNKS / u), **MATCHED_ARGS)
        out[u] = "".join(s.text for s in seg).strip()
        print(f"  {i}/{len(utt_ids)}", end="\r", flush=True)
    print()
    MATCHED_CACHE.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    which = "stored"
    if "--baseline" in sys.argv:
        which = sys.argv[sys.argv.index("--baseline") + 1]
    if which not in ("stored", "matched"):
        sys.exit("--baseline must be 'stored' or 'matched'")

    rows = json.load(open(BENCH))[CONFIG]["rows"]
    if limit:
        rows = rows[:limit]
    ref = {r["utt_id"]: r["reference"] for r in rows}
    if which == "matched":
        base = matched_baseline(list(ref))
    else:
        base = {r["utt_id"]: r["hypothesis"] for r in rows}

    missing = [u for u in base if not (CHUNKS / u).exists()]
    if missing:
        sys.exit(f"{len(missing)} test WAV(s) not found under {CHUNKS}")

    proc = subprocess.run(
        [str(SPIKE), str(MODEL), *[str(CHUNKS / u) for u in base]],
        capture_output=True, text=True,
        env={**os.environ, "KOTHA_THREADS": THREADS},
    )
    Path("gate_out.txt").write_text(proc.stdout + "\n[stderr]\n" + proc.stderr)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        sys.exit("spike failed — full output in gate_out.txt")

    # "── name.wav (...)" then the transcript on the next non-empty line
    got, cur = {}, None
    for line in proc.stdout.splitlines():
        m = re.match(r"── (\S+\.wav)", line)
        if m:
            cur = m.group(1)
        elif cur and line.strip():
            got[cur], cur = line.strip(), None

    identical = [u for u in base if u in got and norm(got[u]) == norm(base[u])]

    print(f"\nbaseline   {which}")
    print(f"decoded    {len(got)}/{len(base)}")
    print(f"identical  {len(identical)}/{len(base)}  (byte-for-byte vs faster-whisper)")
    for label, hyps in (("faster-whisper", base), ("rust spike", got)):
        errs = total = 0
        for u in base:
            if u in hyps:
                e, t = cer(hyps[u], ref[u])
                errs, total = errs + e, total + t
        print(f"CER {label:15s} {errs / total:.4f}")
    for line in proc.stdout.splitlines():
        if line.startswith("RTF"):
            print(line)

    print("\ndiffering:")
    for u in base:
        if u in got and u not in identical:
            print(f"\n{u}\n  fw   {base[u]}\n  rust {got[u]}")


if __name__ == "__main__":
    main()
