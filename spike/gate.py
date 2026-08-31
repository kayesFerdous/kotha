#!/usr/bin/env python3
"""Phase 0's acceptance test: does the Rust spike agree with faster-whisper?

The baseline is a faster-whisper decode of 50 test utterances that already
exists in the paper repo, made on the same machine with the same parameters
(language=bn, beam_size=1, suppress_tokens=[]). Reusing it means the comparison
holds hardware, audio and decode settings fixed, and nothing needs re-decoding
in Python.

Reports three things:
  - how many outputs are byte-identical to faster-whisper's
  - CER of each engine against the human references
  - the spike's RTF

Read-only with respect to ~/Documents/ASR.

Usage:  python3 gate.py [--limit N]
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
SPIKE = Path(__file__).parent / "target/release/kotha-spike"
MODEL = Path(__file__).parent.parent / "models/whisper-medium-bn-en-cs-faster"
CONFIG = "int8, 6 threads"   # the baseline config to diff against
THREADS = "6"

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


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    rows = json.load(open(BENCH))[CONFIG]["rows"]
    if limit:
        rows = rows[:limit]
    base = {r["utt_id"]: r["hypothesis"] for r in rows}
    ref = {r["utt_id"]: r["reference"] for r in rows}

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
