#!/usr/bin/env python3
"""Verify the spike's mel spectrogram against Whisper's reference definition.

Feature extraction is the only part of the inference path we implement
ourselves — ct2rs's own version is wrong in three ways (see main.rs) — so it is
the part that needs proof rather than trust. This recomputes the mel from
Whisper's published formula in numpy and diffs it against what the Rust binary
produced for the same WAV.

Needs only numpy: the Slaney mel filterbank is built here rather than pulled
from librosa, since Whisper's filters are exactly librosa's
mel(sr=16000, n_fft=400, n_mels=80, htk=False, norm='slaney').

Usage:  python3 check_mel.py <file.wav>
"""
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

SR, N_FFT, HOP, N_MELS, N_FRAMES = 16000, 400, 160, 80, 3000
N_SAMPLES = N_FRAMES * HOP
SPIKE = Path(__file__).parent / "target/release/kotha-spike"
MODEL = Path(__file__).parent.parent / "models/whisper-medium-bn-en-cs-faster"


def hz_to_mel(hz):
    """Slaney scale: linear below 1 kHz, logarithmic above."""
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    hz = np.asarray(hz, dtype=float)
    # np.where evaluates both branches, so keep the log's argument positive.
    safe = np.maximum(hz, 1e-10)
    return np.where(hz < min_log_hz, hz / f_sp,
                    min_log_mel + np.log(safe / min_log_hz) / logstep)


def mel_to_hz(mel):
    f_sp, min_log_hz = 200.0 / 3, 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    mel = np.asarray(mel, dtype=float)
    return np.where(mel < min_log_mel, f_sp * mel,
                    min_log_hz * np.exp(logstep * (mel - min_log_mel)))


def mel_filters():
    fftfreqs = np.fft.rfftfreq(N_FFT, 1.0 / SR)                    # 201 bins
    mel_f = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(SR / 2), N_MELS + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fftfreqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    # Slaney normalisation: equal area per filter.
    weights *= (2.0 / (mel_f[2:] - mel_f[:-2]))[:, None]
    return weights


def reference_log_mel(samples):
    audio = np.zeros(N_SAMPLES, dtype=np.float64)
    audio[:min(len(samples), N_SAMPLES)] = samples[:N_SAMPLES]

    # torch.stft(center=True) reflection-pads by n_fft // 2 before framing.
    padded = np.pad(audio, N_FFT // 2, mode="reflect")
    # Periodic Hann, matching torch.hann_window's default.
    window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT))

    frames = np.stack([padded[j * HOP: j * HOP + N_FFT] * window
                       for j in range(N_FRAMES)], axis=1)
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=0)) ** 2      # (201, 3000)

    logspec = np.log10(np.maximum(mel_filters() @ power, 1e-10))
    logspec = np.maximum(logspec, logspec.max() - 8.0)
    return ((logspec + 4.0) / 4.0).astype(np.float32)


def read_wav(path):
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR, f"{path} is not {SR} Hz"
        assert w.getsampwidth() == 2, "expected 16-bit PCM"
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        if w.getnchannels() > 1:
            raw = raw.reshape(-1, w.getnchannels()).mean(axis=1)
        return raw.astype(np.float64) / 32768.0


def main():
    wav = Path(sys.argv[1])
    dump = Path("mel_dump.bin")
    subprocess.run([str(SPIKE), str(MODEL), str(wav)],
                   capture_output=True, text=True, check=True,
                   env={**__import__("os").environ, "KOTHA_DUMP_MEL": str(dump)})

    got = np.frombuffer(dump.read_bytes(), dtype="<f4").reshape(N_MELS, N_FRAMES)
    want = reference_log_mel(read_wav(wav))

    diff = np.abs(got - want)
    print(f"shape      {got.shape}")
    print(f"max  |diff| {diff.max():.3e}")
    print(f"mean |diff| {diff.mean():.3e}")
    print(f"range      rust [{got.min():.4f}, {got.max():.4f}]  "
          f"ref [{want.min():.4f}, {want.max():.4f}]")
    # float32 mel values are O(1); 1e-4 is far above fp noise but far below
    # anything that would shift a decode.
    ok = diff.max() < 1e-4
    print("\nPASS — matches Whisper's reference mel" if ok else
          "\nFAIL — feature extraction diverges from Whisper")
    dump.unlink(missing_ok=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
