#!/usr/bin/env python3

"""
split_audio.py
--------------
Splits m4a audio files into 4-6 second WAV chunks at 22050 Hz (LJSpeech standard),
cutting only at silence boundaries to avoid mid-word splits.

Usage:
    python split_audio.py --input_dir /path/to/m4a_files --output_dir /path/to/output
    python split_audio.py --input_dir ./audio --output_dir ./wavs --prefix yt_s1
"""

import os
import re
import json
import argparse
import subprocess
import tempfile
from pathlib import Path


# ── Configuration ────────────────────────────────────────────────────────────
SAMPLE_RATE      = 22050   # Hz  – LJSpeech / most TTS models expect this
CHANNELS         = 1       # mono
TARGET_MIN_SEC   = 3.0     # minimum chunk duration (seconds)
TARGET_MAX_SEC   = 8.0     # maximum chunk duration (seconds)

# silencedetect thresholds – tune if your recordings are noisy
SILENCE_DB       = "-18dB"  # amplitude threshold for "silence"
SILENCE_DURATION = 0.15     # minimum silence length to consider a boundary (s)
# ─────────────────────────────────────────────────────────────────────────────


def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def get_duration(path: str) -> float:
    """Return audio duration in seconds via ffprobe."""
    result = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", path
    ])
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def detect_silence(path: str) -> list[dict]:
    """
    Run ffmpeg silencedetect and return a list of silence intervals.
    Each entry: {"start": float, "end": float}
    """
    result = run([
        "ffmpeg", "-i", path,
        "-af", f"silencedetect=n={SILENCE_DB}:d={SILENCE_DURATION}",
        "-f", "null", "-"
    ], check=False)

    output = result.stderr
    silences = []
    current = {}

    for line in output.splitlines():
        m_start = re.search(r"silence_start:\s*([\d.]+)", line)
        m_end   = re.search(r"silence_end:\s*([\d.]+)", line)
        if m_start:
            current = {"start": float(m_start.group(1))}
        if m_end and current:
            current["end"] = float(m_end.group(1))
            silences.append(current)
            current = {}

    return silences


def silence_midpoints(silences: list[dict]) -> list[float]:
    """Return the midpoint of each silence interval (best cut point)."""
    return [(s["start"] + s["end"]) / 2.0 for s in silences]


def compute_cuts(duration: float, cut_points: list[float]) -> list[tuple[float, float]]:
    """
    Given candidate cut points (silence midpoints) and a total duration,
    greedily select cuts that produce segments between TARGET_MIN_SEC and
    TARGET_MAX_SEC seconds.

    Strategy:
      - Walk forward; once a segment would exceed TARGET_MAX_SEC, cut at the
        most recent valid silence ≥ TARGET_MIN_SEC.
      - If no such silence exists (long unbroken speech), force-cut at
        TARGET_MAX_SEC to avoid losing audio.
    """
    segments: list[tuple[float, float]] = []
    seg_start = 0.0
    candidates = sorted(cut_points)

    i = 0
    while seg_start < duration:
        seg_end = None

        # Find the last candidate that keeps length in [MIN, MAX]
        last_valid = None
        while i < len(candidates):
            t = candidates[i]
            length = t - seg_start
            if length < TARGET_MIN_SEC:
                i += 1
                continue
            if length <= TARGET_MAX_SEC:
                last_valid = t
                i += 1
            else:
                break  # past MAX

        if last_valid is not None:
            seg_end = last_valid
        else:
            # No silence in target window – force cut at MAX or end
            seg_end = min(seg_start + TARGET_MAX_SEC, duration)
            # Advance i past this forced cut
            while i < len(candidates) and candidates[i] <= seg_end:
                i += 1

        # Don't create a tiny tail segment – merge it into the previous one
        remaining = duration - seg_end
        if remaining < TARGET_MIN_SEC and seg_end < duration:
            seg_end = duration

        segments.append((round(seg_start, 4), round(seg_end, 4)))
        seg_start = seg_end

        if seg_start >= duration:
            break

    return segments


def extract_segment(src: str, start: float, end: float, dest: str) -> None:
    """Extract a WAV segment from src between start and end seconds."""
    duration = end - start
    run([
        "ffmpeg", "-y",
        "-i", src,
        "-ss", str(start),
        "-t",  str(duration),
        "-ar", str(SAMPLE_RATE),
        "-ac", str(CHANNELS),
        "-sample_fmt", "s16",   # 16-bit PCM – standard for TTS
        dest
    ])


def process_file(m4a_path: str, output_dir: str, prefix: str, counter: list[int]) -> list[str]:
    """
    Process a single m4a file → list of output WAV filenames (basenames only).
    counter is a mutable [int] so it persists across files.
    """
    print(f"\n► Processing: {m4a_path}")
    duration   = get_duration(m4a_path)
    silences   = detect_silence(m4a_path)
    midpoints  = silence_midpoints(silences)
    segments   = compute_cuts(duration, midpoints)

    print(f"  Duration : {duration:.1f}s  |  Silences found: {len(silences)}  |  Chunks: {len(segments)}")

    produced = []
    for start, end in segments:
        name     = f"{prefix}_{counter[0]:03d}.wav"
        out_path = os.path.join(output_dir, name)
        extract_segment(m4a_path, start, end, out_path)
        length = end - start
        print(f"  [{counter[0]:03d}]  {start:.2f}s → {end:.2f}s  ({length:.2f}s)  → {name}")
        produced.append(name)
        counter[0] += 1

    return produced


def main():
    parser = argparse.ArgumentParser(description="Split m4a files into TTS-ready WAV chunks.")
    parser.add_argument("--input_dir",  required=True, help="Folder containing .m4a files")
    parser.add_argument("--output_dir", required=True, help="Folder to write .wav chunks")
    parser.add_argument("--prefix",     default="yt_s1", help="Filename prefix (default: yt_s1)")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    m4a_files = sorted(input_dir.glob("*.m4a"))
    if not m4a_files:
        print(f"No .m4a files found in {input_dir}")
        return

    print(f"Found {len(m4a_files)} .m4a file(s) in {input_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Prefix      : {args.prefix}")
    print(f"Sample rate : {SAMPLE_RATE} Hz  |  Channels: {CHANNELS}  |  Bit depth: 16-bit PCM")
    print(f"Target chunk: {TARGET_MIN_SEC}–{TARGET_MAX_SEC} s  |  Silence threshold: {SILENCE_DB}")

    counter = [0]   # mutable so process_file can increment it
    all_files = []

    for m4a in m4a_files:
        produced = process_file(str(m4a), str(output_dir), args.prefix, counter)
        all_files.extend(produced)

    print(f"\n✔ Done. {len(all_files)} WAV files written to {output_dir}/")


if __name__ == "__main__":
    main()