#!/usr/bin/env python3
"""
split_audio.py  (v2 – word-timestamp splitting) (Did not work well on continuous speech.)
------------------------------------------------
Splits m4a files into 4-6 second WAV chunks at 22050 Hz (LJSpeech standard).

Instead of silence detection (which fails on continuous speech), this script
uses Whisper's word-level timestamps to find natural break points:
  1. Transcribe the full file with word timestamps
  2. Prefer cuts after sentence-ending punctuation  (. ! ?)
  3. Fall back to clause punctuation                 (, ; :)
  4. Fall back to the largest inter-word gap in the window

Cuts always happen BETWEEN words — never mid-word.

Requirements:
    pip install openai-whisper
    ffmpeg must be installed and on PATH

Usage:
    python split_audio.py --input_dir ./m4a --output_dir ./wavs
    python split_audio.py --input_dir ./m4a --output_dir ./wavs --prefix yt_s1 --model small
"""

import os
import json
import argparse
import subprocess
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 22050   # Hz  — LJSpeech / Coqui / VITS standard
CHANNELS     = 1       # mono
BIT_DEPTH    = "s16"   # 16-bit PCM

TARGET_MIN   = 4.0     # seconds — don't cut before this
TARGET_MAX   = 6.0     # seconds — must cut by this

SENT_END     = set(".!?")   # strong boundary
CLAUSE_END   = set(",;:")   # weaker boundary
# ─────────────────────────────────────────────────────────────────────────────


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def get_duration(path: str) -> float:
    r = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "json", path])
    return float(json.loads(r.stdout)["format"]["duration"])


def to_wav(src: str, dst: str):
    """Convert any audio file → 22050 Hz mono 16-bit WAV."""
    run(["ffmpeg", "-y", "-i", src,
         "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
         "-sample_fmt", BIT_DEPTH, dst])


def get_words(wav_path: str, model, language):
    """
    Transcribe with Whisper word_timestamps=True.
    Returns list of {word, start, end}.
    """
    opts = dict(word_timestamps=True, fp16=False)
    if language:
        opts["language"] = language
    result = model.transcribe(wav_path, **opts)

    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            text = w["word"].strip()
            if text:
                words.append({"word": text, "start": w["start"], "end": w["end"]})
    return words


def boundary_score(word_text: str) -> int:
    """Score how good a cut AFTER this word is. Higher = better."""
    last = word_text.rstrip()[-1] if word_text.strip() else ""
    if last in SENT_END:
        return 3
    if last in CLAUSE_END:
        return 2
    return 1


def inter_word_gap(words: list, idx: int) -> float:
    """Gap in seconds between word[idx] and word[idx+1]."""
    if idx + 1 >= len(words):
        return 1.0
    return max(0.0, words[idx + 1]["start"] - words[idx]["end"])


def compute_cuts(words: list, total_dur: float) -> list:
    """
    Build (start, end) segments of TARGET_MIN–TARGET_MAX seconds,
    always cutting between words.
    """
    segments   = []
    seg_start  = 0.0
    i          = 0                  # index of first word not yet committed

    while i < len(words):
        # Advance until we've accumulated at least TARGET_MIN seconds
        if words[i]["end"] - seg_start < TARGET_MIN:
            i += 1
            continue

        # Collect every word whose END falls within [MIN, MAX] from seg_start
        candidates = []
        j = i
        while j < len(words) and words[j]["end"] - seg_start <= TARGET_MAX:
            candidates.append((
                j,
                boundary_score(words[j]["word"]),
                inter_word_gap(words, j),
                words[j]["end"],
            ))
            j += 1

        if candidates:
            # Best = highest boundary score, then largest gap as tiebreak
            best_j, _, _, cut_time = max(candidates, key=lambda c: (c[1], c[2]))
            next_i = best_j + 1
        else:
            # No word ended inside the window — force cut at the last word before MAX
            fallback = None
            for k in range(i, len(words)):
                if words[k]["end"] - seg_start <= TARGET_MAX:
                    fallback = k
                else:
                    break
            if fallback is not None:
                cut_time = words[fallback]["end"]
                next_i   = fallback + 1
            else:
                # Single word wider than MAX — cut after it anyway
                cut_time = words[i]["end"]
                next_i   = i + 1

        segments.append((round(seg_start, 4), round(cut_time, 4)))
        seg_start = cut_time
        i         = next_i

    # Handle remaining audio after the last cut
    if total_dur - seg_start > 0.1:
        tail = total_dur - seg_start
        if tail < TARGET_MIN and segments:
            # Merge tiny tail into last segment
            s, _ = segments[-1]
            segments[-1] = (s, round(total_dur, 4))
        else:
            segments.append((round(seg_start, 4), round(total_dur, 4)))

    return segments


def extract_wav(src: str, start: float, end: float, dest: str):
    run([
        "ffmpeg", "-y", "-i", src,
        "-ss", str(start), "-t", str(end - start),
        "-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS),
        "-sample_fmt", BIT_DEPTH, dest,
    ])


def process_file(m4a_path, output_dir, prefix, counter, model, language):
    name = Path(m4a_path).name
    print(f"\n► {name}")

    tmp = str(Path(output_dir) / "__tmp__.wav")
    print("  Converting …")
    to_wav(str(m4a_path), tmp)
    duration = get_duration(tmp)
    print(f"  Duration : {duration:.1f}s")

    print("  Transcribing (word timestamps) …")
    words = get_words(tmp, model, language)
    print(f"  Words    : {len(words)}")

    if not words:
        print("  ⚠  No words found — skipping.")
        os.remove(tmp)
        return []

    segments = compute_cuts(words, duration)
    print(f"  Chunks   : {len(segments)}")

    produced = []
    for start, end in segments:
        out_name = f"{prefix}_{counter[0]:03d}.wav"
        out_path = os.path.join(output_dir, out_name)
        extract_wav(tmp, start, end, out_path)

        seg_words = [
            w["word"] for w in words
            if w["start"] >= start - 0.05 and w["end"] <= end + 0.05
        ]
        preview = " ".join(seg_words)[:70]
        print(f"  [{counter[0]:03d}]  {start:.2f}–{end:.2f}s  ({end-start:.2f}s)  {preview}")
        produced.append(out_name)
        counter[0] += 1

    os.remove(tmp)
    return produced


def main():
    ap = argparse.ArgumentParser(
        description="Split m4a audio into TTS-ready WAV chunks using Whisper word timestamps."
    )
    ap.add_argument("--input_dir",  required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--prefix",     default="yt_s1")
    ap.add_argument("--model",      default="base",
                    choices=["tiny", "base", "small", "medium",
                             "large", "large-v2", "large-v3"])
    ap.add_argument("--language",   default=None,
                    help="Force language, e.g. 'en'. Auto-detected if omitted.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    m4a_files = sorted(Path(args.input_dir).glob("*.m4a"))
    if not m4a_files:
        print("No .m4a files found.")
        return

    try:
        import whisper
    except ImportError:
        print("Run:  pip install openai-whisper")
        return

    print(f"Loading Whisper '{args.model}' …")
    model = whisper.load_model(args.model)

    print(f"\nFiles  : {len(m4a_files)}")
    print(f"Output : {out}  |  Prefix: {args.prefix}")
    print(f"Audio  : {SAMPLE_RATE} Hz  mono  16-bit PCM")
    print(f"Chunks : {TARGET_MIN}–{TARGET_MAX}s  (word-boundary cuts)")

    counter   = [0]
    all_files = []
    for m4a in m4a_files:
        all_files.extend(
            process_file(m4a, str(out), args.prefix, counter, model, args.language)
        )

    print(f"\n✔  {len(all_files)} WAV chunks → {out}/")


if __name__ == "__main__":
    main()