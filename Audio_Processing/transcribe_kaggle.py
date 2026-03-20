# %%writefile /tmp/script.py
#Alternate version - bengaliSpeech2text - not better than meta mms
"""
generate_metadata.py  (Bengali edition)
----------------------------------------
Generates metadata.csv in LJSpeech format for Bengali WAV files.

LJSpeech format (pipe-delimited, no header row):
    filename|normalized_transcription

Usage:
    # Recommended – fine-tuned bangla-whisper:
    python generate_metadata.py --wav_dir ./wavs --output metadata.csv

"""

import os
import re
import csv
import time
import argparse
from pathlib import Path
from banglaspeech2text import Speech2Text


stt = Speech2Text("large")


# ── Text normalisation ────────────────────────────────────────────────────────

def normalize_bengali(text: str) -> str:
    """
    Light normalisation for Bengali TTS metadata:
      - Strip surrounding whitespace
      - Collapse multiple spaces / zero-width chars
      - Remove ASCII control characters
      - Keep Bengali punctuation (।  ॥  ,  ?  !)
    """
    text = text.strip()
    # Remove zero-width non-joiner/joiner artefacts sometimes added by ASR
    text = text.replace("\u200c", "").replace("\u200d", "")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Drop non-printable ASCII control chars while keeping Bengali Unicode
    text = "".join(ch for ch in text if ch >= " " or ch == "\t")
    return text.strip()


# ── Backend: bangla-whisper (fine-tuned HuggingFace model) ───────────────────


# ── Backend: Meta MMS ─────────────────────────────────────────────────────────



# ── Backend: Stock Whisper large-v3 ──────────────────────────────────────────

# ── Main ──────────────────────────────────────────────────────────────────────



def generate_metadata(wav_dir: str, output_csv: str) -> None:
    wav_dir   = Path(wav_dir)
    wav_files = sorted(wav_dir.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    print(f"Found    : {len(wav_files)} WAV files")
    print(f"Output   : {output_csv}\n")

    print(f"\n{'─'*65}")

    rows   = []
    errors = []

    for i, wav_path in enumerate(wav_files, 1):
        stem = wav_path.stem
        t0   = time.time()
        try:
            raw   = stt.recognize(str(wav_path))
            text  = normalize_bengali(raw)
            elapsed = time.time() - t0
            print(f"[{i:>4}/{len(wav_files)}]  {stem}  ({elapsed:.1f}s)")
            print(f"          ↳ {text[:80]}{'…' if len(text)>80 else ''}")
            rows.append((stem, text))
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"[{i:>4}/{len(wav_files)}]  {stem}  ✗ ERROR: {exc}")
            errors.append(stem)
            rows.append((stem, ""))

    # Write CSV with UTF-8 BOM so Excel opens Bengali text correctly
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow(row)

    print(f"\n{'─'*65}")
    print(f"✔  metadata.csv written → {output_csv}")
    print(f"   Total   : {len(wav_files)}")
    print(f"   Success : {len(wav_files) - len(errors)}")
    if errors:
        print(f"   Errors  : {', '.join(errors)}")


def main():
    ap = argparse.ArgumentParser(
        description="Generate LJSpeech metadata.csv with Bengali transcription."
    )
    ap.add_argument("--wav_dir", default="/kaggle/input/datasets/tausifr/transcribe-test/test_wavs/",
                    help="Folder with WAV chunks (output of split_audio.py)")
    ap.add_argument("--output",   default="/kaggle/working/metadata_test.csv")
    args = ap.parse_args()
    generate_metadata(args.wav_dir, args.output)


if __name__ == "__main__":
    main()