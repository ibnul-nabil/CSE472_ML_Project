#!/usr/bin/env python3 
# Meta mms works well
"""
generate_metadata.py  (Bengali edition)
----------------------------------------
Generates metadata.csv in LJSpeech format for Bengali WAV files.

LJSpeech format (pipe-delimited, no header row):
    filename|transcription|normalized_transcription

Supported backends (in order of recommended quality for Bengali):

  1. bangla-whisper  [DEFAULT]
     Fine-tuned Whisper model trained specifically on Bengali speech.
     Outputs correct বাংলা script. Best accuracy for Bangladeshi Bengali.
     Model: "bangla-whisper" → HuggingFace: hasan604/bangla-whisper-small
     pip install transformers torch torchaudio

  2. mms
     Meta's Massively Multilingual Speech model, Bengali variant.
     Very good character-level accuracy, proper বাংলা script output.
     Model: facebook/mms-1b-all  (uses Bengali adapter)
     pip install transformers torch torchaudio

  3. whisper-large
     Stock OpenAI Whisper large-v3 forced to Bengali.
     Usable but ~100% WER on raw model; may output Devanagari occasionally.
     Use only if you cannot download the larger fine-tuned models.
     pip install openai-whisper

Usage:
    # Recommended – fine-tuned bangla-whisper:
    python generate_metadata.py --wav_dir ./wavs --output metadata.csv

    # Meta MMS:
    python generate_metadata.py --wav_dir ./wavs --backend mms

    # Stock Whisper large-v3 (fallback):
    python generate_metadata.py --wav_dir ./wavs --backend whisper-large
"""

import os
import re
import csv
import time
import argparse
from pathlib import Path


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

def load_bangla_whisper():
    try:
        from transformers import pipeline
        import torch
    except ImportError:
        raise ImportError("Run:  pip install transformers torch torchaudio")

    #model_id = "hasan604/bangla-whisper-small"
    model_id = "asif00/whisper-bangla"

    print(f"  Loading fine-tuned Bangla Whisper: {model_id}")
    print("  (First run downloads ~500 MB)")

    device = 0 if __import__("torch").cuda.is_available() else -1
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        # chunk_length_s=30,
        # stride_length_s=5,
    )
    # Use a pipeline as a high-level helper
    #pipe = pipeline("automatic-speech-recognition", model="asif00/whisper-bangla")
    return pipe


def transcribe_bangla_whisper(pipe, wav_path: str) -> str:
    result = pipe(wav_path, generate_kwargs={"language": "bengali"})
    return result["text"]


# ── Backend: Meta MMS ─────────────────────────────────────────────────────────

def load_mms():
    try:
        from transformers import Wav2Vec2ForCTC, AutoProcessor
        import torch
    except ImportError:
        raise ImportError("Run:  pip install transformers torch torchaudio")

    model_id = "facebook/mms-1b-all"
    print(f"  Loading Meta MMS model: {model_id}")
    print("  (First run downloads ~4 GB)")

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.set_target_lang("ben")          # Bengali

    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.load_adapter("ben")
    model.eval()

    return {"model": model, "processor": processor}


def transcribe_mms(bundle, wav_path: str) -> str:
    import torch
    import torchaudio

    model     = bundle["model"]
    processor = bundle["processor"]

    waveform, sr = torchaudio.load(wav_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    waveform = waveform.squeeze(0)

    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    ids  = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(ids)[0]
    return text


# ── Backend: Stock Whisper large-v3 ──────────────────────────────────────────

def load_whisper_large():
    try:
        import whisper
    except ImportError:
        raise ImportError("Run:  pip install openai-whisper")

    print("  Loading openai-whisper large-v3 …")
    print("  (First run downloads ~3 GB)")
    return whisper.load_model("large-v3")


def transcribe_whisper_large(model, wav_path: str) -> str:
    result = model.transcribe(wav_path, language="bn", fp16=False)
    return result["text"]


# ── Main ──────────────────────────────────────────────────────────────────────

BACKENDS = {
    "bangla-whisper": (load_bangla_whisper,  transcribe_bangla_whisper),
    "mms":            (load_mms,             transcribe_mms),
    "whisper-large":  (load_whisper_large,   transcribe_whisper_large),
}


def generate_metadata(wav_dir: str, output_csv: str, backend: str) -> None:
    wav_dir   = Path(wav_dir)
    wav_files = sorted(wav_dir.glob("*.wav"))

    if not wav_files:
        print(f"No WAV files found in {wav_dir}")
        return

    print(f"Found    : {len(wav_files)} WAV files")
    print(f"Backend  : {backend}")
    print(f"Output   : {output_csv}\n")

    loader, transcriber = BACKENDS[backend]
    model = loader()

    print(f"\n{'─'*65}")

    rows   = []
    errors = []

    for i, wav_path in enumerate(wav_files, 1):
        stem = wav_path.stem
        t0   = time.time()
        try:
            raw   = transcriber(model, str(wav_path))
            text  = normalize_bengali(raw)
            elapsed = time.time() - t0
            print(f"[{i:>4}/{len(wav_files)}]  {stem}  ({elapsed:.1f}s)")
            print(f"          ↳ {text[:80]}{'…' if len(text)>80 else ''}")
            rows.append((stem, text, text))
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"[{i:>4}/{len(wav_files)}]  {stem}  ✗ ERROR: {exc}")
            errors.append(stem)
            rows.append((stem, "", ""))

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
    ap.add_argument("--wav_dir", default="./wavs",
                    help="Folder with WAV chunks (output of split_audio.py)")
    ap.add_argument("--output",   default="metadata.csv")
    ap.add_argument("--backend",  default="bangla-whisper",
                    choices=list(BACKENDS.keys()),
                    help=(
                        "bangla-whisper = fine-tuned, best quality (default)\n"
                        "mms            = Meta MMS, great character accuracy\n"
                        "whisper-large  = stock large-v3, fallback only"
                    ))
    args = ap.parse_args()
    generate_metadata(args.wav_dir, args.output, args.backend)


if __name__ == "__main__":
    main()