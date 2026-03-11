import torch
import torch.nn.functional as F
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier
from pathlib import Path

# ── 1. Load Model ──────────────────────────────────────────────────────────────
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)

TARGET_SR = 16000   # ECAPA-TDNN expects 16kHz
MAX_SECONDS = 4     # Paper: "we analyzed 4 seconds of the audio signal"
MAX_SAMPLES = TARGET_SR * MAX_SECONDS


# ── 2. Audio Preprocessing ─────────────────────────────────────────────────────
def load_audio(path: str) -> torch.Tensor:
    """Load, resample to 16kHz, convert to mono, trim to 4 seconds."""
    signal, sr = torchaudio.load(path)

    # Resample if needed
    if sr != TARGET_SR:
        signal = torchaudio.functional.resample(signal, sr, TARGET_SR)

    # Mono
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)

    # Trim/pad to 4 seconds (paper spec)
    if signal.shape[1] > MAX_SAMPLES:
        signal = signal[:, :MAX_SAMPLES]
    elif signal.shape[1] < MAX_SAMPLES:
        # Pad with zeros if shorter
        pad = MAX_SAMPLES - signal.shape[1]
        signal = F.pad(signal, (0, pad))

    return signal  # shape: (1, 64000)


# ── 3. Embedding Extraction ────────────────────────────────────────────────────
def get_embedding(audio_path: str) -> torch.Tensor:
    """Returns L2-normalized embedding vector of shape (192,)."""
    signal = load_audio(audio_path)
    with torch.no_grad():
        emb = classifier.encode_batch(signal)   # (1, 1, 192)
    emb = emb.squeeze()                          # (192,)
    return F.normalize(emb, p=2, dim=0)


# ── 4. Build Reference Centroid (from multiple real audio files) ───────────────
def build_centroid(reference_audio_paths: list) -> torch.Tensor:
    """
    Compute centroid from N reference (real) audio files.
    Paper: performance plateaus at ~20 references for CB strategy.
    Returns normalized centroid of shape (192,).
    """
    embeddings = []
    for path in reference_audio_paths:
        emb = get_embedding(path)
        embeddings.append(emb)

    embeddings = torch.stack(embeddings, dim=0)  # (N, 192)
    centroid = embeddings.mean(dim=0)            # (192,)
    return F.normalize(centroid, p=2, dim=0)     # normalize centroid


# ── 5. Centroid-Based Detection (Paper Section II) ────────────────────────────
def centroid_based_detection(
    test_audio_path: str,
    centroid: torch.Tensor,
    threshold: float = 0.5           # tune this on your validation set
) -> dict:
    """
    Implements CB testing strategy from the paper.
    Audio is REAL if similarity > threshold, FAKE if below.
    """
    test_emb = get_embedding(test_audio_path)
    score = torch.dot(centroid, test_emb).item()  # cosine sim (both normalized)

    return {
        "score": score,
        "prediction": "REAL" if score >= threshold else "FAKE",
        "threshold": threshold
    }


# ── 6. Multi-Similarity Detection (Paper Section II, better for noisy audio) ──
def multi_similarity_detection(
    test_audio_path: str,
    reference_embeddings: list,       # list of (192,) tensors
    threshold: float = 0.5
) -> dict:
    """
    Implements MS testing strategy from the paper.
    Takes MAX similarity across all reference embeddings.
    Better for in-the-wild / noisy audio (per paper Fig. 3).
    """
    test_emb = get_embedding(test_audio_path)
    scores = [torch.dot(ref, test_emb).item() for ref in reference_embeddings]
    max_score = max(scores)

    return {
        "score": max_score,
        "all_scores": scores,
        "prediction": "REAL" if max_score >= threshold else "FAKE",
        "threshold": threshold
    }


# ── 7. Full Pipeline ───────────────────────────────────────────────────────────
def run_deepfake_detection(
    reference_dir: str,
    test_dir: str,           # ← changed from single file to directory
    threshold: float = 0.5,
    strategy: str = "CB"
) -> list:
    ref_paths = list(Path(reference_dir).glob("*.wav"))
    test_paths = list(Path(test_dir).glob("*.wav"))
    
    assert len(ref_paths) >= 5, "Need at least 5 reference audios (20+ recommended)"
    assert len(test_paths) > 0, "No .wav files found in test directory"

    print(f"Reference files : {len(ref_paths)}")
    print(f"Test files      : {len(test_paths)}")
    print(f"Strategy        : {'Centroid-Based' if strategy == 'CB' else 'Multi-Similarity'}")

    # Build reference once (reused for all test files)
    if strategy == "CB":
        reference = build_centroid([str(p) for p in ref_paths])
    else:
        reference = [get_embedding(str(p)) for p in ref_paths]

    results = []
    for i, test_path in enumerate(test_paths, 1):
        if strategy == "CB":
            result = centroid_based_detection(str(test_path), reference, threshold)
        else:
            result = multi_similarity_detection(str(test_path), reference, threshold)

        result["file"] = test_path.name
        results.append(result)
        print(f"[{i:03d}/{len(test_paths)}] {test_path.name:<40} score={result['score']:.4f}  →  {result['prediction']}")

    # Summary
    fake_count = sum(1 for r in results if r["prediction"] == "FAKE")
    real_count = len(results) - fake_count
    print(f"\n{'─'*60}")
    print(f"Total  : {len(results)}")
    print(f"FAKE   : {fake_count}  ({100*fake_count/len(results):.1f}%)")
    print(f"REAL   : {real_count}  ({100*real_count/len(results):.1f}%)")

    # Save to CSV
    import csv
    output_csv = Path(test_dir) / "detection_results.csv"
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "score", "prediction", "threshold"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ["file", "score", "prediction", "threshold"]})
    print(f"Results saved → {output_csv}")

    return results


# Usage
results = run_deepfake_detection(
    reference_dir="banglafake_samples/sust_sample_s1/real",
    test_dir="banglafake_samples/sust_sample_s1/fake",         # ← just drop your fake folder here
    threshold=0.70,
    strategy="CB"
)

# reference_dir="banglafake_samples/sust_sample_s1/real",
# test_dir="banglafake_samples/sust_sample_s1/fake",    