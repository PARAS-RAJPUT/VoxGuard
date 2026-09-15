"""
VoxGuard — Layer 1: Synthetic Voice Detector
==============================================
Wraps the pretrained AASIST anti-spoofing model to answer:
    "Is this audio AI-generated (TTS/voice-cloned) or genuine human speech?"

Model: AASIST (Jung et al., 2022) — SOTA on ASVspoof 2019 LA.
Checkpoint: bundled pretrained weights from the official repo
    https://github.com/clovaai/aasist  (models/weights/AASIST.pth)

detect() analyzes only the first ~4 seconds (AASIST's fixed input window).
detect_full() slides that window across the ENTIRE clip and aggregates
results — use this for anything longer than a few seconds, since audio
past the first window is otherwise silently ignored.

Requirements:
    pip install torch soundfile librosa numpy
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa

# ---- Make the AASIST repo importable -----------------------------------
AASIST_REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aasist")
sys.path.insert(0, AASIST_REPO_PATH)

import torch  # noqa: E402
from models.AASIST import Model as AASISTModel  # noqa: E402

# ---- Model config (matches config/AASIST.conf in the official repo) ----
MODEL_CONFIG = {
    "architecture": "AASIST",
    "nb_samp": 64600,          # fixed input length = ~4.04 sec at 16kHz
    "first_conv": 128,
    "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
    "gat_dims": [64, 32],
    "pool_ratios": [0.5, 0.7, 0.5, 0.5],
    "temperatures": [2.0, 2.0, 100.0, 100.0],
}

TARGET_SR = 16000
NB_SAMP = MODEL_CONFIG["nb_samp"]

DEFAULT_CHECKPOINT = os.path.join(AASIST_REPO_PATH, "models", "weights", "AASIST.pth")
LIGHTWEIGHT_CHECKPOINT = os.path.join(AASIST_REPO_PATH, "models", "weights", "AASIST-L.pth")


def _pad_or_repeat(x: np.ndarray, max_len: int = NB_SAMP) -> np.ndarray:
    """Match AASIST's own preprocessing exactly (see aasist/data_utils.py::pad).

    - If audio is longer than max_len: take the first max_len samples.
    - If shorter: tile (repeat) the waveform until it reaches max_len.
    """
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, num_repeats)[:max_len]


class SyntheticVoiceDetector:
    """Layer 1 of the VoxGuard pipeline: synthetic / voice-clone detection."""

    def __init__(self, checkpoint_path: str = DEFAULT_CHECKPOINT, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AASISTModel(MODEL_CONFIG).to(self.device)

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load an audio file and resample to 16kHz mono, matching training data."""
        waveform, sr = sf.read(audio_path, dtype="float32")

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if sr != TARGET_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

        return waveform

    def _score_window(self, waveform: np.ndarray) -> float:
        """Run one ~4s window through the model, return synthetic probability."""
        waveform = _pad_or_repeat(waveform, NB_SAMP)
        x = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]

        return float(probs[1].item())

    def detect(self, audio_path: str) -> dict:
        """
        Analyzes only the FIRST ~4 seconds of audio (AASIST's fixed window).
        Fast, but ignores anything after the first window — prefer
        detect_full() for clips longer than a few seconds.
        """
        waveform = self._load_audio(audio_path)
        synthetic_probability = self._score_window(waveform)
        label = "synthetic" if synthetic_probability > 0.5 else "bonafide"

        return {
            "synthetic_probability": round(synthetic_probability, 4),
            "label": label,
        }

    def detect_full(self, audio_path: str, window_sec: float = 4.0, hop_sec: float = 2.0) -> dict:
        """
        Analyzes the ENTIRE audio file by sliding a window across it,
        instead of only looking at the first ~4 seconds. Aggregates
        per-window scores into one overall result.

        window_sec: length of each analysis window (matches AASIST's ~4s input)
        hop_sec: how far to slide between windows (2s = 50% overlap)
        """
        full_waveform = self._load_audio(audio_path)
        window_len = int(window_sec * TARGET_SR)
        hop_len = int(hop_sec * TARGET_SR)

        if len(full_waveform) <= window_len:
            return self.detect(audio_path)

        scores = []
        start = 0
        while start < len(full_waveform):
            chunk = full_waveform[start:start + window_len]
            scores.append(self._score_window(chunk))
            start += hop_len

        # Use MAX across windows — if any part of the call sounds synthetic,
        # flag the whole call rather than averaging the signal away.
        synthetic_probability = max(scores)
        label = "synthetic" if synthetic_probability > 0.5 else "bonafide"

        return {
            "synthetic_probability": round(synthetic_probability, 4),
            "label": label,
            "num_windows_analyzed": len(scores),
            "per_window_scores": [round(s, 4) for s in scores],
        }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python voxguard_layer1_synthetic_voice_detector.py <audio_file.wav>")
        sys.exit(1)

    detector = SyntheticVoiceDetector()
    result = detector.detect_full(sys.argv[1])
    print(result)