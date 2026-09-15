"""
VoxGuard — Layer 1: Synthetic Voice Detector
==============================================
Wraps the pretrained AASIST anti-spoofing model to answer:
    "Is this audio AI-generated (TTS/voice-cloned) or genuine human speech?"

Model: AASIST (Jung et al., 2022) — SOTA on ASVspoof 2019 LA.
Checkpoint: bundled pretrained weights from the official repo
    https://github.com/clovaai/aasist  (models/weights/AASIST.pth)

Contract with the rest of the VoxGuard pipeline (Person D's fusion engine
should call this exactly like this):

    from voxguard_layer1_synthetic_voice_detector import SyntheticVoiceDetector

    detector = SyntheticVoiceDetector()
    result = detector.detect("path/to/call_audio.wav")
    # result = {
    #     "synthetic_probability": 0.86,   # 0-1 float, >0.5 = likely AI-generated
    #     "label": "synthetic",            # "synthetic" or "bonafide"
    #     "raw_logits": [bonafide, spoof]  # for debugging / calibration
    # }

Requirements (install on your dev machine / Colab, NOT needed to read this file):
    pip install torch soundfile librosa numpy
"""

import os
import sys
import numpy as np
import soundfile as sf
import librosa

# ---- Make the AASIST repo importable -----------------------------------
# Assumes this file sits next to the cloned `aasist/` repo:
#   git clone https://github.com/clovaai/aasist.git
AASIST_REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aasist")
sys.path.insert(0, AASIST_REPO_PATH)

import torch  # noqa: E402  (import after path setup, kept here for clarity)
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
# For lower latency on real-time calls, swap to the lightweight variant:
LIGHTWEIGHT_CHECKPOINT = os.path.join(AASIST_REPO_PATH, "models", "weights", "AASIST-L.pth")


def _pad_or_repeat(x: np.ndarray, max_len: int = NB_SAMP) -> np.ndarray:
    """Match AASIST's own preprocessing exactly (see aasist/data_utils.py::pad).

    - If audio is longer than max_len: take the first max_len samples.
    - If shorter: tile (repeat) the waveform until it reaches max_len.
    This is NOT silence-padding — AASIST was trained with repetition-padding,
    so silence-padding here would hurt accuracy.
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

        # Collapse stereo -> mono if needed
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        if sr != TARGET_SR:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)

        return waveform

    def detect(self, audio_path: str) -> dict:
        """
        Run synthetic-voice detection on a single audio file (or a live-call
        chunk saved to a temp .wav — see chunking note at the bottom of this file).

        Returns a dict matching the contract described in the module docstring.
        """
        waveform = self._load_audio(audio_path)
        waveform = _pad_or_repeat(waveform, NB_SAMP)

        x = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)  # (1, 64600)

        with torch.no_grad():
            _, logits = self.model(x)              # logits shape: (1, 2) -> [bonafide, spoof]
            probs = torch.softmax(logits, dim=1)[0]  # convert to a 0-1 probability

        synthetic_probability = float(probs[1].item())
        label = "synthetic" if synthetic_probability > 0.5 else "bonafide"

        return {
            "synthetic_probability": round(synthetic_probability, 4),
            "label": label,
            "raw_logits": logits[0].tolist(),
        }


if __name__ == "__main__":
    # Quick manual test:
    #   python voxguard_layer1_synthetic_voice_detector.py path/to/audio.wav
    if len(sys.argv) < 2:
        print("Usage: python voxguard_layer1_synthetic_voice_detector.py <audio_file.wav>")
        sys.exit(1)

    detector = SyntheticVoiceDetector()
    result = detector.detect(sys.argv[1])
    print(result)

# -----------------------------------------------------------------------
# NOTE for real-time streaming (relevant to your "how it works" slide):
# AASIST expects a fixed ~4-second window, not an open audio stream. For
# live calls, buffer the incoming stream into overlapping ~4s windows
# (e.g. every 2s, run detect() on the last 4s), and feed each window's
# synthetic_probability into the Risk Engine as a rolling signal rather
# than a single one-shot score. This also naturally handles calls longer
# than 4 seconds without retraining the model.
# -----------------------------------------------------------------------