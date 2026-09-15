"""
VoxGuard — Layer 2: Speaker Verification / Watchlist Matching
================================================================
Uses SpeechBrain's pretrained ECAPA-TDNN to answer:
    "Does this voice match a known/enrolled speaker?"

No training needed — pretrained on VoxCeleb, downloads automatically.

Install: pip install speechbrain torch torchaudio
"""

import torch
import soundfile as sf
import librosa
from speechbrain.inference.speaker import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_ecapa"
)

def get_embedding(audio_path: str):
    waveform, fs = sf.read(audio_path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if fs != 16000:
        waveform = librosa.resample(waveform, orig_sr=fs, target_sr=16000)

    signal = torch.from_numpy(waveform).float().unsqueeze(0)  # shape (1, samples)
    embedding = classifier.encode_batch(signal)
    return embedding.squeeze()

def cosine_similarity(emb1, emb2):
    return torch.nn.functional.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

# --- Demo watchlist (synthetic data — replace names/audio with team demo clips) ---
WATCHLIST = {
    # "person_name": embedding_tensor  (filled in via enroll())
}

def enroll(name: str, audio_path: str):
    WATCHLIST[name] = get_embedding(audio_path)

def check_watchlist(audio_path: str, threshold: float = 0.5):
    query_emb = get_embedding(audio_path)
    best_match, best_score = None, -1
    for name, enrolled_emb in WATCHLIST.items():
        score = cosine_similarity(query_emb, enrolled_emb)
        if score > best_score:
            best_match, best_score = name, score

    matched = best_score >= threshold if best_match else False
    return {
        "watchlist_match": matched,
        "matched_name": best_match if matched else None,
        "confidence": round(best_score, 4) if best_match else 0.0,
    }

def verify_speaker(audio_path: str, claimed_name: str, threshold: float = 0.5):
    """1:1 verification — does this audio match the CLAIMED identity?"""
    if claimed_name not in WATCHLIST:
        return {"verified": False, "reason": "no enrolled reference for this identity"}
    query_emb = get_embedding(audio_path)
    score = cosine_similarity(query_emb, WATCHLIST[claimed_name])
    return {"verified": score >= threshold, "confidence": round(score, 4)}


if __name__ == "__main__":
    # Quick test:
    enroll("teammate_1", "teammate1_reference.wav")
    print(check_watchlist("test_audio.wav"))