# 🛡️ VoxGuard

**Real-Time Voice Cloning & Impersonation Detection**
Smart India Hackathon 2026 — Problem Statement 26104

VoxGuard is a multi-layered voice security system that analyzes a call in
real time to answer three questions at once: *Is this voice AI-generated?
Does it match a known/enrolled speaker? Is the caller asking for something
they shouldn't be?* Each signal is combined into a single, explainable risk
score with a recommended action.

## Why this matters

AI voice cloning has turned identity into an attack surface. A short audio
sample is now enough to convincingly impersonate someone over a phone call
— enabling fraud, OTP theft, and social engineering that traditional
caller-ID and voice checks cannot catch. VoxGuard is built to close that
gap by never relying on a single signal.

## Architecture

| Layer | Question it answers | Model |
|---|---|---|
| Synthetic Voice Detector | Is the audio AI-generated? | AASIST (pretrained on ASVspoof 2019 LA) |
| Speaker Verification / Watchlist | Does this voice match a known or flagged speaker? | ECAPA-TDNN (SpeechBrain, pretrained on VoxCeleb) |
| Context Analyzer | Is the caller requesting money, OTP, or making a threat? | Groq Whisper (transcription) + Groq LLM (intent classification with confidence scoring) |
| Risk Engine | How dangerous is the overall situation? | Custom fusion logic |

**Fusion logic:** the strongest single signal sets the base risk score — a
highly confident threat or fraud detection can trigger HIGH risk on its
own — and any additional corroborating signals (e.g. a synthetic voice
*and* a watchlist match) push the score further up.

## Demo

Run the Streamlit dashboard locally:

```bash
git clone https://github.com/<your-username>/VoxGuard.git
cd VoxGuard
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq API key:

GROQ_API_KEY=your_key_here


Then run:
```bash
python -m streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### What you can do in the dashboard
1. **Enroll a known speaker** — upload a reference voice sample to add them to the watchlist
2. **Analyze an incoming call** — upload call audio and get a full risk breakdown: synthetic voice probability, watchlist match, intent classification, and a fused risk score with recommended action
3. **Live microphone monitoring** — records short audio chunks and analyzes them continuously to simulate real-time call monitoring

## Project structure

VoxGuard/
├── audio_layers/ # Layer 1: synthetic voice / deepfake detection (AASIST)
├── speaker_id/ # Layer 2: speaker verification & watchlist matching
├── context_nlp/ # Layer 5: transcription + intent classification (Groq)
├── risk_engine/ # Fusion logic — combines all signals into a risk score
├── data/ # Test audio samples
├── app.py # Streamlit dashboard
└── requirements.txt


## Datasets & models used

- [ASVspoof 2019 LA](https://www.asvspoof.org/index2019.html) — synthetic/spoofed speech benchmark, used to validate the synthetic voice detector
- [In-the-Wild](https://huggingface.co/datasets/mueller91/In-The-Wild) — real-world deepfake audio, used for generalization testing
- [AASIST](https://github.com/clovaai/aasist) (Jung et al., 2022) — pretrained anti-spoofing checkpoint
- [SpeechBrain ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) — pretrained speaker embedding model
- [Groq API](https://console.groq.com) — Whisper transcription + LLM-based intent classification

## Known limitations (honest, by design)

- The synthetic voice detector, like all anti-spoofing models, generalizes
  better to attack types present in its training data (ASVspoof) than to
  newer/unseen TTS systems — a known open problem in anti-spoofing research,
  not unique to this implementation.
- The watchlist in this demo uses synthetic/self-enrolled data, not a real
  law-enforcement or telecom database. In production this would integrate
  with an authorized watchlist API.
- Real-time processing here is simulated via short microphone-chunk
  analysis, not live telephony call interception — extending to real call
  audio would require telephony integration (e.g. Twilio, SIP trunking).

## Team

Built for SIH 2026 by [byteX] — [Prince, Sarvagya,Paras, Ramya,Nitin, Gaurav].

## Tech stack

Python · PyTorch · SpeechBrain · Whisper (via Groq) · Groq LLM API ·
Streamlit · Librosa · Soundfile