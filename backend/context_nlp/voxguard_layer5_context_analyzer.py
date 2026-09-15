"""
VoxGuard — Layer 5: Context Analyzer (Groq-powered)
=======================================================
Transcribes audio via Groq's Whisper API and classifies intent using
Groq's LLM — replaces brittle keyword matching with real NLP understanding
and a genuine confidence score.
"""

import os
from dotenv import load_dotenv
from groq import Groq


load_dotenv()  # reads .env in the project root and loads it into os.environ

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def analyze_context(audio_path: str) -> dict:
    with open(audio_path, "rb") as f:
        transcript_result = client.audio.transcriptions.create(
            file=f,
            model="whisper-large-v3",
        )
    transcript = transcript_result.text

    prompt = f"""You are a fraud-call detection system. Classify this phone call transcript.

Respond in EXACTLY this format (no extra text):
category: <benign | fraud_financial | threat_violence>
confidence: <0-100>
reason: <one short sentence>

Transcript: "{transcript}\""""

    completion = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    reply = completion.choices[0].message.content.strip()

    # Parse the structured reply
    category, confidence, reason = "benign", 5, "No indicators found"
    for line in reply.splitlines():
        if line.lower().startswith("category:"):
            category = line.split(":", 1)[1].strip().lower()
        elif line.lower().startswith("confidence:"):
            try:
                confidence = int("".join(c for c in line.split(":", 1)[1] if c.isdigit()))
            except ValueError:
                confidence = 50
        elif line.lower().startswith("reason:"):
            reason = line.split(":", 1)[1].strip()

    return {
        "transcript": transcript,
        "intent_category": category,
        "intent_confidence": confidence,
        "intent_reason": reason,
    }


if __name__ == "__main__":
    print(analyze_context("test_audio.wav"))