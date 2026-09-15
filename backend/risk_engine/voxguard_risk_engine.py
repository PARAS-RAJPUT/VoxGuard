"""
VoxGuard — Risk Fusion Engine
================================
Combines outputs from all layers into a single risk score + explanation.

Fusion logic: the strongest single signal sets the base risk score (so one
highly-confident red flag — e.g. a threat detected with 95% confidence —
can drive risk to HIGH on its own), and any additional corroborating
signals push the score further up. This avoids the old flaw where a
weighted-average formula mathematically capped any single signal's
contribution to its fixed weight, making it impossible for even a 100%-
confident threat to cross into HIGH-RISK alone.
"""

def compute_risk(synthetic_result: dict, watchlist_result: dict, context_result: dict) -> dict:
    """
    synthetic_result: output of SyntheticVoiceDetector.detect()
    watchlist_result: output of check_watchlist()
    context_result: output of analyze_context()
    """

    synthetic_prob = synthetic_result["synthetic_probability"]
    watchlist_match = watchlist_result["watchlist_match"]
    watchlist_conf = max(watchlist_result["confidence"], 0)  # clip negative cosine sim to 0
    intent = context_result["intent_category"]
    intent_confidence = context_result.get("intent_confidence", 5)

    # --- individual signal scores (0-100 scale) ---
    synthetic_score = synthetic_prob * 100
    watchlist_score = 100 if watchlist_match else 0

    if intent in ("fraud_financial", "threat_violence"):
        intent_score = intent_confidence
    else:
        intent_score = 5

    # --- fusion: strongest signal sets the base, others add corroboration ---
    signals = [synthetic_score, watchlist_score, intent_score]
    base_score = max(signals)

    other_signals = sorted(signals, reverse=True)[1:]
    corroboration_bonus = sum(s * 0.25 for s in other_signals)

    risk_score = min(base_score + corroboration_bonus, 100)
    risk_score = round(risk_score, 1)

    if risk_score >= 70:
        risk_level = "HIGH-RISK CALL"
    elif risk_score >= 25:
        risk_level = "MEDIUM-RISK CALL"
    else:
        risk_level = "LOW-RISK CALL"

    reasons = []
    if synthetic_prob > 0.5:
        reasons.append("AI-generated speech artifacts detected")
    if watchlist_match:
        reasons.append(f"Speaker matches watchlist entry: {watchlist_result['matched_name']}")
    if intent == "fraud_financial":
        reasons.append(f"{context_result.get('intent_reason', 'Suspicious financial request detected')} (confidence: {intent_confidence}%)")
    if intent == "threat_violence":
        reasons.append(f"{context_result.get('intent_reason', 'Threat-related language detected')} (confidence: {intent_confidence}%)")
    if not reasons:
        reasons.append("No significant risk indicators detected")

    action = "Do not authorize transaction. Perform secondary identity verification." \
        if risk_score >= 70 else \
        "Proceed with standard verification." if risk_score >= 25 else \
        "No action needed."

    return {
        "synthetic_probability": round(synthetic_prob, 4),
        "watchlist_match": watchlist_match,
        "matched_name": watchlist_result.get("matched_name"),
        "intent_category": intent,
        "intent_confidence": intent_confidence,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "reasons": reasons,
        "recommended_action": action,
    }


if __name__ == "__main__":
    # Quick isolated test — no audio files needed
    result = compute_risk(
        {"synthetic_probability": 0.01},
        {"watchlist_match": False, "confidence": 0},
        {"intent_category": "threat_violence", "intent_confidence": 94, "intent_reason": "Explicit threat of violence detected"}
    )
    print(result)