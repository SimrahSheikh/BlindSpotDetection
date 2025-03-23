def get_warning_level(risk_level):
    """Convert numerical prediction into a risk message."""
    messages = {
        0: "✅ SAFE! No immediate danger.",
        1: "⚠️ CAUTION! Potential risk detected.",
        2: "🚨 DANGER! Immediate action required!"
    }
    return messages.get(risk_level, "❓ Unknown risk level.")
