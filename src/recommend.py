"""What-if suggestions: re-predict with one controllable habit improved."""

SCENARIOS = [
    {"feature": "studytime", "text": "Study one level more per week",
     "new_value": lambda v: min(v + 1, 4)},
    {"feature": "absences", "text": "Halve your absences",
     "new_value": lambda v: round(v / 2)},
    {"feature": "goout", "text": "Go out with friends one level less",
     "new_value": lambda v: max(v - 1, 1)},
    {"feature": "Walc", "text": "Lower weekend alcohol use by one level",
     "new_value": lambda v: max(v - 1, 1)},
]


def suggest_improvements(model, student, min_gain=0.1):
    base = model.predict(student)[0]
    suggestions = []
    for s in SCENARIOS:
        current = student.iloc[0][s["feature"]]
        new = s["new_value"](current)
        if new == current:
            continue
        changed = student.copy()
        changed[s["feature"]] = new
        gain = model.predict(changed)[0] - base
        if gain >= min_gain:
            suggestions.append({
                "feature": s["feature"],
                "action": s["text"],
                "from": int(current),
                "to": int(new),
                "gain": round(float(gain), 2),
            })
    return sorted(suggestions, key=lambda d: d["gain"], reverse=True)
