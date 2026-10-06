from modules.productivity_score import ProductivityScore

tracker = ProductivityScore()

# Simulate a session
test_data = [
    ('High',   'Engaged',    90, True),
    ('High',   'Engaged',    85, True),
    ('Medium', 'Confused',   60, True),
    ('Low',    'Bored',      30, False),
    ('Low',    'Frustrated', 20, False),
    ('High',   'Engaged',    80, True),
    ('High',   'Engaged',    88, True),
    ('Medium', 'Confused',   55, True),
]

print("Frame by frame scores:")
for focus, emotion, attn, posture in test_data:
    score = tracker.calculate(focus, emotion, attn, posture)
    print(f"  {focus:8} | {emotion:12} | Attn:{attn} "
          f"| Posture:{'OK' if posture else 'Bad'} "
          f"→ Score: {score}")

print("\n=== Productivity Summary ===")
summary = tracker.get_summary()
for k, v in summary.items():
    if k != 'score_timeline':
        print(f"  {k}: {v}")