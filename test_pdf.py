from modules.pdf_export import generate_pdf
from collections import defaultdict

focus_stats   = {'High': 300, 'Medium': 80, 'Low': 40}
emotion_stats = {'Engaged': 300, 'Bored': 60,
                 'Confused': 40, 'Frustrated': 20}
productivity  = {'average_score': 78, 'peak_score': 97,
                 'grade': 'B', 'grade_label': 'Good Job! 👍'}
distraction   = {'total_distractions': 5,
                 'distractions_per_hour': 12.0}

path = generate_pdf(focus_stats, emotion_stats,
                    productivity, distraction, 1500)
print(f"PDF created at: {path}")