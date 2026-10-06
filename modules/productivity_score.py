class ProductivityScore:
    def __init__(self):
        self.scores = []
        self.weights = {
            'focus':     0.40,  # 40% weight
            'emotion':   0.25,  # 25% weight
            'attention': 0.25,  # 25% weight
            'posture':   0.10   # 10% weight
        }

    def calculate(self, focus_label, emotion_label,
                  attention_score, posture_good):
        """
        Calculate a productivity score 0-100 each frame.
        """
        # Focus score
        focus_score = {'High': 100, 'Medium': 60, 'Low': 20}.get(
            focus_label, 50)

        # Emotion score
        emotion_score = {
            'Engaged':    100,
            'Confused':    60,
            'Bored':       30,
            'Frustrated':  20
        }.get(emotion_label, 50)

        # Attention score (already 0-100)
        attn_score = attention_score

        # Posture score
        posture_score = 100 if posture_good else 40

        # Weighted final score
        final = (
            focus_score   * self.weights['focus']   +
            emotion_score * self.weights['emotion']  +
            attn_score    * self.weights['attention'] +
            posture_score * self.weights['posture']
        )

        final = max(0, min(100, int(final)))
        self.scores.append(final)
        return final

    def get_current(self):
        return self.scores[-1] if self.scores else 0

    def get_average(self):
        return int(sum(self.scores) / len(self.scores)) if self.scores else 0

    def get_peak(self):
        return max(self.scores) if self.scores else 0

    def get_grade(self):
        avg = self.get_average()
        if avg >= 85: return 'A', 'Excellent! 🌟'
        if avg >= 70: return 'B', 'Good Job! 👍'
        if avg >= 55: return 'C', 'Average 😐'
        if avg >= 40: return 'D', 'Needs Improvement 😕'
        return 'F', 'Poor Focus 😴'

    def get_summary(self):
        grade, label = self.get_grade()
        return {
            'average_score':     self.get_average(),
            'peak_score':        self.get_peak(),
            'total_frames':      len(self.scores),
            'grade':             grade,
            'grade_label':       label,
            'score_timeline':    self.scores
        }