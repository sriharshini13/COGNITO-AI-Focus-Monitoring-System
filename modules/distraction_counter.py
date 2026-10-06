import time
from collections import deque

class DistractionCounter:
    def __init__(self, cooldown_secs=5):
        """
        Tracks how many times focus drops from High/Medium to Low.
        cooldown_secs: minimum gap between counting two distractions
        """
        self.count           = 0
        self.cooldown        = cooldown_secs
        self.last_distracted = 0
        self.prev_focus      = None
        self.distraction_log = []  # list of timestamps

    def update(self, focus_label):
        """
        Call this every frame with the current focus label.
        Returns True if a new distraction was just counted.
        """
        now = time.time()
        new_distraction = False

        # Detect transition INTO Low focus
        if (self.prev_focus in ['High', 'Medium'] and
                focus_label == 'Low' and
                (now - self.last_distracted) > self.cooldown):

            self.count           += 1
            self.last_distracted  = now
            self.distraction_log.append(now)
            new_distraction       = True

        self.prev_focus = focus_label
        return new_distraction

    def get_count(self):
        return self.count

    def get_rate(self, session_secs):
        """Distractions per hour"""
        if session_secs < 1:
            return 0
        return round(self.count / (session_secs / 3600), 1)

    def get_log(self):
        return self.distraction_log

    def get_summary(self, session_secs):
        return {
            'total_distractions': self.count,
            'distractions_per_hour': self.get_rate(session_secs),
            'distraction_timestamps': self.distraction_log
        }