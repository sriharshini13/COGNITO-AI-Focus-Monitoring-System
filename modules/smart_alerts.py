import time
from collections import deque

class SmartAlertSystem:
    """
    Context-aware alert system that avoids
    alert fatigue by being smart about when
    and how to alert the user.
    """

    def __init__(self):
        self.last_alert_time  = {}
        self.alert_counts     = {}
        self.focus_window     = deque(maxlen=30)
        self.emotion_window   = deque(maxlen=30)
        self.last_beep_time   = 0
        self.consecutive_low  = 0
        self.session_start    = time.time()

        # Cooldowns per alert type (seconds)
        self.cooldowns = {
            'low_focus':   45,
            'drowsy':      30,
            'emergency':   10,
            'yawn':        90,
            'break':       300,
            'emotion_tip': 120,
            'beep':        15,
        }

        # Emotion tips
        self.emotion_tips = {
            'Bored': [
                '😴 Feeling bored? Try a harder challenge!',
                '🎯 Switch to a more engaging task.',
                '⚡ Take a 2-min walk to reset energy.',
            ],
            'Confused': [
                '😕 Confused? Break the topic into steps.',
                '📖 Review the basics before continuing.',
                '💬 Try explaining it out loud to yourself.',
            ],
            'Frustrated': [
                '😤 Frustrated? Step away for 2 minutes.',
                '🧘 Take 3 deep breaths and reset.',
                '✅ Focus on what you already know.',
            ]
        }
        self.emotion_tip_idx = {
            'Bored': 0,
            'Confused': 0,
            'Frustrated': 0
        }

    def update(self, focus_label, emotion_label,
               ear, attention_score, profile):
        """
        Main update — call every frame.
        Returns list of alerts to show.
        """
        now     = time.time()
        alerts  = []
        elapsed = now - self.session_start

        # Update windows
        self.focus_window.append(focus_label)
        self.emotion_window.append(emotion_label)

        # ── Only alert after 10 seconds ──────────────
        if elapsed < 10:
            return alerts, False

        # ── Emergency drowsiness ─────────────────────
        if ear < profile.get('emergency_ear', 0.15):
            if self._can_alert('emergency', now):
                alerts.append({
                    'type':    'emergency',
                    'message': profile['alerts']['emergency'],
                    'icon':    '🚨',
                    'priority': 5
                })

        # ── Drowsiness levels ────────────────────────
        elif ear < profile.get('drowsy_ear_threshold', 0.20):
            if self._can_alert('drowsy', now):
                alerts.append({
                    'type':    'warning',
                    'message': profile['alerts']['drowsy'],
                    'icon':    '😴',
                    'priority': 4
                })

        # ── Sustained low focus (10+ frames) ─────────
        low_count = sum(
            1 for f in self.focus_window
            if f == 'Low')
        if low_count >= 15:
            if self._can_alert('low_focus', now):
                alerts.append({
                    'type':    'warning',
                    'message': profile['alerts']['low_focus'],
                    'icon':    '⚠️',
                    'priority': 3
                })

        # ── Emotion-based tips ────────────────────────
        neg_emotions = ['Bored','Confused','Frustrated']
        if emotion_label in neg_emotions:
            neg_count = sum(
                1 for e in self.emotion_window
                if e == emotion_label)
            if neg_count >= 20:
                if self._can_alert('emotion_tip', now):
                    tips = self.emotion_tips.get(
                        emotion_label, [])
                    if tips:
                        idx = self.emotion_tip_idx\
                            .get(emotion_label, 0)
                        tip = tips[idx % len(tips)]
                        self.emotion_tip_idx[
                            emotion_label] = idx + 1
                        alerts.append({
                            'type':    'tip',
                            'message': tip,
                            'icon':    '💡',
                            'priority': 2
                        })

        # ── Break reminder ────────────────────────────
        break_secs = profile.get(
            'break_interval_secs', 2700)
        if elapsed > break_secs:
            if self._can_alert('break', now):
                alerts.append({
                    'type':    'break',
                    'message': profile['alerts']['break'],
                    'icon':    '☕',
                    'priority': 3
                })

        # ── Beep logic ────────────────────────────────
        should_beep = False
        if focus_label == 'Low' or \
                ear < profile.get(
                    'drowsy_ear_threshold', 0.20):
            if (now - self.last_beep_time) > \
                    self.cooldowns['beep']:
                self.last_beep_time = now
                should_beep = True

        # Sort by priority
        alerts.sort(
            key=lambda x: x['priority'],
            reverse=True)

        return alerts[:1], should_beep

    def _can_alert(self, alert_type, now):
        last = self.last_alert_time.get(
            alert_type, 0)
        cooldown = self.cooldowns.get(
            alert_type, 60)
        if (now - last) >= cooldown:
            self.last_alert_time[alert_type] = now
            return True
        return False

    def reset(self):
        self.last_alert_time  = {}
        self.alert_counts     = {}
        self.focus_window.clear()
        self.emotion_window.clear()
        self.last_beep_time  = 0
        self.session_start   = time.time()