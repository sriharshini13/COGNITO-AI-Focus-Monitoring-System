import json
import os
from datetime import datetime

BASE_DIR  = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
LOGS_DIR  = os.path.join(
    BASE_DIR, 'session_logs', 'users')

os.makedirs(LOGS_DIR, exist_ok=True)


class SessionLogger:

    def __init__(self, username='default'):
        self.username  = username
        self.log_file  = os.path.join(
            LOGS_DIR,
            f"{self.username}_sessions.json")
        self._ensure()

    def _ensure(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                json.dump([], f)

    def _load(self):
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except:
            return []

    def _save(self, data):
        with open(self.log_file, 'w') as f:
            json.dump(data, f, indent=2)

    def save_session(self, focus_stats,
                     emotion_stats,
                     prod_summary,
                     dist_summary,
                     duration_secs,
                     extra=None):          # ← FIXED: accepts extra dict
        sessions = self._load()
        tf   = sum(focus_stats.values())   or 1
        te   = sum(emotion_stats.values()) or 1
        dm   = int(duration_secs / 60)
        ds   = int(duration_secs % 60)
        prod = prod_summary.get('average_score', 0)
        extra = extra or {}

        s = {
            'id':       len(sessions) + 1,
            'username': self.username,
            'date':     datetime.now().strftime('%Y-%m-%d'),
            'time':     datetime.now().strftime('%H:%M:%S'),
            'duration': f"{dm}m {ds}s",
            'duration_secs': int(duration_secs),

            'focus_high_pct':   round(focus_stats.get('High',   0) / tf * 100, 1),
            'focus_medium_pct': round(focus_stats.get('Medium', 0) / tf * 100, 1),
            'focus_low_pct':    round(focus_stats.get('Low',    0) / tf * 100, 1),

            'emotion_engaged_pct':    round(emotion_stats.get('Engaged',    0) / te * 100, 1),
            'emotion_bored_pct':      round(emotion_stats.get('Bored',      0) / te * 100, 1),
            'emotion_confused_pct':   round(emotion_stats.get('Confused',   0) / te * 100, 1),
            'emotion_frustrated_pct': round(emotion_stats.get('Frustrated', 0) / te * 100, 1),

            'avg_productivity':  prod,
            'peak_productivity': prod_summary.get('peak_score', prod),
            'grade':             self._grade(prod),

            'total_distractions':    dist_summary.get('total_distractions',    0),
            'distractions_per_hour': dist_summary.get('distractions_per_hour', 0),

            # Extra fields saved from session_state
            'yawn_count':       extra.get('yawn_count',        0),
            'max_focus_streak': extra.get('max_focus_streak',  0),
        }
        sessions.append(s)
        self._save(sessions)
        return s

    def get_all_sessions(self):
        return self._load()

    def get_last_n(self, n=10):
        return self._load()[-n:]

    def clear(self):
        self._save([])

    def _grade(self, score):
        if score >= 85: return 'A'
        if score >= 70: return 'B'
        if score >= 55: return 'C'
        if score >= 40: return 'D'
        return 'F'

    @staticmethod
    def get_all_users_sessions():
        all_s = []
        if not os.path.exists(LOGS_DIR):
            return []
        for f in os.listdir(LOGS_DIR):
            if not f.endswith('_sessions.json'):
                continue
            path = os.path.join(LOGS_DIR, f)
            try:
                with open(path) as fh:
                    all_s.extend(json.load(fh))
            except:
                pass
        all_s.sort(key=lambda x: (
            x.get('date', ''),
            x.get('time', '')))
        return all_s