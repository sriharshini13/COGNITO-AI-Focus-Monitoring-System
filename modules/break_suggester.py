import json
import os
import statistics
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
LOGS_DIR = os.path.join(
    BASE_DIR, 'session_logs', 'users')

class BreakSuggester:

    def __init__(self, username='default'):
        self.username        = username
        self.sessions        = self._load()
        self.default_mins    = 45
        self.personal_mins   = self._analyse()
        self.last_break_time = None
        self.next_break_mins = self.personal_mins

    def _load(self):
        path = os.path.join(
            LOGS_DIR,
            f"{self.username}_sessions.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return []

    def _analyse(self):
        if len(self.sessions) < 3:
            return self.default_mins
        drop_times = []
        for s in self.sessions:
            dur_mins = s.get(
                'duration_secs', 0) / 60
            low_pct  = s.get(
                'focus_low_pct', 0)
            if low_pct > 25 and dur_mins > 10:
                drop_times.append(
                    dur_mins * 0.75)
        if not drop_times:
            return self.default_mins
        avg_drop  = statistics.mean(drop_times)
        suggested = max(15, int(avg_drop * 0.80))
        return min(suggested, 60)

    def start_session(self):
        self.last_break_time = datetime.now()

    def check_break(self, session_secs,
                    recent_focus_history):
        mins_elapsed   = session_secs / 60
        time_due       = mins_elapsed >= \
                         self.next_break_mins
        focus_declining = False
        if len(recent_focus_history) >= 20:
            recent    = recent_focus_history[-20:]
            low_count = sum(
                1 for f in recent
                if f == 'Low')
            focus_declining = low_count >= 12

        if time_due or focus_declining:
            reason = 'time' if time_due \
                     else 'focus_decline'
            return self._build_suggestion(
                mins_elapsed, reason)

        if mins_elapsed >= (
                self.next_break_mins - 5):
            return {
                'type':       'upcoming',
                'message':    'Break in ~5 minutes',
                'mins_until': int(
                    self.next_break_mins -
                    mins_elapsed),
                'reason':     'upcoming'
            }
        return None

    def _build_suggestion(self,
                          mins_elapsed,
                          reason):
        import random
        duration   = self._suggest_duration(
            mins_elapsed)
        next_break = mins_elapsed + \
                     self.personal_mins + \
                     duration
        self.next_break_mins = next_break
        messages = {
            'time': [
                f'⏰ {self.personal_mins} min'
                f' session done! Take a'
                f' {duration}-min break.',
                f'☕ Time for a break!'
                f' ({duration} mins recommended)',
            ],
            'focus_decline': [
                '📉 Focus declining — a short'
                ' break will help you recharge!',
                f'🧠 Your brain needs rest.'
                f' Take {duration} mins off.',
            ]
        }
        msg = random.choice(
            messages.get(
                reason, messages['time']))
        return {
            'type':            'break_now',
            'message':         msg,
            'duration_mins':   duration,
            'reason':          reason,
            'next_break_mins': int(next_break),
        }

    def _suggest_duration(self, mins_elapsed):
        if mins_elapsed < 30:   return 5
        elif mins_elapsed < 60: return 7
        elif mins_elapsed < 90: return 10
        else:                   return 15

    def get_summary(self):
        return {
            'personal_break_interval':
                self.personal_mins,
            'based_on_sessions':
                len(self.sessions),
            'next_break_at_mins':
                self.next_break_mins,
        }