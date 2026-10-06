import json
import os
import statistics
from datetime import datetime, timedelta
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))
LOGS_DIR = os.path.join(
    BASE_DIR, 'session_logs', 'users')


class AnalyticsEngine:

    def __init__(self, username='default'):
        self.username = username
        self.sessions = self._load()

    def _load(self):
        path = os.path.join(
            LOGS_DIR,
            f"{self.username}_sessions.json")
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                return json.load(f)
        except:
            return []

    # ── Recommendations ──────────────────────

    def best_study_time(self):
        if not self.sessions:
            return None
        from collections import defaultdict
        hs = defaultdict(list)
        for s in self.sessions:
            try:
                h = int(s['time'].split(':')[0])
                hs[h].append(s['avg_productivity'])
            except:
                pass
        if not hs:
            return None
        best = max(hs,
            key=lambda h: statistics.mean(hs[h]))
        ranked = sorted(hs.items(),
            key=lambda x: statistics.mean(x[1]),
            reverse=True)
        return {
            'best_hour':     best,
            'best_hour_str': self._fmt_hour(best),
            'avg_score':     round(
                statistics.mean(hs[best]), 1),
            'ranked_hours': [{
                'hour':     h,
                'hour_str': self._fmt_hour(h),
                'avg':      round(
                    statistics.mean(v), 1)
            } for h, v in ranked[:5]]
        }

    def recommended_break_interval(self):
        if len(self.sessions) < 3:
            return {
                'recommended_mins': 45,
                'tip': 'Based on standard '
                       'Pomodoro technique.'
            }
        drops = []
        for s in self.sessions:
            dur = s.get('duration_secs', 0) / 60
            low = s.get('focus_low_pct', 0)
            if low > 25 and dur > 10:
                drops.append(dur * 0.75)
        if not drops:
            return {
                'recommended_mins': 45,
                'tip': 'Keep sessions consistent.'
            }
        avg  = statistics.mean(drops)
        mins = max(15, min(60, int(avg * 0.80)))
        return {
            'recommended_mins': mins,
            'tip': f'Focus drops after ~{int(avg)}'
                   f' mins. Break at {mins} mins.'
        }

    def distraction_emotion_analysis(self):
        if not self.sessions:
            return {
                'worst_emotion': 'N/A',
                'tip': 'Complete more sessions.'
            }
        em = defaultdict(list)
        for s in self.sessions:
            d = s.get('total_distractions', 0)
            for e in ['Bored','Confused',
                      'Frustrated','Engaged']:
                p = s.get(
                    f'emotion_{e.lower()}_pct', 0)
                if p > 20:
                    em[e].append(d)
        if not em:
            return {
                'worst_emotion': 'Engaged',
                'tip': 'Great emotional control!'
            }
        worst = max(em,
            key=lambda e: statistics.mean(em[e]))
        tips = {
            'Bored':
                'Boredom causes distractions. '
                'Try harder challenges.',
            'Confused':
                'Confusion triggers distractions. '
                'Break tasks into steps.',
            'Frustrated':
                'Frustration hurts focus. '
                'Take short breaks when stuck.',
            'Engaged':
                'You focus best when engaged. '
                'Seek interesting material.'
        }
        return {
            'worst_emotion': worst,
            'avg_distractions': round(
                statistics.mean(em[worst]), 1),
            'tip': tips.get(worst, '')
        }

    def weekly_improvement_tips(self):
        today    = datetime.now().date()
        wk_start = today - timedelta(
            days=today.weekday())
        lw_start = wk_start - timedelta(days=7)

        tw, lw = [], []
        for s in self.sessions:
            try:
                d = datetime.strptime(
                    s['date'],'%Y-%m-%d').date()
                if d >= wk_start:
                    tw.append(s['avg_productivity'])
                elif d >= lw_start:
                    lw.append(s['avg_productivity'])
            except:
                pass

        ta = round(statistics.mean(tw),1) if tw else 0
        la = round(statistics.mean(lw),1) if lw else 0
        ch = round(ta - la, 1)

        tips = []
        if not tw:
            tips.append(
                '📅 No sessions this week. '
                'Start one today!')
        elif ch > 10:
            tips.append(
                '🚀 Excellent improvement! '
                'Keep the momentum.')
        elif ch > 0:
            tips.append(
                '📈 Slight improvement. '
                'Consistency is key!')
        elif ch < -10:
            tips.append(
                '📉 Focus dropped this week. '
                'Try shorter sessions.')
        else:
            tips.append(
                '➡️ Stable performance. '
                'Push for improvement!')

        if tw:
            this_sessions = [
                s for s in self.sessions
                if self._this_week(s)]
            ad = statistics.mean([
                s.get('total_distractions', 0)
                for s in this_sessions]) \
                if this_sessions else 0
            if ad > 5:
                tips.append(
                    '⚠️ High distractions. '
                    'Silence notifications.')
            hp = [s.get('focus_high_pct', 0)
                  for s in this_sessions]
            if hp and statistics.mean(hp) < 50:
                tips.append(
                    '🎯 High focus below 50%. '
                    'Try morning sessions.')
            ep = [s.get(
                'emotion_engaged_pct', 0)
                  for s in this_sessions]
            if ep and statistics.mean(ep) < 40:
                tips.append(
                    '😴 Low engagement. '
                    'Vary your study material.')

        if len(tips) < 3:
            tips.append(
                '💡 Aim for 3+ sessions per '
                'day for best results.')

        return {
            'tips':               tips[:5],
            'this_week_avg':      ta,
            'last_week_avg':      la,
            'change':             ch,
            'this_week_sessions': len(tw),
        }

    def predict_todays_productivity(self):
        if not self.sessions:
            return {
                'predicted':  70,
                'confidence': 'Low',
                'weekday':
                    datetime.now().strftime('%A'),
                'tip': 'Complete more sessions '
                       'to improve predictions.'
            }
        wd = datetime.now().weekday()
        same = [
            s['avg_productivity']
            for s in self.sessions
            if self._get_weekday(s) == wd]

        if same:
            pred = round(
                statistics.mean(same), 0)
            conf = ('High' if len(same) >= 5
                    else 'Medium'
                    if len(same) >= 2
                    else 'Low')
        else:
            pred = round(statistics.mean([
                s['avg_productivity']
                for s in self.sessions]), 0)
            conf = 'Low'

        tips = {
            'High':   'Based on strong history.',
            'Medium': 'Build more consistency!',
            'Low':    'Keep logging sessions!'
        }
        return {
            'predicted':   int(pred),
            'confidence':  conf,
            'weekday':
                datetime.now().strftime('%A'),
            'tip':         tips[conf],
            'sample_size': len(same)
        }

    # ── Analytics ────────────────────────────

    def daily_summary(self):
        today = datetime.now().strftime('%Y-%m-%d')
        yest  = (datetime.now() -
                 timedelta(days=1))\
                .strftime('%Y-%m-%d')
        ts = [s for s in self.sessions
              if s['date'] == today]
        ys = [s for s in self.sessions
              if s['date'] == yest]

        def summarise(lst):
            if not lst: return None
            return {
                'sessions':   len(lst),
                'avg_prod':   round(
                    statistics.mean([
                        s['avg_productivity']
                        for s in lst]), 1),
                'avg_focus':  round(
                    statistics.mean([
                        s['focus_high_pct']
                        for s in lst]), 1),
                'total_dist': sum(
                    s.get(
                        'total_distractions', 0)
                    for s in lst),
            }
        return {
            'date':      today,
            'today':     summarise(ts),
            'yesterday': summarise(ys),
        }

    def weekly_trends(self):
        result = {}
        today  = datetime.now().date()
        for i in range(6, -1, -1):
            d  = today - timedelta(days=i)
            ds = d.strftime('%Y-%m-%d')
            day = [s for s in self.sessions
                   if s['date'] == ds]
            if day:
                result[ds] = {
                    'sessions': len(day),
                    'avg_prod': round(
                        statistics.mean([
                            s['avg_productivity']
                            for s in day]), 1),
                    'avg_focus': round(
                        statistics.mean([
                            s['focus_high_pct']
                            for s in day]), 1),
                }
            else:
                result[ds] = None
        return result

    def monthly_report(self):
        now   = datetime.now()
        month = now.strftime('%Y-%m')
        ms    = [s for s in self.sessions
                 if s['date'].startswith(month)]
        if not ms:
            return None
        prods  = [s['avg_productivity']
                  for s in ms]
        gc     = defaultdict(int)
        for s in ms:
            gc[s['grade']] += 1
        best_g = max(gc, key=gc.get)

        weekly = defaultdict(list)
        for s in ms:
            try:
                d  = datetime.strptime(
                    s['date'], '%Y-%m-%d')
                wn = f"W{d.isocalendar()[1]}"
                weekly[wn].append(
                    s['avg_productivity'])
            except:
                pass

        return {
            'month':          now.strftime(
                '%B %Y'),
            'total_sessions': len(ms),
            'avg_prod':       round(
                statistics.mean(prods), 1),
            'total_study_hrs': round(
                sum(s.get('duration_secs', 0)
                    for s in ms) / 3600, 1),
            'best_grade':     best_g,
            'grade_counts':   dict(gc),
            'weekly_breakdown': {
                wn: {
                    'avg_prod': round(
                        statistics.mean(v), 1),
                    'sessions': len(v)
                }
                for wn, v in
                sorted(weekly.items())
            }
        }

    def personal_bests(self):
        if not self.sessions:
            return None
        prods   = [s['avg_productivity']
                   for s in self.sessions]
        best    = max(self.sessions,
            key=lambda s: s['avg_productivity'])
        longest = max(self.sessions,
            key=lambda s: s.get(
                'duration_secs', 0))

        streak  = 0
        today   = datetime.now().date()
        check   = today
        date_set = {s['date']
                    for s in self.sessions}
        while check.strftime(
                '%Y-%m-%d') in date_set:
            streak += 1
            check  -= timedelta(days=1)

        weekly = defaultdict(list)
        for s in self.sessions:
            try:
                d  = datetime.strptime(
                    s['date'], '%Y-%m-%d')
                wn = d.isocalendar()[:2]
                weekly[wn].append(
                    s['avg_productivity'])
            except:
                pass
        bwa = max(
            (statistics.mean(v)
             for v in weekly.values()),
            default=0)

        return {
            'best_session': {
                'score': best['avg_productivity'],
                'date':  best['date'],
                'grade': best['grade'],
            },
            'longest_session': {
                'duration':
                    longest.get('duration','—'),
                'date': longest['date'],
            },
            'current_streak':  streak,
            'best_week_avg':   round(bwa, 1),
            'total_sessions':  len(self.sessions),
            'total_study_hrs': round(
                sum(s.get('duration_secs', 0)
                    for s in self.sessions)
                / 3600, 1),
            'overall_avg':     round(
                statistics.mean(prods), 1),
        }

    def best_vs_worst(self):
        if len(self.sessions) < 2:
            return None
        best  = max(self.sessions,
            key=lambda s: s['avg_productivity'])
        worst = min(self.sessions,
            key=lambda s: s['avg_productivity'])
        return {
            'best': {
                'date':  best['date'],
                'time':  best['time'][:5],
                'productivity':
                    best['avg_productivity'],
                'focus_high':
                    best['focus_high_pct'],
                'engaged':
                    best['emotion_engaged_pct'],
                'distractions':
                    best['total_distractions'],
                'duration_mins': round(
                    best.get(
                        'duration_secs',0)/60,1),
            },
            'worst': {
                'date':  worst['date'],
                'time':  worst['time'][:5],
                'productivity':
                    worst['avg_productivity'],
                'focus_high':
                    worst['focus_high_pct'],
                'engaged':
                    worst['emotion_engaged_pct'],
                'distractions':
                    worst['total_distractions'],
                'duration_mins': round(
                    worst.get(
                        'duration_secs',0)/60,1),
            },
            'difference': round(
                best['avg_productivity'] -
                worst['avg_productivity'], 1)
        }

    def heatmap_data(self):
        days  = ['Mon','Tue','Wed',
                 'Thu','Fri','Sat','Sun']
        hours = list(range(6, 24))
        grid  = {d: {h: []
                     for h in hours}
                 for d in days}
        for s in self.sessions:
            try:
                dt  = datetime.strptime(
                    s['date'], '%Y-%m-%d')
                day = days[dt.weekday()]
                hr  = int(
                    s['time'].split(':')[0])
                if hr in hours:
                    grid[day][hr].append(
                        s['avg_productivity'])
            except:
                pass
        result = {}
        for d in days:
            result[d] = {}
            for h in hours:
                v = grid[d][h]
                result[d][h] = round(
                    statistics.mean(v), 0) \
                    if v else None
        return result

    def session_comparison(self):
        if len(self.sessions) < 2:
            return None
        this_s = self.sessions[-1]
        last_s = self.sessions[-2]
        keys   = [
            'avg_productivity',
            'focus_high_pct',
            'emotion_engaged_pct',
            'total_distractions',
            'focus_low_pct',
        ]
        return {
            'this':       {k: this_s.get(k, 0)
                           for k in keys},
            'last':       {k: last_s.get(k, 0)
                           for k in keys},
            'this_date':  this_s.get('date',''),
            'last_date':  last_s.get('date',''),
            'this_grade': this_s.get('grade','—'),
            'last_grade': last_s.get('grade','—'),
        }

    def full_report(self):
        return {
            'recommendations': {
                'best_study_time':
                    self.best_study_time(),
                'break_interval':
                    self.recommended_break_interval(),
                'distraction_emotion':
                    self.distraction_emotion_analysis(),
                'weekly_tips':
                    self.weekly_improvement_tips(),
                'predicted_today':
                    self.predict_todays_productivity(),
            },
            'analytics': {
                'daily':         self.daily_summary(),
                'weekly':        self.weekly_trends(),
                'monthly':       self.monthly_report(),
                'bests':         self.personal_bests(),
                'best_vs_worst': self.best_vs_worst(),
                'heatmap':       self.heatmap_data(),
                'comparison':
                    self.session_comparison(),
            }
        }

    # ── Helpers ───────────────────────────────

    def _fmt_hour(self, h):
        if h == 0:  return '12 AM'
        if h < 12:  return f'{h} AM'
        if h == 12: return '12 PM'
        return f'{h-12} PM'

    def _get_weekday(self, s):
        try:
            return datetime.strptime(
                s['date'],
                '%Y-%m-%d').weekday()
        except:
            return -1

    def _this_week(self, s):
        try:
            today    = datetime.now().date()
            wk_start = today - timedelta(
                days=today.weekday())
            d = datetime.strptime(
                s['date'],
                '%Y-%m-%d').date()
            return d >= wk_start
        except:
            return False