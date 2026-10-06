from flask import (Flask, render_template,
                   Response, jsonify, request,
                   redirect, url_for,
                   session as flask_session,
                   make_response)
import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
import os
import sys
import csv
import io
import random
from datetime import datetime, timedelta
from collections import defaultdict
from werkzeug.security import (
    generate_password_hash,
    check_password_hash)
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

from modules.distraction_counter import DistractionCounter
from modules.productivity_score import ProductivityScore
from modules.session_logger import SessionLogger
from modules.analytics_engine import AnalyticsEngine
from modules.confidence_scorer import ConfidenceScorer
from modules.smart_alerts import SmartAlertSystem
from modules.break_suggester import BreakSuggester

app = Flask(__name__)
app.secret_key = 'cognito_secret_2024'

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USERS_FILE = os.path.join(BASE_DIR, 'session_logs', 'users.json')
UPLOAD_DIR = os.path.join(BASE_DIR, 'webapp', 'static', 'uploads')
FACE_MODEL = os.path.join(BASE_DIR, 'face_landmarker.task')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'session_logs', 'users'), exist_ok=True)
os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)

print("Loading models...")
focus_model   = joblib.load(os.path.join(BASE_DIR, 'models', 'cognito_model.pkl'))
emotion_model = joblib.load(os.path.join(BASE_DIR, 'models', 'emotion_model.pkl'))
print("Models loaded!")

face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
        num_faces=1,
        min_face_detection_confidence=0.5))

# ── Feature helpers ──────────────────────────────────────────────────────────
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_EAR(lm, w, h):
    L = [(lm[i].x*w, lm[i].y*h) for i in [33,160,158,133,153,144]]
    R = [(lm[i].x*w, lm[i].y*h) for i in [362,385,387,263,373,380]]
    def ear(e):
        return (euclidean(e[1],e[5]) + euclidean(e[2],e[4])) / (2.0*euclidean(e[0],e[3]))
    return (ear(L) + ear(R)) / 2.0

def get_MAR(lm, w, h):
    m = [(lm[i].x*w, lm[i].y*h) for i in [61,291,13,14,78,308]]
    return (euclidean(m[2],m[3]) + euclidean(m[4],m[5])) / (2.0*euclidean(m[0],m[1]))

def get_gaze(lm, w, h):
    return ((lm[468].x+lm[473].x)/2.0, (lm[468].y+lm[473].y)/2.0)

def get_head_pose(lm, w, h):
    return ((lm[1].y - lm[152].y), (lm[263].x - lm[33].x))

def calc_attention(ear, mar, gx, gy, pitch, yaw):
    score = 100.0
    if ear < 0.15:    score -= 40
    elif ear < 0.20:  score -= 20
    elif ear < 0.25:  score -= 10
    if mar > 0.6:     score -= 25
    elif mar > 0.4:   score -= 10
    gd = abs(gx-0.5) + abs(gy-0.5)
    if gd > 0.3:      score -= 20
    elif gd > 0.15:   score -= 10
    if abs(yaw) < 0.1:    score -= 15
    if abs(pitch) > 0.15: score -= 10
    return max(0, min(100, int(score)))

# ── User helpers ─────────────────────────────────────────────────────────────
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(u):
    with open(USERS_FILE, 'w') as f:
        json.dump(u, f, indent=2)

def get_current_user():
    if 'username' not in flask_session:
        return None
    return load_users().get(flask_session['username'])

def login_required(f):
    from functools import wraps
    @wraps(f)
    def dec(*a, **kw):
        if 'username' not in flask_session:
            return redirect(url_for('login'))
        return f(*a, **kw)
    return dec

def admin_required(f):
    from functools import wraps
    @wraps(f)
    def dec(*a, **kw):
        if 'username' not in flask_session:
            return redirect(url_for('login'))
        u = get_current_user()
        if not u or not u.get('is_admin'):
            return redirect(url_for('index'))
        return f(*a, **kw)
    return dec

# ── Profiles ──────────────────────────────────────────────────────────────────
PROFILES = {
    'student': {
        'name': 'Student', 'icon': '👨‍🎓', 'color': '#89b4fa',
        'gradient': 'linear-gradient(135deg,#89b4fa,#cba6f7)',
        'description': 'E-learning & online classes',
        'low_focus_alert_secs': 300, 'drowsy_ear_threshold': 0.20,
        'emergency_ear': 0.15, 'break_interval_secs': 2700,
        'metrics': ['Focus','Emotion','Attention','Productivity','Distractions'],
        'alerts': {
            'low_focus': 'Focus dropping! Try refocusing.',
            'drowsy': 'Feeling sleepy? Short break!',
            'break': 'Time for a 5 min break!',
            'yawn': 'Yawning! Rest your eyes.',
            'emergency': 'Very drowsy! Rest immediately!'
        }
    },
    'worker': {
        'name': 'Remote Worker', 'icon': '💻', 'color': '#a6e3a1',
        'gradient': 'linear-gradient(135deg,#a6e3a1,#94e2d5)',
        'description': 'Work from home productivity',
        'low_focus_alert_secs': 420, 'drowsy_ear_threshold': 0.20,
        'emergency_ear': 0.15, 'break_interval_secs': 3600,
        'metrics': ['Focus','Emotion','Attention','Productivity','Distractions'],
        'alerts': {
            'low_focus': 'Productivity dropping! Refocus.',
            'drowsy': 'Tired! Coffee break?',
            'break': '1 hour done. Take a break!',
            'yawn': 'Fatigue detected! Step away.',
            'emergency': 'Extreme fatigue! Rest now!'
        }
    },
    'driver': {
        'name': 'Driver', 'icon': '🚗', 'color': '#f38ba8',
        'gradient': 'linear-gradient(135deg,#f38ba8,#fab387)',
        'description': 'Drowsiness & alertness detection',
        'low_focus_alert_secs': 30, 'drowsy_ear_threshold': 0.22,
        'emergency_ear': 0.18, 'break_interval_secs': 5400,
        'metrics': ['Attention','Drowsiness','Gaze'],
        'alerts': {
            'low_focus': '⚠️ Stay focused on the road!',
            'drowsy': '🚨 DROWSINESS! Pull over safely!',
            'break': '🛑 Drive time exceeded!',
            'yawn': '⚠️ Too tired to drive!',
            'emergency': '🚨 EMERGENCY! Pull over NOW!'
        }
    },
    'gamer': {
        'name': 'Gamer', 'icon': '🎮', 'color': '#cba6f7',
        'gradient': 'linear-gradient(135deg,#cba6f7,#89b4fa)',
        'description': 'Focus & reaction monitoring',
        'low_focus_alert_secs': 180, 'drowsy_ear_threshold': 0.18,
        'emergency_ear': 0.14, 'break_interval_secs': 3600,
        'metrics': ['Focus','Attention','Distractions'],
        'alerts': {
            'low_focus': 'Focus slipping! May affect performance.',
            'drowsy': 'Eye strain! Rest eyes.',
            'break': '1 hour gaming! Take a break.',
            'yawn': 'Fatigue! Hydrate and stretch.',
            'emergency': 'Severe eye strain! Stop and rest!'
        }
    },
    'medical': {
        'name': 'Medical Pro', 'icon': '🏥', 'color': '#fab387',
        'gradient': 'linear-gradient(135deg,#fab387,#f9e2af)',
        'description': 'Surgeon & doctor alertness',
        'low_focus_alert_secs': 60, 'drowsy_ear_threshold': 0.22,
        'emergency_ear': 0.17, 'break_interval_secs': 5400,
        'metrics': ['Focus','Attention','Drowsiness'],
        'alerts': {
            'low_focus': '⚠️ Alertness dropping!',
            'drowsy': '🚨 Drowsiness! Notify supervisor.',
            'break': 'Extended session! Consider handover.',
            'yawn': '⚠️ Fatigue signs detected!',
            'emergency': '🚨 CRITICAL! Immediate rest!'
        }
    },
    'athlete': {
        'name': 'Athlete', 'icon': '🏃', 'color': '#94e2d5',
        'gradient': 'linear-gradient(135deg,#94e2d5,#a6e3a1)',
        'description': 'Mental focus training',
        'low_focus_alert_secs': 120, 'drowsy_ear_threshold': 0.20,
        'emergency_ear': 0.15, 'break_interval_secs': 1800,
        'metrics': ['Focus','Attention','Emotion'],
        'alerts': {
            'low_focus': 'Mental focus dropping! Recenter.',
            'drowsy': 'Fatigue detected! Recovery needed.',
            'break': '30 min done! Recovery time.',
            'yawn': 'Physical fatigue showing! Rest.',
            'emergency': 'Critical fatigue! Stop training!'
        }
    }
}

# ── Quiz bank ────────────────────────────────────────────────────────────────
QUIZ_BANK = {
    'student': [
        {'q': 'What does EAR stand for in drowsiness detection?',
         'opts': ['Eye Aspect Ratio','Eye Alert Rate','Eyelid Area Ratio','Eye Attention Record'],
         'ans': 0, 'explanation': 'EAR = Eye Aspect Ratio. Measures eye openness using 6 landmark points per eye.'},
        {'q': 'Which ML model does COGNITO use for focus detection?',
         'opts': ['Neural Network','Random Forest','SVM','KNN'],
         'ans': 1, 'explanation': 'Random Forest trained on 9000+ video clips classifies focus as High/Medium/Low.'},
        {'q': 'What is the Pomodoro technique?',
         'opts': ['25 min work, 5 min break','45 min work, 15 min break','50 min work, 10 min break','30 min work, 5 min break'],
         'ans': 0, 'explanation': '25 min focused work + 5 min break = 1 Pomodoro. After 4, take a longer break.'},
        {'q': 'What causes most distractions during study?',
         'opts': ['Noise','Phone notifications','Hunger','Mind wandering'],
         'ans': 3, 'explanation': 'Research shows mind wandering is the #1 cause. External triggers are secondary.'},
        {'q': 'Best lighting for studying?',
         'opts': ['Dim light','Bright overhead','Natural daylight','No light'],
         'ans': 2, 'explanation': 'Natural daylight reduces eye strain and improves alertness and concentration.'},
        {'q': 'How many hours of sleep improves memory consolidation?',
         'opts': ['4-5 hours','6-7 hours','7-9 hours','10+ hours'],
         'ans': 2, 'explanation': '7-9 hours is optimal. Sleep consolidates memories and clears brain waste products.'},
        {'q': 'What is spaced repetition?',
         'opts': ['Studying same topic daily','Reviewing at increasing intervals','Taking breaks every 5 mins','Reading topics in random order'],
         'ans': 1, 'explanation': 'Spaced repetition reviews material at growing intervals, maximising long-term retention.'},
        {'q': 'What is active recall?',
         'opts': ['Re-reading your notes','Testing yourself on the material','Highlighting key points','Listening to lectures'],
         'ans': 1, 'explanation': 'Active recall forces your brain to retrieve information, strengthening memory far more than re-reading.'},
        {'q': 'How long does it typically take to form a habit?',
         'opts': ['7 days','21 days','66 days','100 days'],
         'ans': 2, 'explanation': 'Research by Phillippa Lally found habits take an average of 66 days to form automatically.'},
    ],
    'worker': [
        {'q': 'What is deep work?',
         'opts': ['Working late at night','Distraction-free focused work','Working from home','Multitasking'],
         'ans': 1, 'explanation': 'Cal Newport defines deep work as cognitively demanding tasks done without distraction.'},
        {'q': 'The 20-20-20 rule means?',
         'opts': ['Every 20 mins look 20ft away for 20 secs','20 tasks in 20 mins each','20 sec exercise every 20 mins','20 deep breaths every 20 mins'],
         'ans': 0, 'explanation': 'Reduces eye strain. Every 20 mins, look at something 20 feet away for 20 seconds.'},
        {'q': 'Best way to handle interruptions at work?',
         'opts': ['Respond immediately always','Ignore all messages','Batch check at set times','Work with notifications on'],
         'ans': 2, 'explanation': 'Batching notifications reduces context switching which costs 23 mins of focus per interrupt.'},
        {'q': 'What is time blocking?',
         'opts': ['Blocking distracting websites','Scheduling specific tasks to fixed time slots','Working without breaks','Setting alarms every hour'],
         'ans': 1, 'explanation': 'Time blocking assigns specific work to calendar slots, reducing decision fatigue.'},
        {'q': 'What does GTD stand for?',
         'opts': ['Get Things Done','Go To Dashboard','Goal Tracking Daily','General Task Design'],
         'ans': 0, 'explanation': 'Getting Things Done (GTD) by David Allen is a productivity method for capturing and organising tasks.'},
    ],
    'driver': [
        {'q': 'Most dangerous time for drowsy driving?',
         'opts': ['Morning rush hour','2-4 AM and 1-3 PM','Evening commute','Midnight'],
         'ans': 1, 'explanation': 'Circadian rhythm dips at 2-4AM and 1-3PM, making these the highest risk periods.'},
        {'q': 'First sign of drowsiness while driving?',
         'opts': ['Blurry vision','Heavy eyelids','Frequent yawning','Slow reaction'],
         'ans': 2, 'explanation': 'Frequent yawning is the earliest warning sign. Pull over before it gets worse.'},
        {'q': 'Safe action when drowsy while driving?',
         'opts': ['Open the window','Turn up the music','Pull over and rest','Drink coffee and continue'],
         'ans': 2, 'explanation': 'Only pulling over and resting is truly safe. Coffee takes 20 mins to work.'},
        {'q': 'How does driving drowsy compare to drunk driving?',
         'opts': ['Much safer','Slightly safer','About the same','More dangerous'],
         'ans': 2, 'explanation': 'Studies show 24hrs without sleep impairs driving similar to 0.10% blood alcohol.'},
    ],
    'gamer': [
        {'q': 'What is flow state?',
         'opts': ['When the game lags','Complete absorption and focus','Playing with friends','Winning a match'],
         'ans': 1, 'explanation': 'Flow is optimal experience — complete immersion where time seems to disappear.'},
        {'q': 'How long should gaming sessions be for eye health?',
         'opts': ['2 hours max','1 hour with breaks','30 mins','No limit'],
         'ans': 1, 'explanation': '1 hour with 5-10 min breaks prevents eye strain and maintains sharp focus.'},
        {'q': 'What improves reaction time?',
         'opts': ['Energy drinks','Sleep and hydration','Longer sessions','Skipping breaks'],
         'ans': 1, 'explanation': 'Sleep and hydration are the most evidence-based ways to improve reaction time.'},
        {'q': 'What is tilt in gaming?',
         'opts': ['Screen angle','Emotional state affecting performance','Controller sensitivity','Graphics setting'],
         'ans': 1, 'explanation': 'Tilt is an emotional state (usually frustration) that negatively impacts decision-making and performance.'},
    ],
    'medical': [
        {'q': 'Safe max continuous procedure duration?',
         'opts': ['6 hours','8 hours','4-5 hours max','12 hours'],
         'ans': 2, 'explanation': 'Beyond 4-5 hours, cognitive performance drops significantly increasing risk.'},
        {'q': 'Signs of decision fatigue?',
         'opts': ['Increased energy','Slower decisions, more errors','Better focus','Improved memory'],
         'ans': 1, 'explanation': 'Decision fatigue causes slower processing and increased error rates.'},
        {'q': 'Best micro-break activity for medical staff?',
         'opts': ['Check phone','Deep breathing exercises','Drink coffee','Read emails'],
         'ans': 1, 'explanation': 'Deep breathing activates the parasympathetic system, rapidly reducing stress.'},
        {'q': 'What is cognitive load?',
         'opts': ['Physical tiredness','Mental effort used in working memory','Eye strain level','Decision speed'],
         'ans': 1, 'explanation': 'Cognitive load is the mental effort being used in working memory. High load increases error rates.'},
    ],
    'athlete': [
        {'q': 'What is mental imagery?',
         'opts': ['Watching game footage','Visualizing performance in your mind','Reading sports books','Analyzing statistics'],
         'ans': 1, 'explanation': 'Mental imagery activates the same neural pathways as actual physical practice.'},
        {'q': 'Best time for mental focus training?',
         'opts': ['Just before competition','During warmup','Daily consistent practice','After competition'],
         'ans': 2, 'explanation': 'Daily practice builds neural pathways. Consistency beats intensity every time.'},
        {'q': 'What is the zone in sports?',
         'opts': ['A training location','Peak performance mental state','A coaching technique','A recovery method'],
         'ans': 1, 'explanation': 'The zone is a flow state where athletes perform at their absolute peak.'},
        {'q': 'What hormone increases during high-pressure performance?',
         'opts': ['Serotonin','Melatonin','Cortisol','Insulin'],
         'ans': 2, 'explanation': 'Cortisol (stress hormone) rises under pressure. Managing it is key to consistent peak performance.'},
    ],
}

# ── Points ────────────────────────────────────────────────────────────────────
def calc_points(session):
    pts  = int(session.get('avg_productivity', 0)) * 2
    pts += int(session.get('focus_high_pct', 0))
    pts += max(0, 50 - int(session.get('total_distractions', 0)) * 5)
    pts += {'A':100,'B':75,'C':50,'D':25,'F':0}.get(session.get('grade','F'), 0)
    pts += min(50, int(session.get('duration_secs', 0) / 60))
    return max(0, pts)

# ── Motivational messages ─────────────────────────────────────────────────────
def get_motivational(score, focus, streak):
    if score >= 85:
        msgs = [
            "🔥 You're on fire! Keep it up!",
            "💪 Outstanding focus! Crushing it!",
            "⚡ Peak performance activated!",
            "🎯 Laser sharp! Nothing stops you!",
            "🏆 Champion level focus!",
            "🚀 You're in the zone! Stay there!",
        ]
    elif score >= 70:
        msgs = [
            "👍 Great work! Stay consistent!",
            "📈 Strong session! Keep pushing!",
            "✨ You're doing really well!",
            "🌟 Solid focus! Maintain this pace!",
            "💯 Above average! Well done!",
        ]
    elif score >= 50:
        msgs = [
            "💡 Good effort! Try to refocus.",
            "🎯 You've got this! Lock back in.",
            "📚 Stay with it — almost there!",
            "🧘 Take a breath and refocus.",
            "⬆️ You can push higher — try!",
        ]
    else:
        msgs = [
            "⚠️ Focus dropping — refocus now!",
            "🔄 Reset and try again — you can!",
            "💪 Shake it off, get back in zone!",
            "🧠 Breathe and reset your focus!",
            "🌊 Ride it out — focus will return!",
        ]
    if streak >= 50:
        msgs.append(f"🔥🔥 {streak} streak frames! Legendary!")
    elif streak >= 30:
        msgs.append(f"🔥 {streak} frame streak! Incredible!")
    return random.choice(msgs)

# ── Session state ─────────────────────────────────────────────────────────────
session_state = {
    'active':              False,
    'profile':             'student',
    'username':            None,
    'start_time':          None,
    'focus_label':         'Detecting...',
    'focus_confidence':    0,
    'focus_proba':         {},
    'emotion_label':       'Detecting...',
    'emotion_confidence':  0,
    'drowsiness_level':    'Awake',
    'drowsiness_severity': 'none',
    'attention_score':     0,
    'attention_label':     'Waiting...',
    'prod_score':          0,
    'distraction_count':   0,
    'alert':               None,
    'alert_type':          None,
    'alert_icon':          '🔔',
    'beep':                False,
    'beep_type':           'warning',
    'break_suggestion':    None,
    'admin_alert':         None,
    # ── Quiz state (FIXED) ───────────
    'quiz_due':            False,
    'last_quiz_time':      0,
    'quiz_shown_time':     0,   # NEW: when quiz was shown to frontend
    'quiz_streak':         0,
    # ── Focus tracking ──────────────
    'focus_stats':  defaultdict(int, {'High':0,'Medium':0,'Low':0}),
    'emotion_stats': defaultdict(int, {'Engaged':0,'Bored':0,'Confused':0,'Frustrated':0}),
    'focus_history':  [],
    'score_history':  [],
    'ear_history':    [],
    'yawn_count':     0,
    'current_ear':    0.3,
    # ── Streak & goal ───────────────
    'focus_streak':   0,
    'max_streak':     0,
    'goal_score':     75,
    'goal_reached':   False,
    # ── Motivational ────────────────
    'motivational':   '🎯 Session started!',
    'last_motiv_time': 0,
}

active_sessions      = {}
distraction_tracker  = DistractionCounter()
productivity_tracker = ProductivityScore()
confidence_scorer    = ConfidenceScorer(focus_model, emotion_model)
smart_alert_system   = SmartAlertSystem()
break_suggester      = BreakSuggester()
cap                  = None

# ════════════════════════════════════════════════════════════════════════════
#                              AUTH ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        data  = request.json
        un    = data.get('username','').strip()
        pw    = data.get('password','')
        users = load_users()
        if un in users and check_password_hash(users[un]['password'], pw):
            flask_session['username'] = un
            first = not users[un].get('onboarded', False)
            return jsonify({'success': True, 'redirect': '/onboarding' if first else '/'})
        return jsonify({'success': False, 'error': 'Invalid username or password'})
    return render_template('login.html')

@app.route('/register', methods=['POST'])
def register():
    data  = request.json
    un    = data.get('username','').strip()
    pw    = data.get('password','')
    name  = data.get('name','').strip()
    users = load_users()
    if not un or not pw or not name:
        return jsonify({'success': False, 'error': 'All fields required'})
    if un in users:
        return jsonify({'success': False, 'error': 'Username already taken'})
    if len(pw) < 6:
        return jsonify({'success': False, 'error': 'Password needs 6+ chars'})
    is_admin = len(users) == 0
    users[un] = {
        'username': un, 'name': name,
        'password': generate_password_hash(pw),
        'created': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'onboarded': False, 'photo': None,
        'default_profile': 'student',
        'total_sessions': 0, 'total_points': 0,
        'is_admin': is_admin,
    }
    save_users(users)
    flask_session['username'] = un
    return jsonify({'success': True, 'redirect': '/onboarding'})

@app.route('/logout')
def logout():
    un = flask_session.get('username')
    if un and un in active_sessions:
        del active_sessions[un]
    flask_session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/onboarding')
@login_required
def onboarding():
    user = get_current_user()
    return render_template('onboarding.html', user=user, profiles=PROFILES)

@app.route('/api/complete_onboarding', methods=['POST'])
@login_required
def complete_onboarding():
    data  = request.json
    users = load_users()
    un    = flask_session['username']
    users[un]['onboarded']       = True
    users[un]['default_profile'] = data.get('profile', 'student')
    save_users(users)
    return jsonify({'success': True})

@app.route('/profile')
@login_required
def profile_page():
    user     = get_current_user()
    un       = flask_session.get('username')
    logger   = SessionLogger(un)
    sessions = logger.get_all_sessions()
    engine   = AnalyticsEngine(un)
    bests    = engine.personal_bests()
    return render_template('profile.html', user=user, profiles=PROFILES, sessions=sessions, bests=bests)

@app.route('/api/update_profile', methods=['POST'])
@login_required
def update_profile():
    users = load_users()
    un    = flask_session['username']
    data  = request.json
    if 'name' in data:
        users[un]['name'] = data['name']
    if 'default_profile' in data:
        users[un]['default_profile'] = data['default_profile']
    save_users(users)
    return jsonify({'success': True})

@app.route('/api/upload_photo', methods=['POST'])
@login_required
def upload_photo():
    if 'photo' not in request.files:
        return jsonify({'success': False})
    file  = request.files['photo']
    un    = flask_session['username']
    fname = secure_filename(f"{un}_photo.jpg")
    path  = os.path.join(UPLOAD_DIR, fname)
    file.save(path)
    users = load_users()
    users[un]['photo'] = f"/static/uploads/{fname}"
    save_users(users)
    return jsonify({'success': True, 'photo': users[un]['photo']})

@app.route('/api/clear_history', methods=['POST'])
@login_required
def clear_history():
    un     = flask_session.get('username')
    logger = SessionLogger(un)
    logger.clear()
    users = load_users()
    if un in users:
        users[un]['total_sessions'] = 0
        users[un]['total_points']   = 0
        save_users(users)
    return jsonify({'success': True})

# ════════════════════════════════════════════════════════════════════════════
#                              MAIN ROUTES
# ════════════════════════════════════════════════════════════════════════════

@app.route('/')
@login_required
def index():
    user     = get_current_user()
    un       = flask_session.get('username')
    logger   = SessionLogger(un)
    sessions = logger.get_all_sessions()
    last_s   = sessions[-1] if sessions else None
    engine   = AnalyticsEngine(un)
    bests    = engine.personal_bests()
    return render_template('index.html', profiles=PROFILES, user=user,
                           last_session=last_s, total_sessions=len(sessions),
                           bests=bests)

@app.route('/monitor/<profile_id>')
@login_required
def monitor(profile_id):
    if profile_id not in PROFILES:
        profile_id = 'student'
    user = get_current_user()
    quiz = QUIZ_BANK.get(profile_id, [])
    return render_template('monitor.html',
        profile=PROFILES[profile_id],
        profile_id=profile_id,
        user=user,
        quiz_bank=json.dumps(quiz))

@app.route('/dashboard')
@login_required
def dashboard():
    user   = get_current_user()
    un     = flask_session.get('username')
    engine = AnalyticsEngine(un)
    report = engine.full_report()
    return render_template('dashboard.html', report=report, profiles=PROFILES, user=user)

@app.route('/report')
@login_required
def report():
    user     = get_current_user()
    un       = flask_session.get('username')
    logger   = SessionLogger(un)
    sessions = list(reversed(logger.get_all_sessions()))
    return render_template('report.html', sessions=sessions, user=user)

@app.route('/leaderboard')
@login_required
def leaderboard():
    user  = get_current_user()
    users = load_users()
    board = []
    for un, u in users.items():
        if u.get('is_admin'):
            continue
        board.append({
            'name': u['name'], 'username': un,
            'total_points': u.get('total_points', 0),
            'total_sessions': u.get('total_sessions', 0),
            'photo': u.get('photo'),
        })
    board.sort(key=lambda x: x['total_points'], reverse=True)
    for i, u in enumerate(board):
        u['rank'] = i + 1
    return render_template('leaderboard.html', board=board, user=user, current_user=user)

@app.route('/admin')
@admin_required
def admin():
    user  = get_current_user()
    users = load_users()
    all_u = [{**v, 'username': k, 'is_active': k in active_sessions}
             for k, v in users.items() if not v.get('is_admin')]
    try:
        all_s = SessionLogger.get_all_users_sessions()
    except:
        all_s = []
    return render_template('admin.html', user=user, all_users=all_u,
                           active_sessions=active_sessions,
                           sessions=list(reversed(all_s))[:50])

# ── Admin APIs ────────────────────────────────────────────────────────────────

@app.route('/api/admin/users')
@admin_required
def admin_users():
    users  = load_users()
    result = []
    for un, u in users.items():
        if u.get('is_admin'):
            continue
        active = active_sessions.get(un, {})
        result.append({
            'username': un, 'name': u['name'],
            'photo': u.get('photo'),
            'total_sessions': u.get('total_sessions', 0),
            'total_points': u.get('total_points', 0),
            'is_active': un in active_sessions,
            'focus_label': active.get('focus_label','—'),
            'attention_score': active.get('attention_score', 0),
            'prod_score': active.get('prod_score', 0),
            'elapsed': active.get('elapsed','—'),
            'profile': active.get('profile','—'),
            'drowsiness_level': active.get('drowsiness_level','—'),
            'alert': active.get('alert'),
        })
    return jsonify(result)

@app.route('/api/admin/send_alert', methods=['POST'])
@admin_required
def admin_send_alert():
    data   = request.json
    target = data.get('username')
    msg    = data.get('message','')
    if target in active_sessions:
        active_sessions[target]['admin_alert'] = {'message': msg, 'time': time.time()}
    return jsonify({'success': True})

@app.route('/api/admin/user_sessions/<username>')
@admin_required
def admin_user_sessions(username):
    logger   = SessionLogger(username)
    sessions = logger.get_all_sessions()
    return jsonify(list(reversed(sessions))[:20])

@app.route('/api/export_csv')
@login_required
def export_csv():
    un       = flask_session.get('username')
    logger   = SessionLogger(un)
    sessions = logger.get_all_sessions()
    si       = io.StringIO()
    fields   = ['date','time','duration','avg_productivity','grade',
                'focus_high_pct','focus_medium_pct','focus_low_pct',
                'emotion_engaged_pct','emotion_bored_pct',
                'total_distractions','distractions_per_hour','yawn_count']
    writer = csv.DictWriter(si, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for s in sessions:
        writer.writerow({k: s.get(k,'') for k in fields})
    resp = make_response(si.getvalue())
    resp.headers['Content-Disposition'] = f'attachment; filename={un}_sessions.csv'
    resp.headers['Content-type'] = 'text/csv'
    return resp

# ── Quiz routes (FULLY FIXED) ─────────────────────────────────────────────────

@app.route('/api/quiz_answer', methods=['POST'])
@login_required
def quiz_answer():
    data    = request.json
    correct = data.get('correct', False)
    un      = flask_session.get('username')

    # FIX: Always reset quiz_due and record time when answer comes in
    session_state['quiz_due']        = False
    session_state['last_quiz_time']  = time.time()
    session_state['quiz_shown_time'] = 0  # reset shown timer

    if correct:
        session_state['quiz_streak'] = session_state.get('quiz_streak', 0) + 1
        streak = session_state['quiz_streak']
        bonus  = 20 if streak >= 3 else 10
        if un:
            users = load_users()
            if un in users:
                users[un]['total_points'] = users[un].get('total_points', 0) + bonus
                save_users(users)
        msg = f'🎉 Correct! +{bonus} pts'
        if streak >= 3:
            msg += f' (🔥 {streak} streak!)'
    else:
        session_state['quiz_streak'] = 0
        msg = '❌ Wrong! Keep studying!'
        bonus = 0

    return jsonify({
        'success': True, 'points': bonus,
        'message': msg,
        'streak': session_state['quiz_streak']
    })

@app.route('/api/quiz_dismiss', methods=['POST'])
@login_required
def quiz_dismiss():
    """Called when user closes quiz without answering (timeout)"""
    session_state['quiz_due']        = False
    session_state['last_quiz_time']  = time.time()
    session_state['quiz_shown_time'] = 0
    return jsonify({'success': True})

@app.route('/api/save_note', methods=['POST'])
@login_required
def save_note():
    """Save a text note to the most recent session record"""
    data = request.json
    note = data.get('note', '').strip()
    if not note:
        return jsonify({'success': False})
    un       = flask_session.get('username')
    logger   = SessionLogger(un)
    sessions = logger.get_all_sessions()
    if sessions:
        sessions[-1]['note'] = note
        logger._save(sessions)
    return jsonify({'success': True})

@app.route('/api/sparkline')
@login_required
def sparkline():
    """Return last 7 session scores + labels for homepage sparkline"""
    un       = flask_session.get('username')
    logger   = SessionLogger(un)
    sessions = logger.get_last_n(7)
    scores   = [s.get('avg_productivity', 0) for s in sessions]
    labels   = [s.get('date','')[-5:] + ' ' + s.get('time','')[:5] for s in sessions]
    return jsonify({'scores': scores, 'labels': labels})

@app.route('/api/daily_goal', methods=['GET','POST'])
@login_required
def daily_goal():
    """Get or set today's study goal and compute progress"""
    un    = flask_session.get('username')
    users = load_users()

    if request.method == 'POST':
        data  = request.json
        goal  = int(data.get('goal_mins', 120))
        if un in users:
            users[un]['daily_goal_mins'] = goal
            save_users(users)
    else:
        goal = users.get(un, {}).get('daily_goal_mins', 120)

    # Calculate studied minutes today
    today    = datetime.now().strftime('%Y-%m-%d')
    logger   = SessionLogger(un)
    sessions = logger.get_all_sessions()
    today_secs = sum(
        s.get('duration_secs', 0)
        for s in sessions
        if s.get('date','') == today
    )
    studied_mins = round(today_secs / 60, 1)

    return jsonify({
        'goal_mins':    goal,
        'studied_mins': studied_mins,
        'pct':          min(100, round((studied_mins / goal) * 100)) if goal else 0
    })

@app.route('/api/set_goal', methods=['POST'])
@login_required
def set_goal():
    data = request.json
    session_state['goal_score']   = data.get('goal', 75)
    session_state['goal_reached'] = False
    return jsonify({'success': True})

# ════════════════════════════════════════════════════════════════════════════
#                           SESSION CONTROL
# ════════════════════════════════════════════════════════════════════════════

@app.route('/api/start_session', methods=['POST'])
@login_required
def start_session():
    global cap, distraction_tracker, productivity_tracker
    global smart_alert_system, break_suggester

    data       = request.json
    profile_id = data.get('profile', 'student')
    goal       = data.get('goal', 75)
    un         = flask_session.get('username')

    distraction_tracker  = DistractionCounter()
    productivity_tracker = ProductivityScore()
    smart_alert_system   = SmartAlertSystem()
    break_suggester      = BreakSuggester(un)
    break_suggester.start_session()

    now = time.time()
    session_state.update({
        'active':              True,
        'profile':             profile_id,
        'username':            un,
        'start_time':          now,
        'focus_label':         'Detecting...',
        'focus_confidence':    0,
        'focus_proba':         {},
        'emotion_label':       'Detecting...',
        'emotion_confidence':  0,
        'drowsiness_level':    'Awake',
        'drowsiness_severity': 'none',
        'attention_score':     0,
        'attention_label':     'Waiting...',
        'prod_score':          0,
        'distraction_count':   0,
        'alert':               None,
        'alert_type':          None,
        'alert_icon':          '🔔',
        'beep':                False,
        'beep_type':           'warning',
        'break_suggestion':    None,
        'admin_alert':         None,
        # Quiz: reset cleanly each session
        'quiz_due':            False,
        'last_quiz_time':      now,   # start timer from session start
        'quiz_shown_time':     0,
        'quiz_streak':         0,
        'focus_stats':  defaultdict(int, {'High':0,'Medium':0,'Low':0}),
        'emotion_stats': defaultdict(int, {'Engaged':0,'Bored':0,'Confused':0,'Frustrated':0}),
        'focus_history':  [],
        'score_history':  [],
        'ear_history':    [],
        'yawn_count':     0,
        'current_ear':    0.3,
        'focus_streak':   0,
        'max_streak':     0,
        'goal_score':     goal,
        'goal_reached':   False,
        'motivational':   '🎯 Session started! Good luck!',
        'last_motiv_time': now,
    })

    cap = cv2.VideoCapture(0)
    return jsonify({'status': 'started'})

@app.route('/api/stop_session', methods=['POST'])
@login_required
def stop_session():
    global cap
    session_state['active'] = False
    un = flask_session.get('username')

    if cap:
        cap.release()
        cap = None

    if un and un in active_sessions:
        del active_sessions[un]

    tf = sum(session_state['focus_stats'].values())
    if tf > 0:
        elapsed = time.time() - session_state['start_time']
        prod_s  = productivity_tracker.get_summary()
        dist_s  = distraction_tracker.get_summary(elapsed)

        logger = SessionLogger(un)
        extra  = {
            'yawn_count':       session_state.get('yawn_count', 0),
            'max_focus_streak': session_state.get('max_streak', 0),
        }
        saved = logger.save_session(
            dict(session_state['focus_stats']),
            dict(session_state['emotion_stats']),
            prod_s, dist_s, elapsed, extra)

        pts   = calc_points(saved)
        users = load_users()
        if un and un in users:
            users[un]['total_sessions'] = users[un].get('total_sessions', 0) + 1
            users[un]['total_points']   = users[un].get('total_points', 0) + pts
            save_users(users)

    return jsonify({'status': 'stopped'})

# ── Video feed ─────────────────────────────────────────────────────────────────
def generate_frames():
    global cap
    frame_count = 0

    while session_state['active'] and cap:
        ret, frame = cap.read()
        if not ret:
            break

        h, w   = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if frame_count % 5 == 0:
            result  = face_detector.detect(mp_img)
            prof    = PROFILES[session_state['profile']]
            now     = time.time()
            elapsed = now - session_state['start_time']

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                try:
                    ear       = get_EAR(lm, w, h)
                    mar       = get_MAR(lm, w, h)
                    gx, gy    = get_gaze(lm, w, h)
                    pitch,yaw = get_head_pose(lm, w, h)
                    feats     = [ear, mar, gx, gy, pitch, yaw]

                    focus, fconf, fproba   = confidence_scorer.predict_focus(feats)
                    emotion, econf, _      = confidence_scorer.predict_emotion(feats)
                    attn                   = calc_attention(ear, mar, gx, gy, pitch, yaw)
                    prod                   = productivity_tracker.calculate(focus, emotion, attn, True)
                    dl, dsev, _            = confidence_scorer.drowsiness_level(ear)
                    al, _                  = confidence_scorer.attention_level(attn)

                    # Focus streak
                    if focus == 'High':
                        session_state['focus_streak'] += 1
                        session_state['max_streak'] = max(
                            session_state['max_streak'],
                            session_state['focus_streak'])
                    else:
                        session_state['focus_streak'] = 0

                    # Goal check
                    if not session_state['goal_reached'] and prod >= session_state['goal_score']:
                        session_state['goal_reached'] = True

                    # Motivational every 30s
                    if now - session_state['last_motiv_time'] > 30:
                        session_state['motivational'] = get_motivational(
                            prod, focus, session_state['focus_streak'])
                        session_state['last_motiv_time'] = now

                    session_state.update({
                        'focus_label':         focus,
                        'focus_confidence':    fconf,
                        'focus_proba':         fproba,
                        'emotion_label':       emotion,
                        'emotion_confidence':  econf,
                        'drowsiness_level':    dl,
                        'drowsiness_severity': dsev,
                        'attention_score':     attn,
                        'attention_label':     al,
                        'prod_score':          prod,
                        'current_ear':         ear,
                        'distraction_count':   distraction_tracker.get_count(),
                    })

                    session_state['focus_stats'][focus]     += 1
                    session_state['emotion_stats'][emotion] += 1
                    session_state['focus_history'].append(focus)
                    session_state['score_history'].append(prod)
                    session_state['ear_history'].append(round(ear, 3))

                    if mar > 0.6:
                        session_state['yawn_count'] += 1

                    distraction_tracker.update(focus)

                    alerts, should_beep = smart_alert_system.update(focus, emotion, ear, attn, prof)

                    if should_beep:
                        session_state['beep'] = True
                        if dsev == 'severe':
                            session_state['beep_type'] = 'emergency'
                        elif focus == 'Low':
                            session_state['beep_type'] = 'warning'
                        else:
                            session_state['beep_type'] = 'info'

                    if alerts:
                        a = alerts[0]
                        session_state.update({
                            'alert':      a['message'],
                            'alert_type': a['type'],
                            'alert_icon': a['icon'],
                        })
                    else:
                        session_state['alert'] = None

                    brk = break_suggester.check_break(elapsed, session_state['focus_history'])
                    session_state['break_suggestion'] = brk

                    # Quiz every 10 mins — FIXED: auto-expire shown quiz after 45s
                    lqt = session_state.get('last_quiz_time', 0)
                    shown_t = session_state.get('quiz_shown_time', 0)
                    quiz_due = session_state.get('quiz_due', False)

                    # Auto-expire quiz if shown for >45s with no answer
                    if quiz_due and shown_t > 0 and (now - shown_t) > 45:
                        session_state['quiz_due']        = False
                        session_state['last_quiz_time']  = now
                        session_state['quiz_shown_time'] = 0

                    # Trigger new quiz every 600s (10 min) after last one
                    elif not quiz_due and elapsed > 60 and (now - lqt) > 600:
                        session_state['quiz_due']        = True
                        session_state['quiz_shown_time'] = now  # mark when it was triggered

                    # Active sessions tracking
                    un2 = session_state.get('username')
                    if un2:
                        mm, ss = divmod(int(elapsed), 60)
                        prev = active_sessions.get(un2, {})
                        active_sessions[un2] = {
                            'focus_label':      focus,
                            'attention_score':  attn,
                            'prod_score':       prod,
                            'elapsed':          f"{mm:02d}:{ss:02d}",
                            'profile':          session_state['profile'],
                            'drowsiness_level': dl,
                            'alert':            session_state['alert'],
                            'admin_alert':      prev.get('admin_alert'),
                        }

                    # Overlay on frame
                    fc = {'High':(0,220,100),'Medium':(0,165,255),'Low':(50,50,220)}.get(focus,(255,255,255))
                    cv2.rectangle(frame, (0,0), (w,62), (12,12,20), -1)
                    cv2.putText(frame, f"Focus: {focus} ({fconf:.0f}%)", (10,24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, fc, 2)
                    cv2.putText(frame, f"Attn:{attn}  Emotion:{emotion}  EAR:{ear:.3f}",
                                (10,48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160,160,160), 1)
                    mm2, ss2 = divmod(int(elapsed), 60)
                    cv2.putText(frame, f"{mm2:02d}:{ss2:02d}", (w-90,36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)

                    if dsev != 'none':
                        bc = {'mild':(0,200,255),'moderate':(0,100,255),'severe':(0,0,200)}.get(dsev,(0,0,200))
                        cv2.rectangle(frame, (0,h-40), (w,h), (12,12,20), -1)
                        cv2.putText(frame, f"! {dl}", (10,h-12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, bc, 2)

                except Exception as e:
                    print(f"Frame error: {e}")
            else:
                session_state['focus_label']   = 'No Face'
                session_state['emotion_label'] = ''

        ret2, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        frame_count += 1

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace;boundary=frame')

@app.route('/api/state')
def get_state():
    elapsed = int(time.time() - session_state['start_time']) if session_state['start_time'] else 0
    mm, ss  = divmod(elapsed, 60)
    tf = sum(session_state['focus_stats'].values()) or 1
    te = sum(session_state['emotion_stats'].values()) or 1

    un = session_state.get('username')
    admin_alert = None
    if un and un in active_sessions:
        aa = active_sessions[un].get('admin_alert')
        if aa and (time.time() - aa['time']) < 10:
            admin_alert = aa['message']
            active_sessions[un]['admin_alert'] = None

    beep      = session_state.get('beep', False)
    beep_type = session_state.get('beep_type', 'warning')
    session_state['beep'] = False  # reset after reading

    quiz_due = session_state.get('quiz_due', False)

    return jsonify({
        'focus_label':         session_state['focus_label'],
        'focus_confidence':    session_state['focus_confidence'],
        'focus_proba':         session_state['focus_proba'],
        'emotion_label':       session_state['emotion_label'],
        'emotion_confidence':  session_state['emotion_confidence'],
        'drowsiness_level':    session_state['drowsiness_level'],
        'drowsiness_severity': session_state['drowsiness_severity'],
        'attention_score':     session_state['attention_score'],
        'attention_label':     session_state['attention_label'],
        'prod_score':          session_state['prod_score'],
        'distraction_count':   session_state['distraction_count'],
        'yawn_count':          session_state['yawn_count'],
        'alert':               session_state['alert'],
        'alert_type':          session_state['alert_type'],
        'alert_icon':          session_state['alert_icon'],
        'beep':                beep,
        'beep_type':           beep_type,
        'break_suggestion':    session_state['break_suggestion'],
        'admin_alert':         admin_alert,
        'quiz_due':            quiz_due,
        'elapsed':             f"{mm:02d}:{ss:02d}",
        'elapsed_secs':        elapsed,
        'score_history':       session_state['score_history'][-60:],
        'ear_history':         session_state['ear_history'][-60:],
        'focus_streak':        session_state.get('focus_streak', 0),
        'max_streak':          session_state.get('max_streak', 0),
        'goal_score':          session_state.get('goal_score', 75),
        'goal_reached':        session_state.get('goal_reached', False),
        'motivational':        session_state.get('motivational', ''),
        'quiz_streak':         session_state.get('quiz_streak', 0),
        'current_ear':         session_state.get('current_ear', 0),
        'focus_high_pct':      round(session_state['focus_stats']['High'] / tf * 100, 1),
        'focus_med_pct':       round(session_state['focus_stats']['Medium'] / tf * 100, 1),
        'focus_low_pct':       round(session_state['focus_stats']['Low'] / tf * 100, 1),
        'emotion_engaged_pct': round(session_state['emotion_stats']['Engaged'] / te * 100, 1),
        'emotion_bored_pct':   round(session_state['emotion_stats']['Bored'] / te * 100, 1),
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("  COGNITO — http://localhost:5000")
    print("  First registered user = Admin")
    print("="*50 + "\n")
    app.run(debug=True, threaded=True)