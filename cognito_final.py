import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import threading
import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── Load models ──────────────────────────────────────────────────
print("Loading COGNITO models...")
focus_model   = joblib.load('models/cognito_model.pkl')
emotion_model = joblib.load('models/emotion_model.pkl')
print("Models loaded!")

# ── Setup Face Landmarker ────────────────────────────────────────
MODEL_PATH   = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options      = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# ── Feature functions ────────────────────────────────────────────
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_EAR(lm, w, h):
    L = [(lm[i].x * w, lm[i].y * h) for i in [33, 160, 158, 133, 153, 144]]
    R = [(lm[i].x * w, lm[i].y * h) for i in [362, 385, 387, 263, 373, 380]]
    def ear(e):
        return (euclidean(e[1], e[5]) + euclidean(e[2], e[4])) / (2.0 * euclidean(e[0], e[3]))
    return (ear(L) + ear(R)) / 2.0

def get_MAR(lm, w, h):
    mouth = [(lm[i].x * w, lm[i].y * h) for i in [61, 291, 13, 14, 78, 308]]
    return (euclidean(mouth[2], mouth[3]) + euclidean(mouth[4], mouth[5])) / (2.0 * euclidean(mouth[0], mouth[1]))

def get_gaze(lm, w, h):
    gaze_x = (lm[468].x + lm[473].x) / 2.0
    gaze_y = (lm[468].y + lm[473].y) / 2.0
    return gaze_x, gaze_y

def get_head_pose(lm, w, h):
    pitch = (lm[1].y - lm[152].y)
    yaw   = (lm[263].x - lm[33].x)
    return pitch, yaw

# ── Colors ───────────────────────────────────────────────────────
FOCUS_COLORS = {
    'High':   (0, 255, 0),
    'Medium': (0, 165, 255),
    'Low':    (0, 0, 255)
}
EMOTION_COLORS = {
    'Engaged':    (0, 255, 0),
    'Bored':      (0, 0, 255),
    'Confused':   (0, 165, 255),
    'Frustrated': (0, 0, 200)
}
EMOTION_ICONS = {
    'Engaged':    '😊',
    'Bored':      '😴',
    'Confused':   '😕',
    'Frustrated': '😤'
}

# ── Quiz questions ───────────────────────────────────────────────
QUIZZES = [
    {"q": "What does EAR stand for?",
     "options": ["A) Eye Aspect Ratio", "B) Eye Area Range", "C) Ear Audio Recognition"],
     "answer": "A"},
    {"q": "Which library detects facial landmarks?",
     "options": ["A) OpenCV", "B) Scikit-learn", "C) MediaPipe"],
     "answer": "C"},
    {"q": "What is the main goal of COGNITO?",
     "options": ["A) Face Recognition", "B) Focus Monitoring", "C) Emotion Detection"],
     "answer": "B"},
    {"q": "Which ML model is used in COGNITO?",
     "options": ["A) Neural Network", "B) Random Forest", "C) SVM"],
     "answer": "B"},
]

# ── Popup functions ──────────────────────────────────────────────
def show_break_popup():
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("COGNITO - Break Time! ☕",
        "You have been unfocused for 5 minutes!\n\n"
        "Take a 5 minute break:\n"
        "  • Look away from the screen\n"
        "  • Stretch your neck and shoulders\n"
        "  • Drink some water\n\n"
        "Come back refreshed!")
    root.destroy()

def show_quiz_popup(quiz):
    root = tk.Tk()
    root.title("COGNITO - Focus Quiz!")
    root.geometry("420x320")
    root.configure(bg="#1e1e2e")
    tk.Label(root, text="Focus Check Quiz!",
             font=("Arial", 14, "bold"),
             bg="#1e1e2e", fg="#cdd6f4").pack(pady=10)
    tk.Label(root, text=quiz["q"],
             font=("Arial", 11),
             bg="#1e1e2e", fg="white",
             wraplength=380).pack(pady=10)
    result_var = tk.StringVar()
    def check(ans):
        result_var.set("Correct! 🎉" if ans == quiz["answer"] else f"Wrong! Answer: {quiz['answer']}")
        for b in buttons: b.config(state="disabled")
    buttons = []
    for opt in quiz["options"]:
        b = tk.Button(root, text=opt, font=("Arial", 10),
                      bg="#313244", fg="white",
                      command=lambda o=opt[0]: check(o), width=38)
        b.pack(pady=3)
        buttons.append(b)
    tk.Label(root, textvariable=result_var,
             font=("Arial", 11, "bold"),
             bg="#1e1e2e", fg="#a6e3a1").pack(pady=8)
    tk.Button(root, text="Close", command=root.destroy,
              bg="#f38ba8", fg="white").pack()
    root.mainloop()

def show_eye_exercise_popup():
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo("COGNITO - Eye Exercise! 👁️",
        "Yawning detected!\n\n"
        "Quick eye exercise:\n"
        "  1. Close eyes for 5 seconds\n"
        "  2. Look left, right, up, down\n"
        "  3. Roll eyes in a circle\n"
        "  4. Focus on distant object for 20 sec")
    root.destroy()

# ── Session analytics ────────────────────────────────────────────
def show_analytics(focus_stats, emotion_stats, focus_timeline, session_secs):
    mins, secs = divmod(int(session_secs), 60)
    total_f    = sum(focus_stats.values())   or 1
    total_e    = sum(emotion_stats.values()) or 1

    fig = plt.figure(figsize=(14, 8), facecolor='#1e1e2e')
    fig.suptitle(f'COGNITO Session Report  |  Duration: {mins:02d}:{secs:02d}',
                 color='white', fontsize=15, fontweight='bold')

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)

    # ── 1. Focus Pie chart ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    f_labels = [k for k, v in focus_stats.items() if v > 0]
    f_sizes  = [v for v in focus_stats.values() if v > 0]
    f_colors = ['#a6e3a1', '#fab387', '#f38ba8'][:len(f_labels)]
    ax1.pie(f_sizes, labels=f_labels, colors=f_colors,
            autopct='%1.1f%%', textprops={'color': 'white'})
    ax1.set_title('Focus Distribution', color='white', fontsize=11)
    ax1.set_facecolor('#1e1e2e')

    # ── 2. Emotion Bar chart ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    e_labels = list(emotion_stats.keys())
    e_values = [emotion_stats[k] / total_e * 100 for k in e_labels]
    e_colors = ['#a6e3a1', '#89b4fa', '#fab387', '#f38ba8']
    bars = ax2.bar(e_labels, e_values, color=e_colors)
    ax2.set_ylabel('% of session', color='white')
    ax2.set_title('Emotion Distribution', color='white', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#313244')
    for bar, val in zip(bars, e_values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', color='white', fontsize=8)

    # ── 3. Focus Timeline ────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    focus_map = {'High': 2, 'Medium': 1, 'Low': 0}
    y_vals    = [focus_map.get(f, 1) for f in focus_timeline]
    x_vals    = list(range(len(y_vals)))
    if y_vals:
        ax3.fill_between(x_vals, y_vals, alpha=0.4, color='#89b4fa')
        ax3.plot(x_vals, y_vals, color='#cdd6f4', linewidth=1)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Low', 'Med', 'High'], color='white')
    ax3.set_title('Focus Over Time', color='white', fontsize=11)
    ax3.tick_params(colors='white')
    ax3.set_facecolor('#313244')
    ax3.set_xlabel('Time (frames)', color='white', fontsize=8)

    # ── 4. Stats Summary ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    ax4.set_facecolor('#313244')

    best_focus   = max(focus_stats,   key=focus_stats.get)
    best_emotion = max(emotion_stats, key=emotion_stats.get)

    summary = (
        f"Session Duration: {mins:02d}:{secs:02d}    |    "
        f"Dominant Focus: {best_focus}    |    "
        f"Dominant Emotion: {best_emotion}\n\n"
        f"High Focus: {focus_stats['High']/total_f*100:.1f}%    "
        f"Medium Focus: {focus_stats['Medium']/total_f*100:.1f}%    "
        f"Low Focus: {focus_stats['Low']/total_f*100:.1f}%\n\n"
        f"Engaged: {emotion_stats['Engaged']/total_e*100:.1f}%    "
        f"Bored: {emotion_stats['Bored']/total_e*100:.1f}%    "
        f"Confused: {emotion_stats['Confused']/total_e*100:.1f}%    "
        f"Frustrated: {emotion_stats['Frustrated']/total_e*100:.1f}%"
    )
    ax4.text(0.5, 0.5, summary, transform=ax4.transAxes,
             fontsize=11, color='white', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#313244', alpha=0.8))
    ax4.set_title('Session Summary', color='white', fontsize=11)

    plt.savefig('session_report.png', dpi=120,
                bbox_inches='tight', facecolor='#1e1e2e')
    print("Report saved to session_report.png")
    plt.show()

# ── State tracking ───────────────────────────────────────────────
focus_stats   = defaultdict(int, {'High': 0, 'Medium': 0, 'Low': 0})
emotion_stats = defaultdict(int, {'Engaged': 0, 'Bored': 0,
                                   'Confused': 0, 'Frustrated': 0})
focus_timeline  = []
low_start       = None
last_break      = 0
last_quiz       = 0
last_eye        = 0
quiz_index      = 0
popup_active    = False

def trigger_popup(func, *args):
    global popup_active
    if not popup_active:
        popup_active = True
        def run():
            global popup_active
            func(*args)
            popup_active = False
        threading.Thread(target=run, daemon=True).start()

# ── Webcam loop ──────────────────────────────────────────────────
print("Starting COGNITO... Press Q to quit and see analytics")
cap           = cv2.VideoCapture(0)
focus_label   = "Detecting..."
emotion_label = "Detecting..."
focus_color   = (255, 255, 255)
emotion_color = (255, 255, 255)
frame_count   = 0
session_start = time.time()
lm            = None

# Mini graph history
focus_history = []
MAX_HISTORY   = 150

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_count % 5 == 0:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            try:
                ear            = get_EAR(lm, w, h)
                mar            = get_MAR(lm, w, h)
                gaze_x, gaze_y = get_gaze(lm, w, h)
                pitch, yaw     = get_head_pose(lm, w, h)
                feats          = np.array([[ear, mar, gaze_x,
                                            gaze_y, pitch, yaw]])

                focus_label   = focus_model.predict(feats)[0]
                emotion_label = emotion_model.predict(feats)[0]
                focus_color   = FOCUS_COLORS.get(focus_label,   (255,255,255))
                emotion_color = EMOTION_COLORS.get(emotion_label,(255,255,255))

                focus_stats[focus_label]     += 1
                emotion_stats[emotion_label] += 1
                focus_timeline.append(focus_label)

                fv = {'High': 2, 'Medium': 1, 'Low': 0}.get(focus_label, 1)
                focus_history.append(fv)
                if len(focus_history) > MAX_HISTORY:
                    focus_history.pop(0)

                now = time.time()

                # Break trigger
                if focus_label == 'Low':
                    if low_start is None: low_start = now
                    elif (now-low_start)>300 and (now-last_break)>300:
                        last_break = now
                        trigger_popup(show_break_popup)
                else:
                    low_start = None

                # Quiz trigger
                if (now-last_quiz)>600 and (now-session_start)>60:
                    last_quiz  = now
                    q          = QUIZZES[quiz_index % len(QUIZZES)]
                    quiz_index += 1
                    trigger_popup(show_quiz_popup, q)

                # Yawn trigger
                if mar > 0.6 and (now-last_eye) > 120:
                    last_eye = now
                    trigger_popup(show_eye_exercise_popup)

            except:
                pass
        else:
            focus_label   = "No Face"
            emotion_label = ""
            lm            = None

    # ── Draw UI ──────────────────────────────────────────────────
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 75), (20, 20, 30), -1)
    cv2.putText(frame, "COGNITO",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (150, 150, 255), 2)
    cv2.putText(frame, f"Focus: {focus_label}",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                focus_color, 2)
    cv2.putText(frame, f"Emotion: {emotion_label}",
                (w//2, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                emotion_color, 2)

    # Timer
    elapsed    = int(time.time() - session_start)
    mins, secs = divmod(elapsed, 60)
    cv2.putText(frame, f"{mins:02d}:{secs:02d}",
                (w-100, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (200,200,200), 2)

    # Mini graph
    gw, gh = 200, 60
    gx, gy = w-gw-10, h-gh-10
    cv2.rectangle(frame, (gx, gy), (gx+gw, gy+gh), (30,30,30), -1)
    if len(focus_history) > 1:
        for i in range(1, len(focus_history)):
            x1 = gx + int((i-1)*gw/MAX_HISTORY)
            x2 = gx + int(i    *gw/MAX_HISTORY)
            y1 = gy + gh - int(focus_history[i-1]*gh/2)
            y2 = gy + gh - int(focus_history[i]  *gh/2)
            col = [(0,0,255),(0,165,255),(0,255,0)][focus_history[i]]
            cv2.line(frame, (x1,y1), (x2,y2), col, 2)
    cv2.putText(frame, "Focus Graph",
                (gx, gy-5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (180,180,180), 1)

    # Yawn warning
    try:
        if lm and get_MAR(lm, w, h) > 0.6:
            cv2.putText(frame, "Yawning detected!",
                        (10, h-15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,165,255), 2)
    except:
        pass

    cv2.imshow('COGNITO - Focus & Emotion Monitor', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()

# ── Show analytics ───────────────────────────────────────────────
print("\nGenerating session analytics...")
show_analytics(focus_stats, emotion_stats,
               focus_timeline, time.time()-session_start)