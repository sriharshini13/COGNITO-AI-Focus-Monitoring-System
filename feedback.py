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

# ── Load model ───────────────────────────────────────────────────
print("Loading COGNITO model...")
model = joblib.load('models/cognito_model.pkl')
print("Model loaded!")

# ── Setup Face Landmarker ────────────────────────────────────────
MODEL_PATH = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
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
COLORS = {
    'High':   (0, 255, 0),
    'Medium': (0, 165, 255),
    'Low':    (0, 0, 255)
}

# ── Quiz questions ───────────────────────────────────────────────
QUIZZES = [
    {"q": "What does EAR stand for in COGNITO?",
     "options": ["A) Eye Aspect Ratio", "B) Eye Area Range", "C) Ear Audio Recognition"],
     "answer": "A"},
    {"q": "Which library detects facial landmarks?",
     "options": ["A) OpenCV", "B) MediaPipe", "C) Scikit-learn"],
     "answer": "B"},
    {"q": "What is the main goal of COGNITO?",
     "options": ["A) Face Recognition", "B) Focus Monitoring", "C) Emotion Detection"],
     "answer": "B"},
    {"q": "Which ML model is used in COGNITO?",
     "options": ["A) Neural Network", "B) SVM", "C) Random Forest"],
     "answer": "C"},
]

# ── Popup functions ──────────────────────────────────────────────
def show_break_popup():
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "COGNITO - Break Time! ☕",
        "You have been unfocused for 5 minutes!\n\n"
        "Take a 5 minute break:\n"
        "  • Look away from screen\n"
        "  • Stretch your neck and shoulders\n"
        "  • Drink some water\n\n"
        "Come back refreshed!"
    )
    root.destroy()

def show_quiz_popup(quiz):
    root = tk.Tk()
    root.title("COGNITO - Focus Quiz!")
    root.geometry("400x300")
    root.configure(bg="#1e1e2e")

    tk.Label(root, text="Focus Check Quiz!", font=("Arial", 14, "bold"),
             bg="#1e1e2e", fg="#cdd6f4").pack(pady=10)

    tk.Label(root, text=quiz["q"], font=("Arial", 11),
             bg="#1e1e2e", fg="white", wraplength=360).pack(pady=10)

    result_var = tk.StringVar()

    def check(ans):
        if ans == quiz["answer"]:
            result_var.set("Correct! Great job! 🎉")
        else:
            result_var.set(f"Wrong! Answer was {quiz['answer']}")
        for btn in buttons:
            btn.config(state="disabled")

    buttons = []
    for opt in quiz["options"]:
        btn = tk.Button(root, text=opt, font=("Arial", 10),
                        bg="#313244", fg="white",
                        command=lambda o=opt[0]: check(o),
                        width=35)
        btn.pack(pady=3)
        buttons.append(btn)

    tk.Label(root, textvariable=result_var, font=("Arial", 11, "bold"),
             bg="#1e1e2e", fg="#a6e3a1").pack(pady=10)

    tk.Button(root, text="Close", command=root.destroy,
              bg="#f38ba8", fg="white").pack()

    root.mainloop()

def show_eye_exercise_popup():
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "COGNITO - Eye Exercise! 👁️",
        "Yawning detected! Time for an eye exercise:\n\n"
        "  1. Close your eyes for 5 seconds\n"
        "  2. Look left, right, up, down slowly\n"
        "  3. Roll your eyes in a circle\n"
        "  4. Focus on a distant object for 20 seconds\n\n"
        "This reduces eye strain!"
    )
    root.destroy()

# ── State tracking ───────────────────────────────────────────────
low_start         = None
last_break_popup  = 0
last_quiz_popup   = 0
last_eye_popup    = 0
quiz_index        = 0
popup_active      = False

def trigger_popup(func, *args):
    global popup_active
    if not popup_active:
        popup_active = True
        def run():
            global popup_active
            func(*args)
            popup_active = False
        threading.Thread(target=run, daemon=True).start()

# ── Focus history for graph ──────────────────────────────────────
focus_history = []
MAX_HISTORY   = 100

# ── Webcam loop ──────────────────────────────────────────────────
print("Starting COGNITO with Feedback System...")
print("Press Q to quit and see session summary")

cap           = cv2.VideoCapture(0)
focus_label   = "Detecting..."
focus_color   = (255, 255, 255)
frame_count   = 0
session_start = time.time()
lm            = None

# Session stats
stats = {'High': 0, 'Medium': 0, 'Low': 0}

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

                features    = np.array([[ear, mar, gaze_x, gaze_y, pitch, yaw]])
                focus_label = model.predict(features)[0]
                focus_color = COLORS.get(focus_label, (255, 255, 255))

                # Track stats
                if focus_label in stats:
                    stats[focus_label] += 1

                # Focus history for mini graph
                focus_val = {'High': 2, 'Medium': 1, 'Low': 0}.get(focus_label, 1)
                focus_history.append(focus_val)
                if len(focus_history) > MAX_HISTORY:
                    focus_history.pop(0)

                now = time.time()

                # ── Low focus tracking ───────────────────────
                if focus_label == 'Low':
                    if low_start is None:
                        low_start = now
                    elif (now - low_start) > 300 and (now - last_break_popup) > 300:
                        last_break_popup = now
                        trigger_popup(show_break_popup)
                else:
                    low_start = None

                # ── Quiz trigger (every 10 mins) ─────────────
                if (now - last_quiz_popup) > 600 and (now - session_start) > 60:
                    last_quiz_popup = now
                    q = QUIZZES[quiz_index % len(QUIZZES)]
                    quiz_index += 1
                    trigger_popup(show_quiz_popup, q)

                # ── Yawn / eye exercise trigger ───────────────
                if mar > 0.6 and (now - last_eye_popup) > 120:
                    last_eye_popup = now
                    trigger_popup(show_eye_exercise_popup)

            except:
                pass
        else:
            focus_label = "No Face"
            focus_color = (255, 255, 255)
            lm = None

    # ── Draw UI ──────────────────────────────────────────────────
    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 65), (20, 20, 30), -1)
    cv2.putText(frame, f"COGNITO", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)
    cv2.putText(frame, f"Focus: {focus_label}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, focus_color, 2)

    # Session timer
    elapsed    = int(time.time() - session_start)
    mins, secs = divmod(elapsed, 60)
    cv2.putText(frame, f"{mins:02d}:{secs:02d}", (w - 100, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)

    # ── Mini focus graph ─────────────────────────────────────────
    graph_h = 60
    graph_w = 200
    graph_x = w - graph_w - 10
    graph_y = h - graph_h - 10
    cv2.rectangle(frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h),
                  (30, 30, 30), -1)

    if len(focus_history) > 1:
        for i in range(1, len(focus_history)):
            x1 = graph_x + int((i - 1) * graph_w / MAX_HISTORY)
            x2 = graph_x + int(i * graph_w / MAX_HISTORY)
            y1 = graph_y + graph_h - int(focus_history[i - 1] * graph_h / 2)
            y2 = graph_y + graph_h - int(focus_history[i] * graph_h / 2)
            col = [(0,0,255),(0,165,255),(0,255,0)][focus_history[i]]
            cv2.line(frame, (x1, y1), (x2, y2), col, 2)

    cv2.putText(frame, "Focus Graph", (graph_x, graph_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    # ── Yawn warning ─────────────────────────────────────────────
    try:
        if lm and get_MAR(lm, w, h) > 0.6:
            cv2.putText(frame, "Yawning! Eye exercise recommended",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 165, 255), 2)
    except:
        pass

    cv2.imshow('COGNITO - Focus Monitor', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()

# ── Session Summary ──────────────────────────────────────────────
total  = sum(stats.values()) or 1
elapsed = int(time.time() - session_start)
mins, secs = divmod(elapsed, 60)

print("\n" + "="*40)
print("       COGNITO SESSION SUMMARY")
print("="*40)
print(f"Total session time : {mins:02d}:{secs:02d}")
print(f"High Focus         : {stats['High']/total*100:.1f}%")
print(f"Medium Focus       : {stats['Medium']/total*100:.1f}%")
print(f"Low Focus          : {stats['Low']/total*100:.1f}%")
print("="*40)