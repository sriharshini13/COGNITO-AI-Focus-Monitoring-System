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
import winsound
import urllib.request
import os

# ── Import all modules ───────────────────────────────────────────
from modules.distraction_counter    import DistractionCounter
from modules.productivity_score     import ProductivityScore
from modules.pdf_export             import generate_pdf
from modules.session_logger         import SessionLogger
from modules.multi_person           import MultiPersonTracker
from modules.face_recognition_module import FaceRecognizer
from modules.dashboard              import show_dashboard

# ── Download pose model if needed ───────────────────────────────
FACE_MODEL = 'face_landmarker.task'
POSE_MODEL = 'pose_landmarker.task'
if not os.path.exists(POSE_MODEL):
    print("Downloading pose model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/1/"
        "pose_landmarker_lite.task",
        POSE_MODEL)
    print("Done!")

# ── Load ML models ───────────────────────────────────────────────
print("Loading COGNITO models...")
focus_model   = joblib.load('models/cognito_model.pkl')
emotion_model = joblib.load('models/emotion_model.pkl')
print("Models loaded!")

# ── Setup MediaPipe ──────────────────────────────────────────────
face_detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=FACE_MODEL),
        num_faces=4,
        min_face_detection_confidence=0.5))

pose_detector = vision.PoseLandmarker.create_from_options(
    vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=POSE_MODEL),
        output_segmentation_masks=False,
        num_poses=1))

# ── Feature functions ────────────────────────────────────────────
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_EAR(lm, w, h):
    L = [(lm[i].x*w, lm[i].y*h) for i in [33,160,158,133,153,144]]
    R = [(lm[i].x*w, lm[i].y*h) for i in [362,385,387,263,373,380]]
    def ear(e):
        return (euclidean(e[1],e[5])+euclidean(e[2],e[4])) / \
               (2.0*euclidean(e[0],e[3]))
    return (ear(L)+ear(R))/2.0

def get_MAR(lm, w, h):
    m = [(lm[i].x*w, lm[i].y*h) for i in [61,291,13,14,78,308]]
    return (euclidean(m[2],m[3])+euclidean(m[4],m[5])) / \
           (2.0*euclidean(m[0],m[1]))

def get_gaze(lm, w, h):
    return (lm[468].x+lm[473].x)/2.0, \
           (lm[468].y+lm[473].y)/2.0

def get_head_pose(lm, w, h):
    return (lm[1].y-lm[152].y), (lm[263].x-lm[33].x)

def check_posture(pose_lm, w, h):
    try:
        ls = pose_lm[11]; rs = pose_lm[12]; nose = pose_lm[0]
        if abs(ls.y-rs.y)>0.05 or \
           (nose.y-(ls.y+rs.y)/2.0)>0.1:
            return "Sit Straight!", False
        return "Good Posture", True
    except:
        return "Posture N/A", None

def calc_attention(ear, mar, gaze_x, gaze_y, pitch, yaw):
    score = 100.0
    if ear < 0.15:    score -= 40
    elif ear < 0.20:  score -= 20
    elif ear < 0.25:  score -= 10
    if mar > 0.6:     score -= 25
    elif mar > 0.4:   score -= 10
    gd = abs(gaze_x-0.5)+abs(gaze_y-0.5)
    if gd > 0.3:      score -= 20
    elif gd > 0.15:   score -= 10
    if abs(yaw) < 0.1:  score -= 15
    if abs(pitch)>0.15: score -= 10
    return max(0, min(100, int(score)))

# ── Colors ───────────────────────────────────────────────────────
FOCUS_COLORS = {
    'High':   (0,255,0),
    'Medium': (0,165,255),
    'Low':    (0,0,255)
}
EMOTION_COLORS = {
    'Engaged':    (0,255,0),
    'Bored':      (0,0,255),
    'Confused':   (0,165,255),
    'Frustrated': (0,0,200)
}
PERSON_COLORS = [
    (0,255,0),(0,165,255),
    (255,0,255),(0,255,255)
]

# ── Quiz questions ───────────────────────────────────────────────
QUIZZES = [
    {"q": "What does EAR stand for?",
     "options": ["A) Eye Aspect Ratio",
                 "B) Eye Area Range",
                 "C) Ear Audio Recognition"],
     "answer": "A"},
    {"q": "Which library detects facial landmarks?",
     "options": ["A) OpenCV","B) Scikit-learn","C) MediaPipe"],
     "answer": "C"},
    {"q": "What is the main goal of COGNITO?",
     "options": ["A) Face Recognition",
                 "B) Focus Monitoring",
                 "C) Emotion Detection"],
     "answer": "B"},
    {"q": "Which ML model is used in COGNITO?",
     "options": ["A) Neural Network",
                 "B) Random Forest","C) SVM"],
     "answer": "B"},
]

# ── Popups ───────────────────────────────────────────────────────
def show_break_popup():
    root = tk.Tk(); root.withdraw()
    messagebox.showinfo("COGNITO - Break Time! ☕",
        "Low focus for 5 minutes!\n\n"
        "Take a break:\n"
        "  • Look away from screen\n"
        "  • Stretch your neck\n"
        "  • Drink some water")
    root.destroy()

def show_quiz_popup(quiz):
    root = tk.Tk()
    root.title("COGNITO - Focus Quiz!")
    root.geometry("420x320")
    root.configure(bg="#1e1e2e")
    tk.Label(root, text="Focus Check Quiz!",
             font=("Arial",14,"bold"),
             bg="#1e1e2e", fg="#cdd6f4").pack(pady=10)
    tk.Label(root, text=quiz["q"],
             font=("Arial",11), bg="#1e1e2e",
             fg="white", wraplength=380).pack(pady=10)
    result_var = tk.StringVar()
    def check(ans):
        result_var.set("Correct! 🎉" if ans==quiz["answer"]
                       else f"Wrong! Answer: {quiz['answer']}")
        for b in buttons: b.config(state="disabled")
    buttons = []
    for opt in quiz["options"]:
        b = tk.Button(root, text=opt, font=("Arial",10),
                      bg="#313244", fg="white",
                      command=lambda o=opt[0]: check(o),
                      width=38)
        b.pack(pady=3); buttons.append(b)
    tk.Label(root, textvariable=result_var,
             font=("Arial",11,"bold"),
             bg="#1e1e2e", fg="#a6e3a1").pack(pady=8)
    tk.Button(root, text="Close", command=root.destroy,
              bg="#f38ba8", fg="white").pack()
    root.mainloop()

def show_eye_popup():
    root = tk.Tk(); root.withdraw()
    messagebox.showinfo("COGNITO - Eye Exercise! 👁️",
        "Yawning detected!\n\n"
        "Quick eye exercise:\n"
        "  1. Close eyes 5 seconds\n"
        "  2. Look left, right, up, down\n"
        "  3. Focus on distant object 20 sec")
    root.destroy()

popup_active = False
def trigger_popup(func, *args):
    global popup_active
    if not popup_active:
        popup_active = True
        def run():
            global popup_active
            func(*args); popup_active = False
        threading.Thread(target=run, daemon=True).start()

def play_alert():
    threading.Thread(
        target=lambda: winsound.Beep(1000,500),
        daemon=True).start()

# ════════════════════════════════════════════════════════════════
#                        STARTUP MENU
# ════════════════════════════════════════════════════════════════
print("\n" + "="*50)
print("       COGNITO v3 — Focus Analysis System")
print("="*50)
print("\nSelect Mode:")
print("  1. Single Person Mode")
print("  2. Classroom Mode (Multi-Person)")
mode_choice = input("\nEnter choice (1 or 2): ").strip()
CLASSROOM_MODE = mode_choice == '2'

# ── Face recognition (single mode only) ─────────────────────────
recognizer    = FaceRecognizer()
current_user  = None
user_greeted  = False
greeting_text = ""
greeting_timer = 0

if not CLASSROOM_MODE:
    print("\nFace Recognition:")
    print("  1. I'm a registered user")
    print("  2. Register me now (webcam)")
    print("  3. Register from photo")
    print("  4. Skip face recognition")
    fr_choice = input("\nEnter choice (1-4): ").strip()

    if fr_choice == '1':
        print("\nLook at the camera for recognition...")
    elif fr_choice == '2':
        name = input("Enter your name: ")
        recognizer.register_from_webcam(name)
    elif fr_choice == '3':
        name  = input("Enter your name: ")
        photo = input("Enter photo path: ")
        recognizer.register_from_photo(name, photo)

# ── Initialize all modules ───────────────────────────────────────
distraction_tracker = DistractionCounter(cooldown_secs=5)
productivity_tracker = ProductivityScore()
session_logger       = SessionLogger()
multi_tracker        = MultiPersonTracker()

# ── Session state ────────────────────────────────────────────────
from collections import defaultdict
focus_stats   = defaultdict(int,
    {'High':0,'Medium':0,'Low':0})
emotion_stats = defaultdict(int,
    {'Engaged':0,'Bored':0,
     'Confused':0,'Frustrated':0})
focus_timeline  = []
score_timeline  = []
focus_history   = []
MAX_HISTORY     = 150

focus_label     = "Detecting..."
emotion_label   = "Detecting..."
posture_label   = "Detecting..."
posture_color   = (255,255,255)
focus_color     = (255,255,255)
emotion_color   = (255,255,255)
attention_score = 0
prod_score      = 0

low_start  = None
last_break = last_quiz = last_eye = last_beep = 0
quiz_index = 0
lm         = None
frame_count  = 0
session_start = time.time()

# ════════════════════════════════════════════════════════════════
#                        WEBCAM LOOP
# ════════════════════════════════════════════════════════════════
mode_label = "Classroom Mode" if CLASSROOM_MODE \
             else "Single Mode"
print(f"\nStarting COGNITO v3 [{mode_label}]...")
print("Press Q to quit and generate report")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w   = frame.shape[:2]
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=rgb)

    if frame_count % 5 == 0:
        face_result = face_detector.detect(mp_img)
        now         = time.time()

        # ── CLASSROOM MODE ───────────────────────────────────────
        if CLASSROOM_MODE:
            if face_result.face_landmarks:
                people = multi_tracker.update(
                    face_result.face_landmarks,
                    focus_model, emotion_model, w, h)

                for i, person in enumerate(people):
                    color    = PERSON_COLORS[i%len(PERSON_COLORS)]
                    x1,y1,x2,y2 = person['bbox']
                    cv2.rectangle(frame,
                                  (x1,y1),(x2,y2), color, 2)
                    label = (f"{person['id']} | "
                             f"{person['focus']} | "
                             f"{person['emotion']}")
                    cv2.rectangle(frame,
                                  (x1,y1-28),
                                  (x1+len(label)*9,y1),
                                  color, -1)
                    cv2.putText(frame, label,
                                (x1+2,y1-8),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.52, (0,0,0), 2)

        # ── SINGLE PERSON MODE ───────────────────────────────────
        else:
            if face_result.face_landmarks:
                lm = face_result.face_landmarks[0]
                try:
                    ear            = get_EAR(lm, w, h)
                    mar            = get_MAR(lm, w, h)
                    gaze_x, gaze_y = get_gaze(lm, w, h)
                    pitch, yaw     = get_head_pose(lm, w, h)
                    feats          = np.array([[ear, mar,
                                               gaze_x, gaze_y,
                                               pitch, yaw]])

                    focus_label   = focus_model.predict(feats)[0]
                    emotion_label = emotion_model.predict(feats)[0]
                    attention_score = calc_attention(
                        ear, mar, gaze_x, gaze_y, pitch, yaw)
                    posture_good   = None

                    focus_color   = FOCUS_COLORS.get(
                        focus_label,   (255,255,255))
                    emotion_color = EMOTION_COLORS.get(
                        emotion_label, (255,255,255))

                    # ── Face recognition ─────────────────────────
                    if not user_greeted and \
                            frame_count % 30 == 0:
                        name, conf = recognizer.recognize(frame)
                        if name:
                            current_user  = name
                            user_greeted  = True
                            info          = recognizer\
                                .get_user_info(name)
                            greeting_text = info['greeting']
                            greeting_timer = now + 5
                            print(f"\n{greeting_text}")
                            if info['history']:
                                last = info['history'][-1]
                                print(f"Last session: "
                                      f"{last['date']} | "
                                      f"Score: "
                                      f"{last['avg_productivity']}")

                    # ── Module updates ───────────────────────────
                    new_dist = distraction_tracker\
                        .update(focus_label)
                    prod_score = productivity_tracker.calculate(
                        focus_label, emotion_label,
                        attention_score,
                        posture_good is True)

                    focus_stats[focus_label]     += 1
                    emotion_stats[emotion_label] += 1
                    focus_timeline.append(focus_label)
                    score_timeline.append(prod_score)

                    fv = {'High':2,'Medium':1,'Low':0}\
                        .get(focus_label,1)
                    focus_history.append(fv)
                    if len(focus_history) > MAX_HISTORY:
                        focus_history.pop(0)

                    # ── Triggers ─────────────────────────────────
                    if focus_label == 'Low':
                        if low_start is None:
                            low_start = now
                        if (now-low_start)>10 and \
                                (now-last_beep)>15:
                            last_beep = now
                            play_alert()
                        if (now-low_start)>300 and \
                                (now-last_break)>300:
                            last_break = now
                            trigger_popup(show_break_popup)
                    else:
                        low_start = None

                    if (now-last_quiz)>600 and \
                            (now-session_start)>60:
                        last_quiz  = now
                        q = QUIZZES[quiz_index%len(QUIZZES)]
                        quiz_index += 1
                        trigger_popup(show_quiz_popup, q)

                    if mar>0.6 and (now-last_eye)>120:
                        last_eye = now
                        trigger_popup(show_eye_popup)

                except:
                    pass

            else:
                focus_label = "No Face"
                lm = None

        # ── Pose detection ───────────────────────────────────────
        if not CLASSROOM_MODE:
            pose_result = pose_detector.detect(mp_img)
            if pose_result.pose_landmarks:
                posture_label, good = check_posture(
                    pose_result.pose_landmarks[0], w, h)
                posture_color = (0,255,0) if good \
                                else (0,0,255)
            else:
                posture_label = "Posture N/A"
                posture_color = (180,180,180)

    # ════════════════════════════════════════════════════════════
    #                        DRAW UI
    # ════════════════════════════════════════════════════════════
    # Top bar
    cv2.rectangle(frame, (0,0), (w,85), (20,20,30), -1)
    cv2.putText(frame, f"COGNITO v3  [{mode_label}]",
                (10,22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (150,150,255), 2)

    if CLASSROOM_MODE:
        n = len(multi_tracker.people)
        cv2.putText(frame,
                    f"People detected: {n}",
                    (10,55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,255,0), 2)
    else:
        cv2.putText(frame,
                    f"Focus: {focus_label}",
                    (10,55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, focus_color, 2)
        cv2.putText(frame,
                    f"Emotion: {emotion_label}",
                    (w//2-50,55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, emotion_color, 2)

    # Timer
    elapsed    = int(time.time()-session_start)
    mins, secs = divmod(elapsed, 60)
    cv2.putText(frame, f"{mins:02d}:{secs:02d}",
                (w-100,45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (200,200,200), 2)

    if not CLASSROOM_MODE:
        # Attention bar
        bx,by,bw2,bh2 = 10,95,200,16
        cv2.rectangle(frame,
                      (bx,by),(bx+bw2,by+bh2),
                      (50,50,50), -1)
        filled    = int(attention_score/100*bw2)
        bar_color = (0,255,0) if attention_score>70 \
                    else (0,165,255) if attention_score>40 \
                    else (0,0,255)
        cv2.rectangle(frame,
                      (bx,by),(bx+filled,by+bh2),
                      bar_color, -1)
        cv2.putText(frame,
                    f"Attention: {attention_score}/100",
                    (bx,by-4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (200,200,200), 1)

        # Productivity score
        cv2.putText(frame,
                    f"Productivity: {prod_score}/100",
                    (220,107),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200,200,200), 1)

        # Distraction count
        dc = distraction_tracker.get_count()
        cv2.putText(frame,
                    f"Distractions: {dc}",
                    (10,130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200,200,200), 1)

        # Posture
        cv2.putText(frame, posture_label,
                    (220,130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, posture_color, 2)

        # User name
        if current_user:
            cv2.putText(frame,
                        f"User: {current_user}",
                        (w-200,22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (150,255,150), 1)

        # Greeting banner
        if greeting_text and \
                time.time() < greeting_timer:
            cv2.rectangle(frame,
                          (0,h-60),(w,h),
                          (20,20,30), -1)
            cv2.putText(frame, greeting_text,
                        (10,h-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0,255,150), 2)

        # Mini focus graph
        gw2,gh2 = 200,55
        gx,gy   = w-gw2-10, h-gh2-10
        cv2.rectangle(frame,
                      (gx,gy),(gx+gw2,gy+gh2),
                      (30,30,30), -1)
        if len(focus_history) > 1:
            for i in range(1, len(focus_history)):
                x1 = gx+int((i-1)*gw2/MAX_HISTORY)
                x2 = gx+int(i    *gw2/MAX_HISTORY)
                y1 = gy+gh2-int(focus_history[i-1]*gh2/2)
                y2 = gy+gh2-int(focus_history[i]  *gh2/2)
                col = [(0,0,255),(0,165,255),
                       (0,255,0)][focus_history[i]]
                cv2.line(frame,(x1,y1),(x2,y2),col,2)
        cv2.putText(frame, "Focus Graph",
                    (gx,gy-5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (180,180,180), 1)

        # Yawn warning
        try:
            if lm and get_MAR(lm,w,h) > 0.6:
                cv2.putText(frame,
                            "Yawning detected!",
                            (10,h-70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0,165,255), 2)
        except:
            pass

    cv2.imshow('COGNITO v3', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
face_detector.close()
pose_detector.close()
cv2.destroyAllWindows()

# ════════════════════════════════════════════════════════════════
#                     SESSION END & REPORTS
# ════════════════════════════════════════════════════════════════
session_secs = time.time() - session_start
mins, secs   = divmod(int(session_secs), 60)

print("\n" + "="*50)
print("         COGNITO v3 SESSION ENDED")
print("="*50)
print(f"Duration    : {mins:02d}:{secs:02d}")

if not CLASSROOM_MODE:
    prod_summary  = productivity_tracker.get_summary()
    dist_summary  = distraction_tracker.get_summary(session_secs)

    print(f"Productivity: {prod_summary['average_score']}/100 "
          f"— Grade {prod_summary['grade']}")
    print(f"Distractions: {dist_summary['total_distractions']}")

    # Update face recognition session count
    if current_user:
        recognizer.update_session_count(current_user)

    # Save session to history
    session_data = session_logger.save_session(
        dict(focus_stats),
        dict(emotion_stats),
        prod_summary,
        dist_summary,
        session_secs,
    )

    # Generate PDF report
    print("\nGenerating PDF report...")
    pdf_path = generate_pdf(
        dict(focus_stats),
        dict(emotion_stats),
        prod_summary,
        dist_summary,
        session_secs,
        output_path=f'session_logs/report_{int(time.time())}.pdf'
    )
    print(f"PDF saved: {pdf_path}")

    # Show analytics dashboard
    print("\nShowing session dashboard...")
    show_dashboard(mode='week')

else:
    # Classroom summary
    print("\n=== Classroom Summary ===")
    summary = multi_tracker.get_classroom_summary()
    for pid, data in summary.items():
        print(f"{pid}: Focus={data['dominant_focus']} | "
              f"Emotion={data['dominant_emotion']} | "
              f"High={data['high_focus_pct']}% | "
              f"Engaged={data['engaged_pct']}%")