import os
import json
import cv2
import numpy as np
from datetime import datetime

FACES_DIR    = 'registered_faces'
FACES_DB     = 'registered_faces/users.json'

os.makedirs(FACES_DIR, exist_ok=True)

class FaceRecognizer:
    def __init__(self):
        self.users        = self._load_users()
        self.current_user = None
        self.greeted      = False
        self._deepface    = None

    def _load_deepface(self):
        """Lazy load DeepFace only when needed"""
        if self._deepface is None:
            from deepface import DeepFace
            self._deepface = DeepFace
        return self._deepface

    def _load_users(self):
        if os.path.exists(FACES_DB):
            with open(FACES_DB, 'r') as f:
                return json.load(f)
        return {}

    def _save_users(self):
        with open(FACES_DB, 'w') as f:
            json.dump(self.users, f, indent=2)

    # ── Register via webcam snapshot ────────────────────────────
    def register_from_webcam(self, name):
        """
        Opens webcam, takes a snapshot and registers the face.
        Returns True if successful.
        """
        print(f"Registering {name} via webcam...")
        print("Press SPACE to capture, Q to cancel")

        cap = cv2.VideoCapture(0)
        captured = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            cv2.putText(display,
                        f"Registering: {name}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0,255,0), 2)
            cv2.putText(display,
                        "SPACE = Capture  |  Q = Cancel",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200,200,200), 1)

            # Draw face guide oval
            h, w = display.shape[:2]
            cv2.ellipse(display,
                        (w//2, h//2),
                        (120, 160), 0, 0, 360,
                        (0,255,0), 2)
            cv2.putText(display,
                        "Align face inside oval",
                        (w//2-120, h//2+180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 1)

            cv2.imshow('COGNITO - Register Face', display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # Save snapshot
                path = os.path.join(
                    FACES_DIR, f"{name.lower().replace(' ','_')}.jpg")
                cv2.imwrite(path, frame)
                self.users[name] = {
                    'photo_path':   path,
                    'registered':   datetime.now().strftime(
                        '%Y-%m-%d %H:%M'),
                    'total_sessions': 0
                }
                self._save_users()
                print(f"✅ {name} registered successfully!")
                captured = True
                break

            elif key == ord('q'):
                print("Registration cancelled.")
                break

        cap.release()
        cv2.destroyAllWindows()
        return captured

    # ── Register via photo file ──────────────────────────────────
    def register_from_photo(self, name, photo_path):
        """
        Register a user from an existing photo file.
        Returns True if successful.
        """
        if not os.path.exists(photo_path):
            print(f"Photo not found: {photo_path}")
            return False

        # Copy photo to registered_faces folder
        img  = cv2.imread(photo_path)
        dest = os.path.join(
            FACES_DIR,
            f"{name.lower().replace(' ','_')}.jpg")
        cv2.imwrite(dest, img)

        self.users[name] = {
            'photo_path':     dest,
            'registered':     datetime.now().strftime(
                '%Y-%m-%d %H:%M'),
            'total_sessions': 0
        }
        self._save_users()
        print(f"✅ {name} registered from photo!")
        return True

    # ── Recognize face from frame ────────────────────────────────
    def recognize(self, frame):
        """
        Try to recognize a face in the frame.
        Returns (name, confidence) or (None, 0)
        """
        if not self.users:
            return None, 0

        DeepFace = self._load_deepface()

        try:
            # Save temp frame
            tmp = 'registered_faces/_tmp_frame.jpg'
            cv2.imwrite(tmp, frame)

            results = DeepFace.find(
                img_path      = tmp,
                db_path       = FACES_DIR,
                model_name    = 'VGG-Face',
                enforce_detection = False,
                silent        = True
            )

            if results and len(results) > 0:
                df = results[0]
                if len(df) > 0:
                    # Get best match
                    best      = df.iloc[0]
                    path      = best['identity']
                    distance  = best.get('distance', 1.0)
                    threshold = 0.4

                    if distance < threshold:
                        # Extract name from filename
                        filename = os.path.basename(path)
                        name     = filename.replace('.jpg','')\
                                           .replace('_',' ')\
                                           .title()
                        # Filter out temp file
                        if '_tmp_frame' not in name:
                            confidence = round(
                                (1 - distance) * 100, 1)
                            return name, confidence

        except Exception as e:
            pass

        return None, 0

    # ── Get user greeting + history ──────────────────────────────
    def get_user_info(self, name):
        """
        Returns greeting and session history for recognized user.
        """
        if name not in self.users:
            return None

        user = self.users[name]
        sessions = user.get('total_sessions', 0)

        if sessions == 0:
            greeting = f"Welcome to COGNITO, {name}! First session!"
        elif sessions < 5:
            greeting = f"Welcome back, {name}! Session #{sessions+1}"
        else:
            greeting = f"Great to see you, {name}! #{sessions+1} session"

        # Load session history for this user
        history = self._load_user_history(name)

        return {
            'name':        name,
            'greeting':    greeting,
            'sessions':    sessions,
            'registered':  user.get('registered', 'N/A'),
            'history':     history
        }

    def _load_user_history(self, name):
        """Load sessions belonging to this user"""
        log_file = 'session_logs/sessions_history.json'
        if not os.path.exists(log_file):
            return []
        with open(log_file, 'r') as f:
            all_sessions = json.load(f)
        return [s for s in all_sessions
                if s.get('user') == name]

    def update_session_count(self, name):
        """Call this when session ends"""
        if name in self.users:
            self.users[name]['total_sessions'] = \
                self.users[name].get('total_sessions', 0) + 1
            self._save_users()

    def get_all_users(self):
        return list(self.users.keys())