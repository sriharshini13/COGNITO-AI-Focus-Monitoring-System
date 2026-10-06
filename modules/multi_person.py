import numpy as np

class MultiPersonTracker:
    def __init__(self):
        self.people = {}  # person_id -> stats

    def update(self, face_landmarks_list, focus_model,
               emotion_model, w, h):
        """
        face_landmarks_list: list of all detected face landmarks
        Returns list of dicts with per-person predictions
        """
        results = []

        for idx, lm in enumerate(face_landmarks_list):
            try:
                ear            = self._get_EAR(lm, w, h)
                mar            = self._get_MAR(lm, w, h)
                gaze_x, gaze_y = self._get_gaze(lm, w, h)
                pitch, yaw     = self._get_head_pose(lm, w, h)
                feats          = np.array([[ear, mar, gaze_x,
                                            gaze_y, pitch, yaw]])

                focus   = focus_model.predict(feats)[0]
                emotion = emotion_model.predict(feats)[0]

                # Get face bounding box for drawing
                xs = [pt.x * w for pt in lm]
                ys = [pt.y * h for pt in lm]
                bbox = (int(min(xs)), int(min(ys)),
                        int(max(xs)), int(max(ys)))

                # Track per person
                pid = f"Person {idx+1}"
                if pid not in self.people:
                    self.people[pid] = {
                        'focus_counts':   {'High':0,'Medium':0,'Low':0},
                        'emotion_counts': {'Engaged':0,'Bored':0,
                                          'Confused':0,'Frustrated':0}
                    }
                self.people[pid]['focus_counts'][focus]     += 1
                self.people[pid]['emotion_counts'][emotion] += 1

                results.append({
                    'id':      pid,
                    'focus':   focus,
                    'emotion': emotion,
                    'bbox':    bbox,
                    'ear':     ear,
                    'mar':     mar,
                })
            except:
                continue

        return results

    def get_classroom_summary(self):
        """Returns summary of all people tracked"""
        summary = {}
        for pid, data in self.people.items():
            total_f = sum(data['focus_counts'].values())   or 1
            total_e = sum(data['emotion_counts'].values()) or 1
            summary[pid] = {
                'dominant_focus':
                    max(data['focus_counts'],
                        key=data['focus_counts'].get),
                'dominant_emotion':
                    max(data['emotion_counts'],
                        key=data['emotion_counts'].get),
                'high_focus_pct':
                    round(data['focus_counts']['High']/total_f*100, 1),
                'engaged_pct':
                    round(data['emotion_counts']['Engaged']/total_e*100, 1),
            }
        return summary

    # ── Feature functions ────────────────────────────────────────
    def _euclidean(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def _get_EAR(self, lm, w, h):
        L = [(lm[i].x*w, lm[i].y*h)
             for i in [33,160,158,133,153,144]]
        R = [(lm[i].x*w, lm[i].y*h)
             for i in [362,385,387,263,373,380]]
        def ear(e):
            return (self._euclidean(e[1],e[5]) +
                    self._euclidean(e[2],e[4])) / \
                   (2.0 * self._euclidean(e[0],e[3]))
        return (ear(L) + ear(R)) / 2.0

    def _get_MAR(self, lm, w, h):
        m = [(lm[i].x*w, lm[i].y*h)
             for i in [61,291,13,14,78,308]]
        return (self._euclidean(m[2],m[3]) +
                self._euclidean(m[4],m[5])) / \
               (2.0 * self._euclidean(m[0],m[1]))

    def _get_gaze(self, lm, w, h):
        return ((lm[468].x+lm[473].x)/2.0,
                (lm[468].y+lm[473].y)/2.0)

    def _get_head_pose(self, lm, w, h):
        return ((lm[1].y  - lm[152].y),
                (lm[263].x - lm[33].x))