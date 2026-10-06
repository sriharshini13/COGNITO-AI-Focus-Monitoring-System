import numpy as np

class ConfidenceScorer:
    """
    Wraps sklearn model to return prediction
    confidence % alongside the label.
    Also computes drowsiness severity from EAR.
    """

    def __init__(self, focus_model, emotion_model):
        self.focus_model   = focus_model
        self.emotion_model = emotion_model

    def predict_focus(self, features):
        """
        Returns (label, confidence_pct, proba_dict)
        """
        feats = np.array(features).reshape(1, -1)
        label = self.focus_model.predict(feats)[0]
        proba = self.focus_model.predict_proba(feats)[0]
        classes = self.focus_model.classes_

        proba_dict = {
            c: round(float(p) * 100, 1)
            for c, p in zip(classes, proba)
        }
        confidence = round(
            float(max(proba)) * 100, 1)

        return label, confidence, proba_dict

    def predict_emotion(self, features):
        """
        Returns (label, confidence_pct, proba_dict)
        """
        feats = np.array(features).reshape(1, -1)
        label = self.emotion_model.predict(feats)[0]
        proba = self.emotion_model.predict_proba(feats)[0]
        classes = self.emotion_model.classes_

        proba_dict = {
            c: round(float(p) * 100, 1)
            for c, p in zip(classes, proba)
        }
        confidence = round(
            float(max(proba)) * 100, 1)

        return label, confidence, proba_dict

    def drowsiness_level(self, ear):
        """
        Returns severity level based on EAR value.
        EAR = Eye Aspect Ratio
        Normal: > 0.25
        Mild:     0.20 - 0.25
        Moderate: 0.17 - 0.20
        Severe:   < 0.17
        """
        if ear > 0.25:
            return 'Awake', 'none', '#a6e3a1'
        elif ear > 0.20:
            return 'Mild Drowsiness', 'mild', '#f9e2af'
        elif ear > 0.17:
            return 'Moderate Drowsiness', 'moderate', '#fab387'
        else:
            return 'Severe Drowsiness', 'severe', '#f38ba8'

    def attention_level(self, score):
        """
        Returns attention label from score 0-100
        """
        if score >= 80:
            return 'Highly Focused', '#a6e3a1'
        elif score >= 60:
            return 'Focused', '#89b4fa'
        elif score >= 40:
            return 'Moderate', '#fab387'
        elif score >= 20:
            return 'Distracted', '#f38ba8'
        else:
            return 'Very Distracted', '#f38ba8'