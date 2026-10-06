import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

# ── Load features ────────────────────────────────────────────────
print("Loading features...")
features_df = pd.read_csv('features/features.csv')

# ── Load original labels ─────────────────────────────────────────
print("Loading original labels...")
train_labels = pd.read_csv('dataset/Labels/TrainLabels.csv')
train_labels.columns = train_labels.columns.str.strip()

# ── Load frames index to match clip_id ───────────────────────────
frames_df = pd.read_csv('features/frames_index.csv')

# ── Map emotion from all 4 DAiSEE columns ────────────────────────
def map_emotion(row):
    scores = {
        'Engaged':    row['Engagement'],
        'Bored':      row['Boredom'],
        'Confused':   row['Confusion'],
        'Frustrated': row['Frustration']
    }
    return max(scores, key=scores.get)

train_labels['emotion'] = train_labels.apply(map_emotion, axis=1)
print("\nEmotion distribution in labels:")
print(train_labels['emotion'].value_counts())

# ── Merge features with emotion labels ───────────────────────────
frames_df['clip_id_clean'] = frames_df['clip_id']
merged = frames_df.merge(train_labels[['ClipID', 'emotion']],
                         left_on='clip_id_clean',
                         right_on='ClipID',
                         how='left')

# Add features
merged = merged.join(features_df[['EAR', 'MAR', 'Gaze_X',
                                   'Gaze_Y', 'Head_Pitch', 'Head_Yaw']])
merged = merged.dropna(subset=['emotion', 'EAR'])

print(f"\nMerged dataset size: {len(merged)}")
print("\nEmotion distribution after merge:")
print(merged['emotion'].value_counts())

# ── Prepare data ─────────────────────────────────────────────────
X = merged[['EAR', 'MAR', 'Gaze_X', 'Gaze_Y', 'Head_Pitch', 'Head_Yaw']]
y = merged['emotion']

# ── Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Balance with SMOTE ───────────────────────────────────────────
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print("Balanced distribution:")
print(pd.Series(y_train_bal).value_counts())

# ── Train ────────────────────────────────────────────────────────
print("\nTraining emotion model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_bal, y_train_bal)

# ── Evaluate ─────────────────────────────────────────────────────
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ── Save ─────────────────────────────────────────────────────────
joblib.dump(model, 'models/emotion_model.pkl')
print("\nEmotion model saved to models/emotion_model.pkl")