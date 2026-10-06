import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# ── Load features ────────────────────────────────────────────────
print("Loading features...")
df = pd.read_csv('features/features.csv')
print(f"Total samples: {len(df)}")
print(f"\nFocus distribution before balancing:")
print(df['focus_level'].value_counts())

# ── Prepare data ─────────────────────────────────────────────────
X = df[['EAR', 'MAR', 'Gaze_X', 'Gaze_Y', 'Head_Pitch', 'Head_Yaw']]
y = df['focus_level']

# ── Split first, then balance only training data ─────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ── Apply SMOTE to fix class imbalance ───────────────────────────
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"Balanced training distribution:")
print(pd.Series(y_train_bal).value_counts())

# ── Train Random Forest ──────────────────────────────────────────
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_bal, y_train_bal)
print("Training complete!")

# ── Evaluate ─────────────────────────────────────────────────────
print("\nEvaluating on test data...")
y_pred = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ── Feature Importance ───────────────────────────────────────────
print("\nFeature Importances:")
features = ['EAR', 'MAR', 'Gaze_X', 'Gaze_Y', 'Head_Pitch', 'Head_Yaw']
for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:12} → {imp:.4f}")

# ── Save model ───────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/cognito_model.pkl')
print("\nModel saved to models/cognito_model.pkl")