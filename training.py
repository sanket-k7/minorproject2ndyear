import pandas as pd
import numpy as np
import json
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

df = pd.read_csv("dataset.csv")

print("\nDataset Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())

# ============================================================
# STEP 2: CLEAN COLUMN NAMES
# ============================================================

df.columns = [col.strip() for col in df.columns]

# ============================================================
# STEP 3: IDENTIFY TARGET COLUMN
# ============================================================

target_column = "Disease"

if target_column not in df.columns:
    raise Exception(f"'{target_column}' column not found!")

# ============================================================
# STEP 4: GET SYMPTOM COLUMNS
# ============================================================

symptom_columns = [col for col in df.columns if col != target_column]

print("\nTotal Symptoms:", len(symptom_columns))

# ============================================================
# STEP 5: CREATE SYMPTOM LIST
# ============================================================

all_symptoms = []

for col in symptom_columns:
    symptoms = df[col].dropna().unique()

    for symptom in symptoms:
        symptom = str(symptom).strip().lower()

        if symptom and symptom != "nan":
            all_symptoms.append(symptom)

# Remove duplicates
all_symptoms = sorted(list(set(all_symptoms)))

print("\nUnique Symptoms:", len(all_symptoms))

# Save symptom list
with open("symptom_list.json", "w") as f:
    json.dump(all_symptoms, f, indent=4)

print("✓ symptom_list.json saved")

# ============================================================
# STEP 6: CREATE FEATURE MATRIX
# ============================================================

print("\nCreating binary feature vectors...")

X = []
y = []

for index, row in df.iterrows():

    symptom_vector = [0] * len(all_symptoms)

    row_symptoms = []

    for col in symptom_columns:

        symptom = str(row[col]).strip().lower()

        if symptom != "nan":
            row_symptoms.append(symptom)

    for symptom in row_symptoms:

        if symptom in all_symptoms:
            idx = all_symptoms.index(symptom)
            symptom_vector[idx] = 1

    X.append(symptom_vector)
    y.append(row[target_column])

X = np.array(X)
y = np.array(y)

print("Feature Matrix Shape:", X.shape)

# ============================================================
# STEP 7: LABEL ENCODING
# ============================================================

print("\nEncoding disease labels...")

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# Save label encoder
joblib.dump(label_encoder, "label_encoder.pkl")

print("✓ label_encoder.pkl saved")

# ============================================================
# STEP 8: FEATURE SCALING
# ============================================================

print("\nScaling features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

print("✓ scaler.pkl saved")

# ============================================================
# STEP 9: TRAIN TEST SPLIT
# ============================================================

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# ============================================================
# STEP 10: RANDOM FOREST TRAINING
# ============================================================

print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("✓ Model training completed")

# ============================================================
# STEP 11: MODEL EVALUATION
# ============================================================

print("\nEvaluating model...")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# ============================================================
# STEP 12: SAVE TRAINED MODEL
# ============================================================

print("\nSaving trained model...")

joblib.dump(model, "disease_predictor_rf.pkl")

print("✓ disease_predictor_rf.pkl saved")

# ============================================================
# STEP 13: FEATURE IMPORTANCE
# ============================================================

print("\nTop Important Symptoms:")

importance_df = pd.DataFrame({
    "Symptom": all_symptoms,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 20 Features:\n")
print(importance_df.head(20))

# ============================================================
# STEP 14: TEST PREDICTION
# ============================================================

print("\nTesting sample prediction...")

sample = X_test[0].reshape(1, -1)

prediction = model.predict(sample)

predicted_disease = label_encoder.inverse_transform(prediction)

print("\nPredicted Disease:", predicted_disease[0])

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print("""
Generated Files:
----------------
✓ disease_predictor_rf.pkl
✓ label_encoder.pkl
✓ scaler.pkl
✓ symptom_list.json

These files are ready for Streamlit deployment.
""")