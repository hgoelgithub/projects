import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1️⃣ Load CSV
df = pd.read_csv("diabetic_data.csv")
print("Shape:", df.shape)

# 2️⃣ Drop ID columns (if any)
df = df.drop(columns=['encounter_id', 'patient_nbr'], errors='ignore')

# 3️⃣ Features + Target
target_col = "readmitted"  # adjust if your target is different
X = df.drop(columns=[target_col])
y = df[target_col]

# 🔑 4️⃣ Label encode target & clean symbols
# Example: remove any leading/trailing whitespace and unwanted chars
y_clean = y.astype(str).str.strip().str.replace('[<>]', '', regex=True)
print("Unique target labels BEFORE clean:", y.unique())
print("Unique target labels AFTER clean:", y_clean.unique())

# Label encode
le = LabelEncoder()
y_encoded = le.fit_transform(y_clean)
print("Classes:", le.classes_)
print("Encoded labels:", set(y_encoded))

# 5️⃣ One-hot encode features
categorical_cols = X.select_dtypes(include="object").columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# ✅ Clean column names to remove special chars: [ ] < > space
X_encoded.columns = (
    X_encoded.columns
    .str.replace('[', '_', regex=False)
    .str.replace(']', '_', regex=False)
    .str.replace('<', 'lt_', regex=False)
    .str.replace('>', 'gt_', regex=False)
    .str.replace(' ', '_', regex=False)
)

print("Example feature names:", X_encoded.columns.tolist()[:10])

# 6️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42
)

# 7️⃣ Train XGBoost
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric="mlogloss",
    n_jobs=-1
)

model.fit(X_train, y_train)

# 8️⃣ Predict + score
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ XGBoost Accuracy: {acc:.4f}")

# 8️⃣ Predict + score
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print(f"\n✅ XGBoost Training Accuracy: {train_acc:.4f}")
print(f"✅ XGBoost Test Accuracy:     {test_acc:.4f}")
