import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("TAIPEI_diabetes.csv")  # Ensure correct file path

# Print column names to check
print("Column Names in Dataset:", df.columns)

# Define the correct target column
target_column = "Diabetic"  # Changed from 'Outcome' to 'Diabetic'
assert target_column in df.columns, f"Error: Column '{target_column}' not found!"

# Drop unnecessary columns if they exist
if "PatientID" in df.columns:
    df = df.drop(columns=["PatientID"])

# Select Features (X) and Target (y)
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target variable
print(f"Dataset Shape: X={X.shape}, y={y.shape}")

# Split dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features and keep column names
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Train the Random Forest model with class balancing
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler with compression
joblib.dump(model, "random_forest_model.pkl", compress=("gzip", 3))
joblib.dump(scaler, "scaler.pkl", compress=("gzip", 3))

print("✅ Data successfully processed!")