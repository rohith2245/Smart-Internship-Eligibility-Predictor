import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data.csv")

# Features and target
X = df.drop("Eligibility", axis=1)
y = df["Eligibility"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Final Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "internship_eligibility_model.pkl")
print("Model saved successfully")

# --------- Prediction for new student ----------
new_student = [[
    8.2,   # CGPA
    7,     # Coding Skills
    3,     # Projects
    0,     # Previous Internship
    2,     # Certifications
    7,     # Communication Skills
    85     # Attendance
]]

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Prediction: Eligible for Internship")
else:
    print("Prediction: Not Eligible for Internship")
