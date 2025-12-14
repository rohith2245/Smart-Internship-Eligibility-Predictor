from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("internship_eligibility_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "CGPA": float(request.form["cgpa"]),
        "Coding_Skills": int(request.form["coding"]),
        "Projects": int(request.form["projects"]),
        "Previous_Internship": int(request.form["internship"]),
        "Certifications": int(request.form["certifications"]),
        "Communication_Skills": int(request.form["communication"]),
        "Attendance": int(request.form["attendance"])
    }

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    result = "Eligible for Internship" if prediction == 1 else "Not Eligible for Internship"

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
