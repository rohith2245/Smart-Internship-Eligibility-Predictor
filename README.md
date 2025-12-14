# Smart Internship Eligibility Predictor

## Overview
The Smart Internship Eligibility Predictor is a Machine Learning–based system designed to predict whether a student is eligible for internship opportunities based on academic performance, technical skills, and related attributes.

This project aims to help students assess their readiness for internships and understand key factors influencing eligibility.

---

## Problem Statement
Many students apply for internships without clear insight into eligibility requirements. This project builds a predictive model that evaluates a student’s profile and classifies eligibility using Machine Learning techniques.

---

## Objectives
- Predict internship eligibility (Eligible / Not Eligible)
- Apply supervised machine learning techniques
- Compare multiple ML models and select the best-performing one
- Provide real-time predictions for new student data

---

## Dataset Description
A synthetic dataset was created to simulate real student profiles.

### Features Used:
- CGPA
- Coding Skills (1–10)
- Number of Projects
- Previous Internship Experience (0/1)
- Certifications Count
- Communication Skills (1–10)
- Attendance Percentage

### Target Variable:
- Eligibility (1 = Eligible, 0 = Not Eligible)

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

---

## Machine Learning Models
- Logistic Regression
- Random Forest Classifier

Random Forest was selected as the final model based on higher accuracy.

---

## Model Performance
- Final Model Accuracy: ~97%

---

## How to Run the Project
1. Clone the repository
2. Install required libraries:
