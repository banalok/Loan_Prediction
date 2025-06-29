# script to test different logic locally before cloud training and deployment

# import joblib

# model = joblib.load("G:\Alok\Loan_Prediction\models\\best_model.pkl")
# print(type(model))

# test_local_inference.py

import joblib
import pandas as pd
from src.feature_engineering import apply_feature_engineering
# Load the trained pipeline model
model_path = "G:\Alok\Loan_Prediction\models\\best_model.pkl" 
model = joblib.load(model_path)
print("Model loaded successfully.")

# sample data matching training input schema
sample = pd.DataFrame({
    "Loan_ID": ["LP999999"],        
    "Gender": ["Male"],
    "Married": ["Yes"],
    "Dependents": ["0"],
    "Education": ["Graduate"],
    "Self_Employed": ["No"],
    "ApplicantIncome": [5000],
    "CoapplicantIncome": [2000],
    "LoanAmount": [150],
    "Loan_Amount_Term": [360],
    "Credit_History": [1.0],
    "Property_Area": ["Urban"],
    "Loan_Status": ["Y"]            
})

csv= apply_feature_engineering(sample)
# Make a prediction
prediction = model.predict(csv)

# Print the prediction
print("Prediction result:", prediction)

