# sagemaker/sagemaker_inference.py

import pandas as pd
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from src.feature_engineering import apply_feature_engineering

ENDPOINT_NAME = "loan-prediction-endpoint"

# predictor
predictor = Predictor(endpoint_name=ENDPOINT_NAME, serializer=CSVSerializer())

# sample data matching training raw input schema
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

engineered_sample = apply_feature_engineering(sample)
print(f"Engineered columns: {engineered_sample.columns}")
# convert to csv
csv = engineered_sample.to_csv(index=False, header=False)

# Make prediction
response = predictor.predict(csv)

# print prediction
print("Prediction result:", response)

# delete endpoint when done to avoid costs
# predictor.delete_endpoint()
