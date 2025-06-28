# sagemaker/sagemaker_inference.py

import pandas as pd
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer

ENDPOINT_NAME = "loan-prediction-endpoint"

# predictor
predictor = Predictor(endpoint_name=ENDPOINT_NAME, serializer=CSVSerializer())

# Example input sample 
sample = pd.DataFrame({
    "gender": ["Male"],
    "married": ["Yes"],
    "dependents": ["0"],
    "education": ["Graduate"],
    "self_employed": ["No"],
    "applicantincome": [5000],
    "coapplicantincome": [0],
    "loanamount": [200],
    "loan_amount_term": [360],
    "credit_history": [1.0],
    "property_area": ["Urban"]
})

# convert to csv
csv = sample.to_csv(index=False, header=False)

# Make prediction
response = predictor.predict(csv)

# print prediction
print("Prediction result:", response)

# delete endpoint when done to avoid costs
# predictor.delete_endpoint()
