# sagemaker/sagemaker_estimator.py

import sagemaker
from sagemaker.sklearn.estimator import SKLearn
from src.config import load_config
from src.data_loader import load_data
import os
import boto3

# initialize SageMaker session 
session = sagemaker.Session()

# initialize SageMaker role
# role = sagemaker.get_execution_role()  # if running in SageMaker Studio or Notebook instance
role = boto3.client('iam').get_role(RoleName='LoanPredictionSMRole')['Role']['Arn']

# default SageMaker S3 bucket
bucket = session.default_bucket()
prefix = "loan-prediction"

cfg = load_config()

# upload training data to S3
input_train = session.upload_data(
    path=cfg["data"]["path"], #"Data/loan_data.csv",  # local path to your dataset
    bucket=bucket,
    key_prefix=f"{prefix}/data"
)
print(f"Training data uploaded to: {input_train}")

# trainer
trainer = SKLearn(
    entry_point="train_script.py",  # your SageMaker training script
    source_dir=".",                 # uploads entire project (including setup.py and src/)
    role=role,
    instance_type="ml.m5.large",
    framework_version="1.2-1",      # scikit-learn version (adjust if needed)
    py_version="py3",
    hyperparameters={},             
    base_job_name="loan-prediction-training"
)

# Launch the training job
trainer.fit({"train": input_train})

print("SageMaker training job complete.")
