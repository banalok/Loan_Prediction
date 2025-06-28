# sagemaker/sagemaker_deploy.py

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# initialize session and clients
session = sagemaker.Session()
sm_client = boto3.client('sagemaker')
iam_client = boto3.client('iam')

# yget SageMaker execution role
role_name = 'LoanPredictionSMRole'  
role_arn = iam_client.get_role(RoleName=role_name)['Role']['Arn']
print(f"Using SageMaker role ARN: {role_arn}")

# get SageMaker default bucket
bucket = session.default_bucket()
print(f"Using S3 bucket: {bucket}")

# retrieve the latest training job name dynamically
training_jobs = sm_client.list_training_jobs(SortBy='CreationTime', SortOrder='Descending', MaxResults=1)
latest_job_name = training_jobs['TrainingJobSummaries'][0]['TrainingJobName']
print(f"Latest training job: {latest_job_name}")

# get model artifacts S3 path from the latest training job
job_desc = sm_client.describe_training_job(TrainingJobName=latest_job_name)
model_artifacts = job_desc['ModelArtifacts']['S3ModelArtifacts']
print(f"Model artifact path: {model_artifacts}")

# SKLearn model object for deployment
# Clean up existing endpoint and endpoint config if they exist
endpoint_name = "loan-prediction-endpoint"

# Delete endpoint if exists
try:
    sm_client.describe_endpoint(EndpointName=endpoint_name)
    print(f"Deleting existing endpoint: {endpoint_name}")
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    waiter = sm_client.get_waiter('endpoint_deleted')
    waiter.wait(EndpointName=endpoint_name)
    print(f"Deleted endpoint: {endpoint_name}")
except sm_client.exceptions.ClientError as e:
    if "Could not find" in str(e) or "does not exist" in str(e):
        print(f"No existing endpoint found: {endpoint_name}")
    else:
        raise e

# Delete endpoint config if exists
try:
    sm_client.describe_endpoint_config(EndpointConfigName=endpoint_name)
    print(f"Deleting existing endpoint config: {endpoint_name}")
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    print(f"Deleted endpoint config: {endpoint_name}")
except sm_client.exceptions.ClientError as e:
    if "Could not find" in str(e) or "does not exist" in str(e):
        print(f"No existing endpoint config found: {endpoint_name}")
    else:
        raise e

# Create SKLearn model object for deployment
model = SKLearnModel(
    model_data=model_artifacts,
    role=role_arn,
    entry_point="inference.py",
    source_dir="./src", 
    py_version="py3",
    framework_version="1.2-1",      # scikit-learn version
    sagemaker_session=session
)

# Deploy the model to endpoint
predictor = model.deploy(
    instance_type="ml.m5.large",
    initial_instance_count=1,
    endpoint_name=endpoint_name,
)

print(f"Model deployed to SageMaker endpoint: {endpoint_name}")
