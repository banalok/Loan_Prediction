from setuptools import setup, find_packages

setup(
    name="loan_prediction",
    version="0.1",
    packages=find_packages(), 
    install_requires=[
        "pandas==1.3.5",
        "numpy==1.21.6",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost",       
        "pyyaml",
        "joblib"
    ],
)
