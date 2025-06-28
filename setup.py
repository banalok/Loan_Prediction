from setuptools import setup, find_packages

setup(
    name="loan_prediction",
    version="0.1",
    packages=find_packages(), 
    install_requires=[
        "pandas==1.1.3",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "xgboost",       
        "pyyaml",
        "joblib"
    ],
)
