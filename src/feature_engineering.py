import pandas as pd
import numpy as np

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # fill missing values
    for col in ['Gender', 'Married', 'Self_Employed']:
        df[col] = df[col].fillna(df[col].mode()[0])
    df['Dependents'] = df['Dependents'].replace('3+', '3').fillna(df['Dependents'].mode()[0]).astype(int)
    df['LoanAmount'] = df.groupby(['Education', 'Self_Employed'])['LoanAmount'].transform(
        lambda x: x.fillna(x.median()))
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History_missing'] = df['Credit_History'].isnull().astype(int)
    df['Credit_History'] = df['Credit_History'].fillna(0)

    # derived features
    df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['HasCoapplicant'] = (df['CoapplicantIncome'] > 0).astype(int)
    df['EMI'] = df['LoanAmount'] * 1000 / df['Loan_Amount_Term']
    df['LoanToIncomeRatio'] = df['LoanAmount'] * 1000 / (df['TotalIncome'] + 1e-6)
    df['TotalIncome_log'] = np.log1p(df['TotalIncome'])

    # drop raw and irrelevant columns
    df.drop(columns=['Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'TotalIncome'], inplace=True)

    return df
