import argparse
import pandas as pd
import os
from src.feature_engineering import apply_feature_engineering
from src.preprocessing import split_data, create_preprocessor
from src.train import train_models
from src.config import load_config
import joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    args = parser.parse_args()

    # Load data
    df = pd.read_csv(os.path.join(args.train, "loan_data.csv"))

    # apply pipeline
    df = apply_feature_engineering(df)
    cfg = load_config()
    X_train, X_test, y_train, y_test = split_data(df, cfg)

    preprocessor = create_preprocessor(df, cfg)

    # Train model
    model = train_models(X_train, y_train, X_test, y_test, preprocessor, cfg)

    # Save model artifact
    joblib.dump(model, os.path.join(args.model_dir, "model.pkl"))
    print("Model saved.")

if __name__ == "__main__":
    main()
