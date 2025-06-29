# script to run preprocessing and training pipeline locally

from src.config import load_config
from src.data_loader import load_data
from src.feature_engineering import apply_feature_engineering
from src.preprocessing import split_data, create_preprocessor
from src.train import train_models

def main():
    # config and data load
    cfg = load_config()
    df = load_data(cfg["data"]["path"])

    # apply domainspecific feature engineering
    df = apply_feature_engineering(df)

    X_train, X_test, y_train, y_test = split_data(df, cfg)

    # create column transformer preprocessor
    preprocessor = create_preprocessor(X_train, cfg)

    # train and evaluate all models, return the best one based on test F1 score
    best_model, best_model_name = train_models(X_train, y_train, X_test, y_test, preprocessor, cfg)

    print(f"\n Pipeline complete. Best model: {best_model_name}")

if __name__ == "__main__":
    main()
