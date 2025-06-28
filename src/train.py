import joblib
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.evaluate import evaluate_model  

def get_model(name: str, class_weight=None, scale_pos_weight=None, random_state=42):
    if name == "RandomForest":
        return RandomForestClassifier(class_weight=class_weight, random_state=random_state)
    elif name == "LogisticRegression":
        return LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=random_state)
    elif name == "XGBoost":
        return XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                             scale_pos_weight=scale_pos_weight, random_state=random_state)
    else:
        raise ValueError(f"Unsupported model type: {name}")

def train_models(X_train, y_train, X_test, y_test, preprocessor, config):
    models = {}
    test_f1_scores = {}

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    for model_name in config["model"]["classifiers"]:
        if model_name == "XGBoost":
            clf = get_model(model_name, scale_pos_weight=pos_weight, random_state=config["model"]["random_state"])
        else:
            clf = get_model(model_name, class_weight='balanced', random_state=config["model"]["random_state"])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        # train
        pipeline.fit(X_train, y_train)
        y_test_pred = pipeline.predict(X_test)
        test_f1 = f1_score(y_test, y_test_pred)

        # save model and tese F1 score
        models[model_name] = pipeline
        test_f1_scores[model_name] = test_f1

        print(f"Trained {model_name} with train accuracy: {test_f1:.4f}")

        # evaluate on test set
        evaluate_model(pipeline, X_test, y_test, config, model_name=model_name)

    # best model based on test F1 score
    best_model_name = max(test_f1_scores, key=test_f1_scores.get)
    best_model = models[best_model_name]

    # Save best model
    joblib.dump(best_model, config["output"]["model_dir"] + "best_model.pkl")
    print(f"Saved best model based on test F1 score: {best_model_name}")

    return best_model, best_model_name
