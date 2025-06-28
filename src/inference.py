# inference.py

import os
import joblib
import pandas as pd

def model_fn(model_dir):
    """
    Load and return the trained pipeline model.
    """
    model_path = os.path.join(model_dir, "model.pkl")
    print(f"Loading model from {model_path}")
    loaded = joblib.load(model_path)
    if isinstance(loaded, tuple):
        model = loaded[0]
    else:
        model = loaded
    print("Model loaded successfully.")
    print("Model type:", type(model))
    return model

def input_fn(input_data, content_type):
    """
    Deserialize input data to pandas DataFrame.
    """
    print(f"Parsing input data with content_type: {content_type}")
    if content_type == "text/csv":
        from io import StringIO
        df = pd.read_csv(StringIO(input_data))
        print(f"Input data columns: {df.columns}")
        print("Input data parsed successfully.")
        return df
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """
    Generate prediction using the loaded model pipeline.
    """
    print("Making prediction.")
    predictions = model.predict(input_data)
    print("Prediction complete.")
    return predictions

def output_fn(prediction, accept):
    """
    Format prediction output.
    """
    print(f"Formatting output as {accept}")
    if accept == "application/json":
        return {"predictions": prediction.tolist()}, accept
    elif accept == "text/csv":
        output = pd.DataFrame(prediction).to_csv(index=False, header=False)
        return output, accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
