from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def create_preprocessor(df, config):
    scale_cols = config["preprocessing"]["scale_columns"]
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    if config["preprocessing"]["target_column"] in categorical_cols:
        categorical_cols.remove(config["preprocessing"]["target_column"])

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), scale_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ], remainder='passthrough')
    return preprocessor

def split_data(df, config):
    target = config["preprocessing"]["target_column"]
    df[target] = df[target].map({'Y': 1, 'N': 0})

    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X, y,
        test_size=config["model"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y
    )
