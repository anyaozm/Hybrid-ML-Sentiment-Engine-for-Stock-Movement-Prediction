import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares features and target for prediction. Assumes 'sentiment' column exists.

    :param df: DataFrame with OHLCV + sentiment
    :return: Tuple of (features, target)
    """
    df = df.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    X = df[["Open", "High", "Low", "Volume", "sentiment"]].dropna()
    y = df["target"].dropna()
    return X.loc[y.index], y

def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Trains a RandomForestClassifier on the given features and target.

    :param X: Feature matrix
    :param y: Target vector
    :return: Trained RandomForestClassifier
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates model on test data.

    :param model: Trained model
    :param X_test: Features
    :param y_test: True labels
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
