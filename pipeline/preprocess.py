import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Optional, Tuple

def get_imputer(name: str) -> Optional[object]:
    """
    Return an imputer instance by name.
    Args:
        name: Name of the imputation method.
    Returns:
        Imputer instance or None.
    """
    return {
        'none': None,
        'mean': SimpleImputer(strategy='mean'),
        'median': SimpleImputer(strategy='median'),
        'knn': KNNImputer(n_neighbors=5),
        'iterative': IterativeImputer(max_iter=10, random_state=42),
    }[name]


def get_scaler(name: str) -> object:
    """
    Return a scaler instance by name.
    Args:
        name: Name of the scaling method.
    Returns:
        Scaler instance.
    """
    return {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(),
    }[name]


def fit_preprocessing(X: pd.DataFrame, imputer: Optional[object], scaler: object) -> Tuple[pd.DataFrame, Optional[object], object]:
    """
    Fit imputer and scaler on X and transform X.
    Args:
        X: Input features DataFrame.
        imputer: Imputer instance or None.
        scaler: Scaler instance.
    Returns:
        X: Transformed DataFrame
        imputer: Fitted imputer
        scaler: Fitted scaler
    """
    columns = X.columns
    if imputer:
        X = imputer.fit_transform(X)
    X = scaler.fit_transform(X)
    return pd.DataFrame(X, columns=columns), imputer, scaler


def transform_input(input_df: pd.DataFrame, imputer: Optional[object], scaler: object, feature_names: list) -> pd.DataFrame:
    """
    Transform user input using fitted imputer and scaler.
    Args:
        input_df: DataFrame with user input.
        imputer: Fitted imputer or None.
        scaler: Fitted scaler.
        feature_names: List of feature names.
    Returns:
        Transformed DataFrame with same columns as feature_names.
    """
    if imputer:
        input_df = imputer.transform(input_df)
    input_df = scaler.transform(input_df)
    return pd.DataFrame(input_df, columns=feature_names)
