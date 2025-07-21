import pandas as pd

def load_pima(path: str = 'data/pima.csv') -> tuple[pd.DataFrame, pd.Series]:
    """
    Load Pima Indians Diabetes Dataset.
    Args:
        path: Path to the CSV file.
    Returns:
        X: Features DataFrame
        y: Labels Series
    """
    df = pd.read_csv(path)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y


def load_frankfurt(path: str = 'data/frankfurt.csv') -> tuple[pd.DataFrame, pd.Series]:
    """
    Load Frankfurt Hospital Diabetes Dataset.
    Args:
        path: Path to the CSV file.
    Returns:
        X: Features DataFrame
        y: Labels Series
    """
    df = pd.read_csv(path)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y


def load_custom_csv(file) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a custom CSV uploaded by user.
    Args:
        file: File-like object or path to CSV file.
    Returns:
        X: Features DataFrame
        y: Labels Series
    Raises:
        ValueError: If 'Outcome' column is missing.
    """
    df = pd.read_csv(file)
    if 'Outcome' not in df.columns:
        raise ValueError("CSV must contain an 'Outcome' column for labels")
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return X, y
