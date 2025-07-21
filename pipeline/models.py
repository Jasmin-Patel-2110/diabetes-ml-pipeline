from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import GridSearchCV
from typing import Tuple, Optional, Dict, Any

# Hyperparameter grids
d_ = dict  # alias
PARAM_GRIDS = {
    'logistic': {
        # 'C': [0.1, 1],
        'solver': ['lbfgs', 'liblinear']
    },
    'svm': {
        'C': [0.7, 0.8, 0.9],
        'kernel': ['linear', 'rbf'],
        'gamma': [0.5, 0.65,0.80 ,1]
    },
    'knn': {
        'n_neighbors': [4, 5, 6],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    'tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [4, 5, 6],
        'min_samples_split': [5, 6, 7]
    },
    'rf': {
        'n_estimators': [50, 75],
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 4, 5, 6],
    }
}

def get_model(name: str) -> object:
    """
    Return a model instance by name.
    Args:
        name: Name of the model.
    Returns:
        Model instance.
    """
    return {
        'logistic': LogisticRegression(),
        'svm': SVC(probability=True),
        'knn': KNeighborsClassifier(),
        'tree': DecisionTreeClassifier(),
        'rf': RandomForestClassifier()
    }[name]
    
def tune_model(model: object, param_grid: Dict[str, Any], X_train, y_train) -> Tuple[object, Dict[str, Any], float]:
    """
    Run GridSearchCV to find the best estimator and params.
    Args:
        model: Model instance.
        param_grid: Hyperparameter grid.
        X_train: Training features.
        y_train: Training labels.
    Returns:
        best_estimator: Best model found.
        best_params: Best hyperparameters.
        best_score: Best cross-validation score.
    """
    grid = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def train_and_predict(model: object, X_train, y_train, X_input) -> Tuple[Any, Optional[Any]]:
    """
    Train the model and predict for input.
    Args:
        model: Model instance.
        X_train: Training features.
        y_train: Training labels.
        X_input: Input features for prediction.
    Returns:
        pred: Predicted class(es)
        prob: Predicted probabilities (if available)
    """
    model.fit(X_train, y_train)
    pred = model.predict(X_input)
    prob = model.predict_proba(X_input)[:, 1] if hasattr(model, "predict_proba") else None
    return pred, prob
