"""Train and evaluate a LightGBM model.

Including optional hyperparameter tuning and feature importance plotting.
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'src')
    )
)


def EHR_LGBM(dataset: pd.DataFrame, y_variable: str,
             hyperparameters: dict | None = None,
             plot: bool = True) -> dict:
    """Train LightGBM on input dataset and return trained model and metrics.

    Args:
        dataset (pd.DataFrame): Cleaned input data.
        y_variable (str): Column name of the binary outcome.
        hyperparameters (dict, optional): User-defined LightGBM parameters.
        plot (bool): Whether to visualize feature importances.

    Returns:
        dict: {'model': trained_model, 'metrics': {...}}
    """
    X = dataset.drop(columns=[y_variable])
    y = dataset[y_variable]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    if hyperparameters is None:
        param_grid = {
            "n_estimators": [100],
            "learning_rate": [0.1],
            "num_leaves": [15, 31]
        }
        grid = GridSearchCV(
            LGBMClassifier(), 
            param_grid, 
            cv=3, 
            scoring="roc_auc")
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
    else:
        model = LGBMClassifier(**hyperparameters)
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred)
    }

    if plot:
        importance = pd.Series(model.feature_importances_, index=X.columns)
        top = importance.sort_values(ascending=False).head(20)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=top, y=top.index)
        plt.title("Top 20 Feature Importances")
        plt.tight_layout()
        plt.show()

    return {"model": model, "metrics": metrics}

