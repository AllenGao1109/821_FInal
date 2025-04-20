"""Train and evaluate an XGBoost model with support for class imbalance.

hyperparameter tuning, and feature importance extraction.
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier


def EHR_XGB(
    dataset: pd.DataFrame,
    y_variable: str,
    hyperparameters: Optional[dict[str, Any]] = None,
    plot: bool = True,
) -> dict[str, Any]:
    """Train an XGBoost model with optional hyperparameter tuning.

    class imbalance handling, and feature importance plot/output.
    """
    # 1. Separate X and y
    X = dataset.drop(columns=[y_variable])
    y = dataset[y_variable]

    # 2. One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Compute scale_pos_weight
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # 5. Hyperparameter tuning
    if hyperparameters is None:
        param_grid = {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [50, 100, 200],
            "subsample": [0.8, 1.0],
        }
        grid_search = GridSearchCV(
            estimator=XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
            ),
            param_grid=param_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
    else:
        model = XGBClassifier(
            **hyperparameters,
            use_label_encoder=False,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
        )
        model.fit(X_train, y_train)

    # 6. Predict and evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
    }

    # 7. Feature importance as table
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance_df = importance.sort_values(ascending=False).reset_index()
    importance_df.columns = ["feature", "importance"]

    # 8. Plot
    if plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=importance_df.head(20))
        plt.title("Top 20 Feature Importances (XGBoost)")
        plt.tight_layout()
        plt.show()

    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": importance_df,
    }
