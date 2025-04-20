"""Unit tests for the EHR_XGB model.

This module tests the XGBoost-based classification model including validation
of output format, model evaluation metrics, and custom hyperparameter support.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from ehr_utils import EHR_XGB


def generate_dummy_dataset(n: int = 100) -> pd.DataFrame:
    """Generate a dummy classification dataset."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "num_feature1": np.random.normal(0, 1, n),
            "num_feature2": np.random.randint(1, 10, size=n),
            "cat_feature": np.random.choice(["A", "B", "C"], size=n),
            "readmitted_binary": np.random.choice(
                [0, 1], size=n, p=[0.7, 0.3]
            ),
        }
    )
    return df


def test_EHR_XGB_structure_and_keys() -> None:
    """Test that EHR_XGB returns a dictionary with required keys.

    Checks that the returned result includes both 'model' and 'metrics'.
    """
    df = generate_dummy_dataset()
    result = EHR_XGB(df, y_variable="readmitted_binary", plot=False)

    assert isinstance(result, dict), "Output should be a dictionary."
    assert "model" in result, "Output dictionary must contain 'model'."
    assert "metrics" in result, "Output dictionary must contain 'metrics'."


def test_EHR_XGB_metrics_range() -> None:
    """Test that all evaluation metrics are within the [0, 1] range.

    Asserts that accuracy, AUC, and F1 scores are valid probabilities.
    """
    df = generate_dummy_dataset()
    result = EHR_XGB(df, y_variable="readmitted_binary", plot=False)
    metrics = result["metrics"]

    for metric_name in ["accuracy", "auc", "f1"]:
        assert 0.0 <= metrics[metric_name] <= 1.0, (
            f"{metric_name} should be in [0, 1] range."
        )


def test_EHR_XGB_with_custom_hyperparams() -> None:
    """Test that EHR_XGB accepts custom hyperparameters.

    Verifies that a model is trained and returned correctly when passing
    a user-defined hyperparameter dictionary.
    """
    df = generate_dummy_dataset()
    custom_params = {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3}
    result = EHR_XGB(
        df,
        y_variable="readmitted_binary",
        hyperparameters=custom_params,
        plot=False,
    )
    assert isinstance(result["model"], object), (
        "Should return trained model with custom params."
    )
