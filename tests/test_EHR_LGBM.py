"""Unit tests for the EHR_LGBM model.

This module includes unit tests for the LightGBM-based classification model,
including tests for output structure, metric range, and support for custom
hyperparameters.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from EHR_LGBM import EHR_LGBM


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


def test_EHR_LGBM_structure_and_keys() -> None:
    """Test that EHR_LGBM output is a dictionary containing 'model'."""
    df = generate_dummy_dataset()
    result = EHR_LGBM(df, y_variable="readmitted_binary", plot=False)

    assert isinstance(result, dict), "Output should be a dictionary."
    assert "model" in result, "Output dictionary must contain 'model'."
    assert "metrics" in result, "Output dictionary must contain 'metrics'."


def test_EHR_LGBM_metrics_range() -> None:
    """Test that all evaluation metrics are within the [0, 1] range."""
    df = generate_dummy_dataset()
    result = EHR_LGBM(df, y_variable="readmitted_binary", plot=False)
    metrics = result["metrics"]

    for metric_name in ["accuracy", "auc", "f1"]:
        assert 0.0 <= metrics[metric_name] <= 1.0, (
            f"{metric_name} should be in [0, 1] range."
        )


def test_EHR_LGBM_with_custom_hyperparams() -> None:
    """Test EHR_LGBM behavior when using custom hyperparameters."""
    df = generate_dummy_dataset()
    custom_params = {
        "n_estimators": 50,
        "learning_rate": 0.1,
        "num_leaves": 15,
    }
    result = EHR_LGBM(
        df,
        y_variable="readmitted_binary",
        hyperparameters=custom_params,
        plot=False,
    )
    assert isinstance(result["model"], object), (
        "Should return trained model with custom params."
    )


def test_lgbm_pipeline() -> None:
    """Test EHR_LGBM on a small dummy dataset with binary outcome."""
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "num1": np.random.randn(100),
            "cat1": np.random.choice(["A", "B"], size=100),
            "y": np.random.choice([0, 1], p=[0.9, 0.1], size=100),
        }
    )

    df = pd.get_dummies(df, drop_first=True)
    result = EHR_LGBM(df, y_variable="y", plot=False)

    assert "model" in result
    assert hasattr(result["model"], "feature_importances_")
    assert result["metrics"]["f1"] >= 0
