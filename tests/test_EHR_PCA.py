"""File for testing EHR_PCA function.

Work by Allen Gao
Last Updated April 19 2025.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.EHR_PCA import EHR_PCA  # Replace with your actual filename


def test_EHR_PCA_basic() -> None:
    """Test whether EHR_PCA correctly returns a list of important variables.

    They are for each principal component.
    """
    # Create a synthetic numeric DataFrame with some missing values.
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "age": np.random.normal(50, 10, 100),
            "bmi": np.random.normal(25, 5, 100),
            "glucose": np.random.normal(100, 15, 100),
            "cholesterol": np.append(
                np.random.normal(200, 20, 95), [np.nan] * 5
            ),
        }
    )

    # Run PCA analysis.
    num_components = 3
    important_vars = EHR_PCA(data, num=num_components, plot=False)

    # Check output type and length.
    assert isinstance(important_vars, list), "Output should be a list."
    assert len(important_vars) == num_components, (
        "List length should match num_components."
    )

    # Ensure all returned variables exist in the original dataset.
    for var in important_vars:
        assert var in data.columns, f"{var} is not a valid column name."


def test_EHR_PCA_plotting(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that EHR_PCA runs without error when plotting is enabled."""
    # Create numeric data.
    np.random.seed(0)
    data = pd.DataFrame(
        {
            "x1": np.random.rand(50),
            "x2": np.random.rand(50),
            "x3": np.random.rand(50),
        }
    )

    # Monkeypatch plt.show() to suppress actual plot display.
    monkeypatch.setattr(plt, "show", lambda: None)

    # Call the function with plot enabled.
    try:
        EHR_PCA(data, num=2, plot=True)
    except Exception as e:
        pytest.fail(f"Plotting failed with exception: {e}")
