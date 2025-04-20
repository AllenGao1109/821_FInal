"""File for creating PCA analysis function.

Work by Allen Gao
Last Updated April 19 2025.

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def EHR_PCA(
    dataset: pd.DataFrame, num: int = 10, plot: bool = True
) -> list[str]:
    """Perform PCA on the dataset and return the most important variables.

    Args:
        dataset (pd.DataFrame): Input data, should be numeric.
        num (int): Number of principal components to retain.
        plot (bool): Whether to plot explained variance ratio.

    Returns:
        list: List of most important variables for each principal component.
    """
    # Ensure the input is a pandas DataFrame
    if not isinstance(dataset, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")

    # Select numeric columns and fill missing values with column means
    numeric_data = dataset.select_dtypes(include=np.number)
    numeric_data = numeric_data.fillna(numeric_data.mean())

    # Standardize the data (important for PCA)
    numeric_data = (numeric_data - numeric_data.mean()) / numeric_data.std(
        ddof=0
    )

    # Fit PCA
    pca = PCA(n_components=num)
    pca.fit(numeric_data)

    # Get the loading matrix (principal axes in feature space)
    loadings = pd.DataFrame(pca.components_, columns=numeric_data.columns)

    # For each principal component, find the variable with the highest one
    important_vars = [loadings.iloc[i].abs().idxmax() for i in range(num)]

    # Plot explained variance ratio if requested
    if plot:
        plt.figure(figsize=(10, 5))
        plt.bar(
            range(1, num + 1),
            pca.explained_variance_ratio_,
            alpha=0.7,
            color="steelblue",
        )
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Explained Variance by Principal Components")
        plt.xticks(range(1, num + 1))
        plt.tight_layout()
        plt.show()

    return important_vars
