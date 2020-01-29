import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt


def run_pca(data, column_set, n_pca=None):
    """Fit a PCA model on data for the columns in column_set."""
    cols=column_set.intersection(set(data.columns))
    
    if not n_pca:
        n_pca = len(cols)
    print(f'PCA excecuted with {n_pca} principal components.')

    pca = PCA(n_pca)
    pca.fit(data[cols])  

    plt.plot(pca.explained_variance_ratio_.cumsum())
    plt.xlabel('Components')
    plt.ylabel('Variance explained')
    plt.show()

    return pca, cols


def display_pca(pca, cols, component, n=5):
    """Visualise PCA results. Which variables are most important in each component."""

    n = max(component, n)

    return pd.DataFrame(pca.components_, columns=cols)[0:n+1].transpose()\
                        .sort_values(component, ascending=False)\
                        .style.bar(align='mid', color=['#d65f5f', '#5fba7d'])