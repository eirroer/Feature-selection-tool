import numpy as np
import pandas as pd
import rnanorm
from sklearn.base import TransformerMixin, BaseEstimator


class NormalizerTMM(BaseEstimator, TransformerMixin):
    def __init__(self,):
        """
        Initialize the VST Normalizer with metadata.
        """
        # Initialize any parameters if needed
        self.tmm = rnanorm.TMM()
        self.tmm.set_output(transform="pandas")

    def fit(self, X, y=None):
        """Fit method required for sklearn Pipeline compatibility."""

        return self  # Return self for sklearn Pipeline compatibility

    def transform(self, X):
        """Apply Trimmed Mean of M-values (TMM) normalization to the input data."""

        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")

        # Check if the DataFrame is empty
        if X.empty:
            raise ValueError("Input DataFrame is empty.")

        # Check if the DataFrame contains only numeric values
        if not np.issubdtype(X.values.dtype, np.number):
            raise ValueError("Input DataFrame must contain only numeric values.")

        # Perform TMM normalization
        tmm_normalized_data = self.tmm.fit_transform(X)

        return tmm_normalized_data


