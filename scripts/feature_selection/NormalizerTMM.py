import numpy as np
import pandas as pd
import conorm
from sklearn.base import TransformerMixin, BaseEstimator


class NormalizerTMM(BaseEstimator, TransformerMixin):
    def __init__(self,):
        """
        Initialize the VST Normalizer with metadata.

        Parameters:
        metadata (pd.DataFrame): Metadata containing sample conditions.
        """

    def fit(self, X, y=None):
        """Fit method required for sklearn Pipeline compatibility."""

        return self  # Return self for sklearn Pipeline compatibility

    def transform(self, X):
        """Apply Variance Stabilizing Transformation (VST)."""
        return conorm.tmm(X)
