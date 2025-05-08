import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from sklearn.base import TransformerMixin, BaseEstimator

class NormalizerVST(BaseEstimator, TransformerMixin):
    def __init__(self, metadata: pd.DataFrame):
        """
        Initialize the VST Normalizer with metadata.

        Parameters:
        metadata (pd.DataFrame): Metadata containing sample conditions.
        """
        if "condition" not in metadata.columns:
            raise ValueError("Metadata must contain a 'condition' column")
        self.metadata = metadata

    def fit(self, X, y=None):
        """Fit method required for sklearn Pipeline compatibility."""

        metadata = self.metadata.loc[X.index, :]  # Align metadata with X

        self.dds = DeseqDataSet(
            counts=X, metadata=metadata, design='~1', quiet=True
        )
        self.dds.vst_fit(use_design=False)
        return self # Return self for sklearn Pipeline compatibility

    def transform(self, X):
        """Apply Variance Stabilizing Transformation (VST)."""
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        # Check if the DataFrame is empty
        if X.empty:
            raise ValueError("Input DataFrame is empty.")
        # Check if the DataFrame contains only numeric values
        
        # Add 1 to avoid log(0) issues
        X = X + 1

        vst_counts = self.dds.vst_transform(X)  # Apply VST transformation
        return vst_counts
