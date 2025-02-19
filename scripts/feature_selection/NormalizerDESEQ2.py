import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from sklearn.base import TransformerMixin, BaseEstimator
from pydeseq2 import preprocessing as deseq2_preprocess

class NormalizerDESEQ2(BaseEstimator, TransformerMixin):
    def __init__(
        self,
    ):
        """
        Initialize the VST Normalizer with metadata.

        Parameters:
        metadata (pd.DataFrame): Metadata containing sample conditions.
        """

    def fit(self, X, y=None):
        """Fit method required for sklearn Pipeline compatibility."""

        self.logmeans, self.filtered_genes = deseq2_preprocess.deseq2_norm_fit(X)
        return self  # Return self for sklearn Pipeline compatibility

    def transform(self, X):
        """Apply Variance Stabilizing Transformation (VST)."""
        deseq2_normalized_counts, _ = deseq2_preprocess.deseq2_norm_transform(
            X, self.logmeans, self.filtered_genes
        )
        return deseq2_normalized_counts
