import os
import logging
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt
import seaborn as sns


class PreFilter:
    def __init__(self, config_data: dict):
        self.config_data = config_data

    def pca(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Perform PCA on the count data and write the results to a file in the output folder."""
        n_components = self.config_data["preprocessing"]["pre_filter_methods"]["pca"][
            "n_components"
        ]
        logging.info(f"Performing PCA with {n_components} components")

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(count_data)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

        def plot_pca(pca_df: pd.DataFrame):
            """Plot the PCA results."""
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x="PC1", y="PC2", data=pca_df)
            plt.title("PCA Plot of Mock Dataset")
            plt.xlabel(
                f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)"
            )
            plt.ylabel(
                f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)"
            )
            plt.tight_layout()
            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            plt.savefig("outputs/pca_plot.png", dpi=300, format="png")

        plot_pca(pca_df)

        return count_data

    def variance_filter(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Filter out genes that have a variance less than the threshold."""
        threshold = self.config_data["preprocessing"]["pre_filter_methods"][
            "variance_filter"
        ]["threshold"]
        logging.info(
            f"Variance filtering all genes with a variance less than {threshold}"
        )

        variance_threshold = VarianceThreshold(threshold=threshold)
        filtered_count_data = variance_threshold.fit_transform(count_data)

        return filtered_count_data

    def expr_percentile_filter(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Filter out genes that have an expression less than the threshold percentile."""
        threshold_percentile = self.config_data["preprocessing"]["pre_filter_methods"][
            "expr_percentile_filter"
        ]["threshold_percentile"]
        logging.info(
            f"Expression percentile filtering all genes with an expression less than the {threshold_percentile} percentile"
        )

        expr_percentile = count_data.quantile(q=threshold_percentile, axis=0)
        filtered_count_data = count_data.loc[
            :, (count_data > expr_percentile).any()
        ]

        return filtered_count_data

    def correlation_filter(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Filter out genes that are highly correlated."""
        correlation_method = self.config_data["preprocessing"]["pre_filter_methods"][
            "correlation_filter"
        ]["correlation_method"]
        threshold = self.config_data["preprocessing"]["pre_filter_methods"][
            "correlation_filter"
        ]["threshold"]
        logging.info(
            f"Correlation filtering all genes with a correlation greater than {threshold}, using the {correlation_method} method"
        )
        correlation_matrix = count_data.corr(method=correlation_method)

        # Find highly correlated pairs
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            column
            for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        # Drop the genes
        filtered_count_data = count_data.drop(columns=to_drop)
        return filtered_count_data

    def apply_pre_filters(self, count_data: pd.DataFrame) -> pd.DataFrame:
        pre_filter_methods = self.config_data["preprocessing"]["pre_filter_methods"]
        active_pre_filter_methods = {
            key: pre_filter_methods[key]
            for key in pre_filter_methods
            if pre_filter_methods[key]["use_method"]
        }
        logging.info(
            "Pre-filtering methods activated in config file", active_pre_filter_methods
        )

        if count_data is None:
            raise ValueError("Count data is None in the pre filter method. Please provide count data.")

        try:
            if pre_filter_methods["pca"]["use_method"]:
                count_data = self.pca(count_data=count_data)
            if pre_filter_methods["variance_filter"]["use_method"]:
                count_data = self.variance_filter(count_data=count_data)
            if pre_filter_methods["expr_percentile_filter"]["use_method"]:
                count_data = self.expr_percentile_filter(count_data=count_data)
            if pre_filter_methods["correlation_filter"]["use_method"]:
                count_data = self.correlation_filter(count_data=count_data)
        except KeyError as e:
            raise KeyError(
                f"The method {e} is not implemented in the pre-filter methods"
            )

        return count_data
