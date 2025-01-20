import os
import logging
import pandas as pd
import numpy as np
from Data import Data
from CountNormalizer import CountNormalizer
from CountScaler import CountScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns


class DataPreprocessor:
    """A class to for preprocessing the data. Holds the methods used for preprocessing the data."""

    def __init__(self, config_data: dict,
                 count_file_path: str,
                 meta_file_path: str,
                 count_test_file: str,
                 meta_test_file: str):
        self.count_file_path = count_file_path
        self.meta_file_path = meta_file_path
        self.count_test_file = count_test_file
        self.meta_test_file = meta_test_file
        self.config_data = config_data

    def preprocess(self) -> Data:
        """Preprocess the data and return the Data object."""
        self.count_data = pd.read_csv(self.count_file_path, delimiter=";", index_col=0, header=0)
        self.meta_data = pd.read_csv(self.meta_file_path, delimiter=";", index_col=0, header=0)
        self.count_data = self.count_data.T  # Transpose the count data

        if self.count_test_file and self.meta_test_file:
            self.count_train_data = self.count_data
            self.meta_train_data = self.meta_data

            self.count_test_data = pd.read_csv(
                self.count_test_file, delimiter=";", index_col=0, header=0
            )
            self.meta_test_data = pd.read_csv(
                self.meta_test_file, delimiter=";", index_col=0, header=0
            )
            self.count_test_data = count_test_data.T  # Transpose the count data

            # Normalize the count data
            normalized_count_test_data = count_normalizer.normalize(count_test_data)

            # Scale the count data
            self.count_test_data = count_scaler.scale(normalized_count_test_data)

        else:  # split the data into training and test set
            logging.info(
                "No optional test files provided. The data will be split into training and test set."
            )

            count_train_data, count_test_data, meta_train_data, meta_test_data = (
                train_test_split(
                    self.count_data,
                    self.meta_data,
                    test_size=self.config_data["preprocessing"]["train_test_split_params"][
                        "test_size"
                    ],
                    random_state=self.config_data["preprocessing"][
                        "train_test_split_params"
                    ]["random_state"],
                )
            )

            self.count_train_data = count_train_data
            self.meta_train_data = meta_train_data
            self.count_test_data = count_test_data
            self.meta_test_data = meta_test_data

        # add threshold filtering on training data
        self.count_train_data = self.threshold_filter(self.count_train_data)

        count_normalizer = CountNormalizer(config_data=self.config_data)
        count_scaler = CountScaler(config_data=self.config_data)

        # Normalize the count data
        normalized_count_data = count_normalizer.normalize(count_data=self.count_train_data, metadata=self.meta_train_data)

        # Scale the count data
        self.count_train_data = count_scaler.scale(normalized_count_data)

        # apply pre-filtering
        self.pre_filter()

        data = Data(self.count_train_data, self.meta_train_data, self.count_test_data, self.meta_test_data)
        return data

    def threshold_filter(self, count_data: pd.DataFrame):
        """Filter out genes that have a count less than the min_count in min_samples samples."""
        min_count = self.config_data["preprocessing"]["pre_filter_methods"]["threshold_filter"]["min_count"]
        min_samples = self.config_data["preprocessing"]["pre_filter_methods"]["threshold_filter"]["min_samples"]
        logging.info(f"Threshold filtering all genes with a count less than the {min_count} in {min_samples} samples")
        threshold_filtered_count_data = count_data.loc[
            (count_data > min_count).sum(axis=1) >= min_samples
        ]
        return threshold_filtered_count_data

    def pca(self):
        """Perform PCA on the count data and write the results to a file in the output folder."""
        n_components = self.config_data["preprocessing"]["pre_filter_methods"]["pca"]["n_components"]
        logging.info(f"Performing PCA with {n_components} components")

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(self.count_train_data)
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

    def variance_filter(self):
        """Filter out genes that have a variance less than the threshold."""
        threshold = self.config_data["preprocessing"]["pre_filter_methods"]["variance_filter"]["threshold"]
        logging.info(f"Variance filtering all genes with a variance less than {threshold}")

        variance_threshold = VarianceThreshold(threshold=threshold)
        self.count_data = variance_threshold.fit_transform(self.count_data)

    def expr_percentile_filter(self):
        """Filter out genes that have an expression less than the threshold percentile."""
        threshold_percentile = self.config_data["preprocessing"]["pre_filter_methods"]["expr_percentile_filter"]["threshold_percentile"]
        logging.info(f"Expression percentile filtering all genes with an expression less than the {threshold_percentile} percentile")

        expr_percentile = self.count_data.quantile(q=threshold_percentile, axis=0)
        self.count_data = self.count_data.loc[:, (self.count_data > expr_percentile).any()]

    def correlation_filter(self):
        """Filter out genes that are highly correlated."""
        correlation_method = self.config_data["preprocessing"]["pre_filter_methods"]["correlation_filter"]["correlation_method"]
        threshold = self.config_data["preprocessing"]["pre_filter_methods"]["correlation_filter"]["threshold"]
        logging.info(f"Correlation filtering all genes with a correlation greater than {threshold}, using the {correlation_method} method")
        correlation_matrix = self.count_train_data.corr(method=correlation_method)

        # Find highly correlated pairs
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

        # Drop the genes
        self.count_train_data = self.count_train_data.drop(columns=to_drop)

    def pre_filter(self):
        pre_filter_methods = self.config_data["preprocessing"]["pre_filter_methods"]
        # logging.info("pre_filter_methods", pre_filter_methods)

        try:
            if pre_filter_methods["pca"]["use_method"]:
                self.pca()
            if pre_filter_methods["variance_filter"]["use_method"]:
                self.variance_filter()
            if pre_filter_methods["expr_percentile_filter"]["use_method"]:
                self.expr_percentile_filter()  
            if pre_filter_methods["correlation_filter"]["use_method"]:
                self.correlation_filter()
        except KeyError as e:
            raise KeyError(
                f'The method {e} is not implemented in the pre-filter method'
            )
