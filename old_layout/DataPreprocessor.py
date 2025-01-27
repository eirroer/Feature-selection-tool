import os
import logging
import pandas as pd
import numpy as np
from Data import Data
from PreFilter import PreFilter
from CountNormalizer import CountNormalizer
from MetadataScaler import MetadataScaler
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

        else:  # split the data into training and test set

            test_size = self.config_data["preprocessing"]["train_test_split_params"]["test_size"]
            random_state = self.config_data["preprocessing"]["train_test_split_params"]["random_state"]

            logging.info(
                f"No optional test files provided. The data will be split into training and test set. Test size: {test_size}, Random state: {random_state}" 
            )

            count_train_data, count_test_data, meta_train_data, meta_test_data = (
                train_test_split(
                    self.count_data,
                    self.meta_data,
                    test_size=test_size,
                    random_state=random_state,
                )
            )

            self.count_train_data = count_train_data
            self.meta_train_data = meta_train_data
            self.count_test_data = count_test_data
            self.meta_test_data = meta_test_data

        # apply threshold filtering on training data
        use_threshold_filter = self.config_data["preprocessing"]["threshold_filter"]["use_method"]

        if use_threshold_filter:
            self.count_train_data = self.threshold_filter(self.count_train_data)

        count_normalizer = CountNormalizer(config_data=self.config_data)
        metadata_scaler = MetadataScaler(config_data=self.config_data)

        # Normalize the count data
        self.count_train_data = count_normalizer.normalize(count_data=self.count_train_data, metadata=self.meta_train_data)

        # Scale the count data
        use_shrinkage_scaling = self.config_data["preprocessing"]["scaling_methods_count"]["shrinkage_scaling"]["use_method"]
        if use_shrinkage_scaling:
            self.count_train_data = self.shrinkage_scaling(self.count_train_data)

        # Scale the metadata
        self.meta_train_data = metadata_scaler.scale(metadata=self.meta_train_data, cols=pd.Series([]))

        # apply PCA
        use_pca = self.config_data["preprocessing"]["pca"]["use_method"]
        if use_pca:
            self.pca(self.count_train_data)

        # apply pre-filtering
        pre_filter = PreFilter(config_data=self.config_data)
        self.count_train_data = pre_filter.apply_pre_filters(count_data=self.count_train_data)

        data = Data(self.count_train_data, self.meta_train_data, self.count_test_data, self.meta_test_data)
        return data

    def threshold_filter(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Filter out genes that have a count less than the min_count in min_samples samples."""
        min_count = self.config_data["preprocessing"]["threshold_filter"]["min_count"]
        min_samples = self.config_data["preprocessing"]["threshold_filter"]["min_samples"]
        logging.info(f"Threshold filtering all genes with a count less than the {min_count} in {min_samples} samples")
        threshold_filtered_count_data = count_data.loc[
            (count_data > min_count).sum(axis=1) >= min_samples
        ]
        return threshold_filtered_count_data
    
    def shrinkage_scaling(self, count_data: pd.DataFrame) -> pd.DataFrame:
        """Apply shrinkage scaling to the count data."""
        # shrinkage_scaling = ShrinkageScaling()
        # X_scaled = shrinkage_scaling.fit_transform(count_data)
        # shrinkage_scaled_data = pd.DataFrame(
        #     X_scaled, columns=count_data.columns, index=count_data.index
        # )
        # return shrinkage_scaled_data
        raise NotImplementedError("Shrinkage scaling not implemented yet.")
    

    def pca(self, count_data: pd.DataFrame) -> None:
        """Perform PCA on the count data and write the results to a file in the output folder."""
        n_components = self.config_data["preprocessing"]["pca"][
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
