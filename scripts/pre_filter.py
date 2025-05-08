import yaml
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold

def threshold_filter(count_data: pd.DataFrame, min_count, min_samples) -> pd.DataFrame:
    """Filter out genes that have a count less than the min_count in min_samples samples."""

    threshold_filtered_count_data = count_data.loc[:, (count_data > min_count).sum(axis=0) >= min_samples]

    # print a lot of information
    print(f"Original count data shape: {count_data.shape}")
    print(f"Threshold filtered count data shape: {threshold_filtered_count_data.shape}")

    # show removed genes
    removed_genes = count_data.columns.difference(threshold_filtered_count_data.columns)
    print(f"Removed genes: {removed_genes}")

    return threshold_filtered_count_data


def variance_filter(count_data: pd.DataFrame, threshold) -> pd.DataFrame:
    """Filter out genes that have a variance less than the threshold."""
    print(f"Variance filtering all genes with a variance less than {threshold}")
    variance_threshold = VarianceThreshold(threshold=threshold)
    filtered_count_data = variance_threshold.fit_transform(count_data)

    return filtered_count_data


def expr_percentile_filter(count_data: pd.DataFrame, threshold_percentile: float) -> pd.DataFrame:
    """Filter out genes that have an expression less than the threshold percentile."""
    print(
        f"Expression percentile filtering all genes with an expression less than the {threshold_percentile} percentile"
    )
    expr_percentile = count_data.quantile(q=threshold_percentile, axis=0)
    filtered_count_data = count_data.loc[:, (count_data > expr_percentile).any()]

    return filtered_count_data


def correlation_filter(count_data: pd.DataFrame, correlation_method: str, threshold: float) -> pd.DataFrame:
    """Filter out genes that are highly correlated."""
    print(
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


def run_pre_filtering(count_data_file, config_file, output_path):
    """Returns the filtered data based on the method given."""

    # Read the config file
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    pre_filter_methods = [
        method
        for method, is_use in config_data["preprocessing"]["pre_filter_methods"].items()
        if is_use["use_method"]
    ]

    pre_filter_params = config_data["preprocessing"]["pre_filter_methods"]

    print("Running pre-filtering...")
    print(f"Count data file: {count_data_file}")
    print(f"Output path: {output_path}")

    # Read the count data
    count_data = pd.read_csv(count_data_file, delimiter=";", index_col=0, header=0)

    # Filter out genes based on the pre-filtering methods
    if "threshold_filter" in pre_filter_methods:
        min_count = pre_filter_params["threshold_filter"]["min_count"]
        min_samples = pre_filter_params["threshold_filter"]["min_samples"]
        count_data = threshold_filter(count_data, min_count, min_samples)

    if "variance_filter" in pre_filter_methods:
        threshold = pre_filter_params["variance_filter"]["threshold"]
        count_data = variance_filter(count_data, threshold)

    if "expr_percentile_filter" in pre_filter_methods:
        threshold_percentile = pre_filter_params["expr_percentile_filter"][
            "threshold_percentile"
        ]
        count_data = expr_percentile_filter(count_data, threshold_percentile)

    if "correlation_filter" in pre_filter_methods:
        correlation_method = pre_filter_params["correlation_filter"][
            "correlation_method"
        ]
        threshold = pre_filter_params["correlation_filter"]["threshold"]
        count_data = correlation_filter(count_data, correlation_method, threshold)

    # Save the filtered data
    count_data.to_csv(output_path, sep=";", index=True, header=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter out genes based on the pre-filtering methods.")
    parser.add_argument("--count_file", required=True, help="Path to the RNA count data file.")
    parser.add_argument("--config_file", required=True, help="Path to the config file.")
    parser.add_argument("--output_path", required=True, help="Path to save the filtered data.")
    args = parser.parse_args()

    run_pre_filtering(
        count_data_file=args.count_file,
        config_file=args.config_file,
        output_path=args.output_path
    )
