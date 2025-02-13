import pandas as pd
import os

def threshold_filter(count_data: pd.DataFrame, min_count: int, min_samples: int) -> pd.DataFrame:
    """Filter out genes that have a count less than the min_count in min_samples samples."""
    print(
        f"Threshold filtering all genes with a count less than the {min_count} in {min_samples} samples"
    )
    threshold_filtered_count_data = count_data.loc[
        (count_data > min_count).sum(axis=1) >= min_samples
    ]
    return threshold_filtered_count_data

def run_threshold_filter(count_file, min_count, min_samples, output_path):
    # Read the count data
    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)

    # Filter out genes that have a count less than the min_count in min_samples samples
    threshold_filtered_count_data = threshold_filter(count_data, min_count, min_samples)

    # Save the filtered data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    threshold_filtered_count_data.to_csv(output_path, sep=";", index=True, header=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter out genes that have a count less than the min_count in min_samples samples.")
    parser.add_argument("--count_file", required=True, help="Path to the RNA count file.")
    parser.add_argument("--min_count", required=True, help="Minimum count value.")
    parser.add_argument("--min_samples", required=True, help="Minimum number of samples.")
    parser.add_argument("--output_path", required=True, help="Path to save the filtered data.")
    args = parser.parse_args()

    run_threshold_filter(
        count_file=args.count_file,
        min_count=int(args.min_count),
        min_samples=int(args.min_samples),
        output_path=args.output_path
    )