import pandas as pd
import os

def threshold_filter(count_file, metadata_file, min_count, min_samples, output_path_count):
    """Filter out genes that have a count less than the min_count in min_samples samples."""
    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    threshold_filtered_count_data = count_data.loc[:, (count_data > min_count).sum(axis=0) >= min_samples]

    # print a lot of information
    print(f"Original count data shape: {count_data.shape}")
    print(f"Threshold filtered count data shape: {threshold_filtered_count_data.shape}")

    # show removed genes
    removed_genes = count_data.columns.difference(threshold_filtered_count_data.columns)
    print(f"Removed genes: {removed_genes}")


    # Save the filtered count data
    os.makedirs(os.path.dirname(output_path_count), exist_ok=True)
    threshold_filtered_count_data.to_csv(
        output_path_count, sep=";", index=True, header=True
    )



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Filter out genes that have a count less than the min_count in min_samples samples.")
    parser.add_argument("--count_file", required=True, help="Path to the RNA count file.")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata file.")
    parser.add_argument("--min_count", required=True, help="Minimum count value.")
    parser.add_argument("--min_samples", required=True, help="Minimum number of samples.")
    parser.add_argument("--output_path_count", required=True, help="Path to save the filtered count data.")
    args = parser.parse_args()

    threshold_filter(
        count_file=args.count_file,
        metadata_file=args.metadata_file,
        min_count=int(args.min_count),
        min_samples=int(args.min_samples),
        output_path_count=args.output_path_count,
    )
