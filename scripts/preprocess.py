import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess(count_file, metadata_file, counts_holdout_test_set=None, metadata_holdout_test_set=None, output_path_counts, output_path_metadata):
    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0) 
    count_data = count_data.T
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    if counts_holdout_test_set and metadata_holdout_test_set:
        count_train_data = count_data
        metadata_train_data = metadata
        count_test_data = pd.read_csv(counts_holdout_test_set, delimiter=";", index_col=0, header=0)
        count_test_data = count_test_data.T
        metadata_test_data = pd.read_csv(metadata_holdout_test_set, delimiter=";", index_col=0, header=0)
    else:
        count_train_data, count_test_data, metadata_train_data, metadata_test_data = train_test_split(count_data, metadata, test_size=0.2, random_state=42)

    os.makedirs(os.path.dirname(output_path_counts), exist_ok=True)
    count_train_data.to_csv(output_path_counts, sep=";", index=True, header=True)

    os.makedirs(os.path.dirname(output_path_metadata), exist_ok=True)
    metadata_train_data.to_csv(output_path_metadata, sep=";", index=True, header=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess RNA count data.")
    parser.add_argument("--count_file", required=True, help="Path to the RNA count file.")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata file.")
    parser.add_argument("--counts_holdout_test_set", help="Path to the test RNA file, if provided.")
    parser.add_argument("--metadata_holdout_test_set", help="Path to the output file for the test RNA, if provided.")
    parser.add_argument("--count_train_data", required=True, help="Path to save the processed count training data.")
    parser.add_argument("--metadata_train_data", required=True, help="Path to save the processed metadata training data.")

    args = parser.parse_args()

    preprocess(args.count_file, args.metadata_file, args.counts_holdout_test_set, args.metadata_holdout_test_set, args.count_train_data, args.metadata_train_data)