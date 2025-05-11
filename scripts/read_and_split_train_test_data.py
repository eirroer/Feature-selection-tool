import os
import pandas as pd
from sklearn.model_selection import train_test_split
from InputFormatChecker import InputFormatChecker


def read_and_split_train_test_data(
            count_file, 
            metadata_file, 
            count_train_output_path,
            metadata_train_output_path,
            count_test_output_path,
            metadata_test_output_path,
            test_size,
            random_state,
            counts_holdout_test_set=None,
            metadata_holdout_test_set=None,
            ):
    
    input_format_checker = InputFormatChecker(
        count_file, metadata_file, counts_holdout_test_set, metadata_holdout_test_set
    )
    input_format_checker.run_format_check()

    # Read the count data
    print(f"Reading count data from {count_file}")
    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0) 
    count_data = count_data.T
    count_data.index.name = "SampleID"

    # Read the metadata
    print(f"Reading metadata from {metadata_file}")
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)
    metadata.index.name = "SampleID"

    if counts_holdout_test_set and metadata_holdout_test_set:
        count_train_data = count_data
        metadata_train_data = metadata

        print(f"Reading test data from {counts_holdout_test_set}")
        count_test_data = pd.read_csv(counts_holdout_test_set, delimiter=";", index_col=0, header=0)
        count_test_data = count_test_data.T
        
        print(f"Reading test metadata from {metadata_holdout_test_set}")
        metadata_test_data = pd.read_csv(metadata_holdout_test_set, delimiter=";", index_col=0, header=0)
    else:
        # No holdout test set provided, split the data into train and test sets
        print(f"Splitting data into train and test sets with test size {test_size} and random state {random_state}")
        # Check if test_size is between 0 and 1
        if test_size <= 0 or test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        # Check if random_state is an integer
        if random_state is not None and not isinstance(random_state, int):
            raise ValueError("random_state must be an integer")

        count_train_data, count_test_data, metadata_train_data, metadata_test_data = train_test_split(count_data, metadata, test_size=test_size, random_state=random_state)


    print(f"Count train data shape: {count_train_data.shape}")
    print(f"Metadata train data shape: {metadata_train_data.shape}")
    print(f"Count test data shape: {count_test_data.shape}")
    print(f"Metadata test data shape: {metadata_test_data.shape}")


    os.makedirs(os.path.dirname(count_train_output_path), exist_ok=True)
    count_train_data.to_csv(count_train_output_path, sep=";", index=True, header=True)

    os.makedirs(os.path.dirname(metadata_train_output_path), exist_ok=True)
    metadata_train_data.to_csv(metadata_train_output_path, sep=";", index=True, header=True)

    os.makedirs(os.path.dirname(count_test_output_path), exist_ok=True)
    count_test_data.to_csv(count_test_output_path, sep=";", index=True, header=True)

    os.makedirs(os.path.dirname(metadata_test_output_path), exist_ok=True)
    metadata_test_data.to_csv(metadata_test_output_path, sep=";", index=True, header=True)


def validate_args(args):
    print("--Arguments received--")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="read_and_split_train_test_data RNA count data.")
    parser.add_argument("--count_file", required=True, help="Path to the RNA count file.")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata file.")
    parser.add_argument("--count_train_output_path", required=True, help="Path to save the processed count training data.")
    parser.add_argument("--metadata_train_output_path", required=True, help="Path to save the processed metadata training data.")
    parser.add_argument("--count_test_output_path", required=True, help="Path to save the processed count test data.")
    parser.add_argument("--metadata_test_output_path", required=True, help="Path to save the processed metadata test data.")
    parser.add_argument("--counts_holdout_test_set", help="Path to the test RNA file, if provided.")
    parser.add_argument("--metadata_holdout_test_set", help="Path to the output file for the test RNA, if provided.")
    parser.add_argument("--test_size", type=float, help="Size of the test set.")
    parser.add_argument("--random_state", type=int, help="Random state for the train-test split.")

    args = parser.parse_args()

    validate_args(args)

    read_and_split_train_test_data(
        count_file=args.count_file, 
        metadata_file=args.metadata_file, 
        counts_holdout_test_set=args.counts_holdout_test_set, 
        metadata_holdout_test_set=args.metadata_holdout_test_set, 
        count_test_output_path=args.count_test_output_path, 
        metadata_test_output_path=args.metadata_test_output_path,
        test_size=args.test_size,
        random_state=args.random_state,
        count_train_output_path=args.count_train_output_path, 
        metadata_train_output_path=args.metadata_train_output_path, 
    )
