# run_feature_selection.py
import argparse
import yaml
from feature_selector import (
    FeatureSelector,
)  # Assuming your class is in feature_selector.py
from data import Data  # Assuming Data class is in data.py


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to input data")
    parser.add_argument("--method", required=True, help="Feature selection method")
    parser.add_argument(
        "--top_n", required=True, type=int, help="Number of top features"
    )
    args = parser.parse_args()

    # Load the data (assuming Data class handles this)
    data = Data(args.data)  # Implement your Data class to load the data

    # Create the FeatureSelector instance
    selector = FeatureSelector(data)

    # Run the selected method with the specified parameters
    if args.method == "random_forest":
        selector.random_forest()
    elif args.method == "xgboost":
        selector.xgboost()
    elif args.method == "lasso":
        selector.lasso()
    elif args.method == "recursive_feature_elimination":
        selector.recursive_feature_elimination()
    else:
        raise ValueError(f"Unknown method {args.method}")


if __name__ == "__main__":
    main()
