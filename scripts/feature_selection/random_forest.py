import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


def random_forest(
    count_file,
    metadata_file,
    n_estimators,
    max_depth,
    random_state,
    output_path_file,
    output_path_plot,
):
    """Run Random Forest and plot the feature importance."""
    print("Running Random Forest...")

    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)
    X = count_data
    y = metadata["condition"].map({"C": 0, "LC": 1})

    rf_model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
    )
    rf_model.fit(X, y.values.ravel())  # Ensure y is 1D

    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Sort and get top N indices

    # Extract feature names and importance values
    top_features = pd.DataFrame(
        {"Feature": X.columns[indices], "Importance": importances[indices]}
    )

    os.makedirs(os.path.dirname(output_path_file), exist_ok=True)
    top_features.to_csv(output_path_file, sep=";", index=False)

    # Plot the feature importance horizontally
    # reverse the order of the top features for better visualization
    top_features = top_features.iloc[::-1]
    plt.figure()
    plt.barh(range(top_features.shape[0]), top_features["Importance"], align="center")
    plt.yticks(range(top_features.shape[0]), top_features["Feature"])
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importance")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path_plot), exist_ok=True)
    plt.savefig(output_path_plot, dpi=300, format="png")

    print("Random Forest completed successfully!")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Random Forest on the count data.")
    parser.add_argument(
        "--count_file",
        type=str,
        help="The file path to the count data.",
        required=True,
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        help="The file path to the metadata.",
        required=True,
    )
    parser.add_argument(
        "--n_estimators",
        type=int,
        help="The number of trees in the forest.",
        required=True,
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        help="The maximum depth of the tree.",
        required=False,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        help="The random seed.",
        required=True,
    )
    parser.add_argument(
        "--output_path_file",
        type=str,
        help="The output path to save the feature importance file.",
        required=True,
    )
    parser.add_argument(
        "--output_path_plot",
        type=str,
        help="The output path to save the feature importance plot.",
        required=True,
    )

    args = parser.parse_args()

    random_forest(
        count_file=args.count_file,
        metadata_file=args.metadata_file,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        output_path_file=args.output_path_file,
        output_path_plot=args.output_path_plot,
    )
