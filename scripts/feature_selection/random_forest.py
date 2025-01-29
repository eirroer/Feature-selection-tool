import os
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def random_forest(
    count_file,
    metadata_file,
    config_file,
    output_path_file,
    output_path_plot,
):
    """Run Random Forest and plot the feature importance."""
    print("Running Random Forest...")

    # Load configuration file
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)

    # Extract parameters from the configuration file
    param_grid = config_data["feature_selection"]["random_forest"]["param_grid"]
    print(f"Parameters: {param_grid}")

    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0)
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    X = count_data
    y = metadata["condition"].map({"C": 0, "LC": 1})
    y = y.values.ravel()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,  # 5-fold cross-validation
        scoring="accuracy",  # Metric to optimize
        n_jobs=-1,  # Use all CPUs
        verbose=2,
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Evaluate on validation data
    best_model = grid_search.best_estimator_
    test_accuracy = best_model.score(X_val, y_val)
    print("Test Accuracy:", test_accuracy)

    importances = best_model.feature_importances_
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
        "--config_file",
        type=str,
        help="The file path to the configuration file.",
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
        config_file=args.config_file,
        output_path_file=args.output_path_file,
        output_path_plot=args.output_path_plot,
    )
