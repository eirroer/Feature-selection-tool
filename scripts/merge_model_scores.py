import os
import pandas as pd

def merge_model_scores(
    random_forest_score_file: str,
    xgboost_score_file: str,
    all_models_score_file: str,
):
    """Merge model scores from two files."""
    # Load the model scores
    random_forest_scores = pd.read_csv(random_forest_score_file, delimiter=",", index_col=0, header=0)
    xgboost_scores = pd.read_csv(xgboost_score_file, delimiter=",", index_col=0, header=0)

    # Merge the dataframes
    all_models_scores = pd.concat([random_forest_scores, xgboost_scores])

    # Save the merged model scores
    os.makedirs(os.path.dirname(all_models_score_file), exist_ok=True)
    all_models_scores.to_csv(all_models_score_file, index=True, header=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Merge model scores.")
    parser.add_argument(
        "--random_forest_score_file",
        required=True,
        help="Path to the model scores file.",
    )
    parser.add_argument(
        "--xgboost_score_file", required=True, help="Path to the model scores file."
    )
    parser.add_argument(
        "--all_models_score_file",
        required=True,
        help="Path to save the merged model scores.",
    )

    args = parser.parse_args()

    merge_model_scores(
        random_forest_score_file=args.random_forest_score_file,
        xgboost_score_file=args.xgboost_score_file,
        all_models_score_file=args.all_models_score_file,
    )