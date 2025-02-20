import os
import pandas as pd
from PIL import Image

def merge_model_scores(
    random_forest_score_file: str,
    xgboost_score_file: str,
    all_models_score_file: str,
    random_forest_roc_curve: str,
    xgboost_roc_curve: str,
    output_path_roc_curve: str,
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

    # make a plot
    image_paths = [random_forest_roc_curve, xgboost_roc_curve]
    images = [Image.open(img) for img in image_paths]

    # Assume all images are the same height
    width, height = zip(*(img.size for img in images))
    total_width = sum(width)  # Sum all widths
    max_height = max(height)  # Keep the max height

    merged_image = Image.new("RGB", (total_width, max_height))

    # Paste images next to each other
    x_offset = 0
    for img in images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]  # Move offset by image width

    os.makedirs(os.path.dirname(output_path_roc_curve), exist_ok=True)
    merged_image.save(output_path_roc_curve)

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
    parser.add_argument(
        "--random_forest_roc_curve",
        required=True,
        help="Path to the ROC curve plot.",
    )
    parser.add_argument(
        "--xgboost_roc_curve", required=True, help="Path to the ROC curve plot."
    )

    parser.add_argument(
        "--output_path_roc_curve", required=True, help="Path to save the ROC curve plot.",
    )

    args = parser.parse_args()

    merge_model_scores(
        random_forest_score_file=args.random_forest_score_file,
        xgboost_score_file=args.xgboost_score_file,
        all_models_score_file=args.all_models_score_file,
        random_forest_roc_curve=args.random_forest_roc_curve,
        xgboost_roc_curve=args.xgboost_roc_curve,
        output_path_roc_curve=args.output_path_roc_curve,
    )
