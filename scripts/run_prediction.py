import os
import pickle
import numpy as np
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


def run_prediciton(
    count_train_file: str,
    count_test_file: str,
    metadata_test_file: str,
    config_file: str,
    model_file: str,
    output_clasification_report: str,
    output_confusion_matrix: str,
    output_roc_curve: str,
    # output_pr_curve: str,
):
    """Run the prediction.

    Args:
        count_test_file (str): The file path to the test count file.
        metadata_test_file (str): The file path to the test metadata file.
        config_file (str): The file path to the configuration file.
        model_file (str): The file path to the model file.
        output_clasiification_report (str): The output path to save the classification report.
        output_confusion_matrix (str): The output path to save the confusion matrix.
        output_roc_curve (str): The output path to save the ROC curve.
        output_pr_curve (str): The output path to save the PR curve.
    """
    # Load the test data
    count_train_data = pd.read_csv(count_train_file, delimiter=";", index_col=0, header=0)
    count_data_test = pd.read_csv(count_test_file, delimiter=";", index_col=0, header=0)
    metadata_test = pd.read_csv(metadata_test_file, delimiter=";", index_col=0, header=0)

    # Load the configuration file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Load the model
    with open(model_file, "rb") as file:
        model = pickle.load(file)

    X_test = count_data_test[count_train_data.columns]
    y_true = metadata_test["condition"].map({"C": 0, "LC": 1})
    y_true = y_true.values.ravel()

    # Make the prediction
    y_pred = model.predict(X_test)

    # Save the classification report
    classification_report_df = pd.DataFrame(
        classification_report(y_true, y_pred, output_dict=True)
    ).T
    classification_report_df.to_csv(output_clasification_report)

    # Save the confusion matrix
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.unique(y_true),
        normalize="true",
    )
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control", "Cancer"])
    disp.plot(cmap=plt.cm.Blues)
    os.makedirs(os.path.dirname(output_confusion_matrix), exist_ok=True)
    plt.savefig(output_confusion_matrix)
    plt.close()

    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(y_true, model.predict_proba(X_test)[:, 1])

    # Create the plot
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker=".", label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.grid()

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_roc_curve), exist_ok=True)

    # Save the ROC curve plot
    plt.savefig(output_roc_curve)
    plt.close()  # Close the figure to free memory

# # Save the PR curve
# plot_precision_recall_curve(model, X_test, y_true)
# plt.savefig(output_pr_curve)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count_train_file", type=str, required=True)
    parser.add_argument("--count_test_file", type=str, required=True)
    parser.add_argument("--metadata_test_file", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--output_clasification_report", type=str, required=True)
    parser.add_argument("--output_confusion_matrix", type=str, required=True)
    parser.add_argument("--output_roc_curve", type=str, required=True)
    # parser.add_argument("--output_pr_curve", type=str, required=True)
    args = parser.parse_args()

    run_prediciton(
        count_train_file=args.count_train_file,
        count_test_file=args.count_test_file,
        metadata_test_file=args.metadata_test_file,
        config_file=args.config_file,
        model_file=args.model_file,
        output_clasification_report=args.output_clasification_report,
        output_confusion_matrix=args.output_confusion_matrix,
        output_roc_curve=args.output_roc_curve,
        # output_pr_curve=args.output_pr_curve,
    )
