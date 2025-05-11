import os
import pickle
import numpy as np
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score


def run_prediciton(
    count_train_file: str,
    count_test_file: str,
    metadata_test_file: str,
    config_file: str,
    model_file_random_forest: str,
    model_file_xgboost: str,
    training_scores_file: str,
    output_clasification_report: str,
    output_confusion_matrix: str,
    output_roc_curve: str,
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
    """

    count_train_data = pd.read_csv(count_train_file, delimiter=";", index_col=0, header=0)
    count_data_test = pd.read_csv(count_test_file, delimiter=";", index_col=0, header=0)
    metadata_test = pd.read_csv(metadata_test_file, delimiter=";", index_col=0, header=0)
    training_scores = pd.read_csv(training_scores_file, delimiter=",", index_col=0, header=0)

    # Load the configuration file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Define model file paths 
    model_files = {
        "Random Forest": model_file_random_forest,
        "XGBoost": model_file_xgboost,
    }

    # Load models into a dictionary
    models = {}
    for name, model_file in model_files.items():
        with open(model_file, "rb") as file:
            models[name] = pickle.load(file)

    # Make sure the same features as training data
    X_test = count_data_test[count_train_data.columns]

    # Read the condition labels from the config file
    condition_labels = {
    "case": config["condition_labels"]["case"],
    "control": config["condition_labels"]["control"]
    }

    # Validate that each condition label is present in the metadata
    missing = {k: v for k, v in condition_labels.items() if v not in metadata_test["condition"].unique()}
    if missing:
        raise ValueError(f"The following condition labels are missing from metadata_test['condition']: {missing}")

    # Get the true labels
    y_true = metadata_test["condition"].map({condition_labels["case"]: 1, condition_labels["control"]: 0})
    y_true = y_true.values.ravel()

    # Create the classification report
    os.makedirs(os.path.dirname(output_clasification_report), exist_ok=True)
    with open(output_clasification_report, "w") as f:
        
        training_scores_rn = training_scores.copy()

        columns_to_merge = [
            "Balanced Accuracy",
            "Accuracy",
            "AUC",
            "Precision",
            "Recall",
            "F1",
        ]

        for col in columns_to_merge:
            # Construct the column names for mean train and CV scores
            train_col = f"mean_train_{col.lower().replace('-', '_').replace(' ', '_')}"
            cv_col = f"mean_CV_{col.lower().replace('-', '_').replace(' ', '_')}"

            # Handle special case for AUC
            if col == "AUC" and "mean_train_roc_auc" in training_scores_rn.columns:
                train_col = "mean_train_roc_auc"
                cv_col = "mean_CV_roc_auc"

            # Check if the columns exist in the DataFrame and merge them
            if train_col in training_scores_rn.columns and cv_col in training_scores_rn.columns:
                training_scores_rn[col] = training_scores_rn[[train_col, cv_col]].mean(axis=1)
                training_scores_rn.drop(columns=[train_col, cv_col], inplace=True)


        # Merge mean and std. dev. values into a single column
        for metric in ['Balanced Accuracy', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1-Score']:
            std_col = f"std_CV_{'_'.join(metric.lower().split())}"  # Construct std. dev. column name
            if std_col in training_scores_rn.columns:
                training_scores_rn[metric] = training_scores_rn[metric].map("{:.4f}".format) + " +/- " + training_scores_rn[std_col].map("{:.4f}".format)

        # Drop the std. dev. columns
        training_scores_rn.drop(columns=[col for col in training_scores.columns if col.startswith(("std_CV_", "std_train_"))], inplace=True)

        # run the predicitons
        all_scores = training_scores_rn.copy()
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]  # Get predicted probabilities for ROC AUC

            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            predition_scores = pd.DataFrame(
                {
                    "Balanced Accuracy": [balanced_accuracy],
                    "Accuracy": [accuracy],
                    "AUC": [roc_auc],
                    "Precision": [precision],
                    "Recall": [recall],
                    "F1": [f1],
                },
                index=[f"{name} (test)"],
            ).round(4)

            f.write(f"{'='*30} {name.upper()} {'='*30}\n")

            # merge the prediction scores with the training scores
            all_scores = pd.concat([all_scores, predition_scores])
            model_scores = all_scores.copy()
            model_scores = model_scores.loc[[f"{name} (train)", f"{name} (CV)", f"{name} (test)"]]

            # remove nan for pretty printing
            model_scores = model_scores.replace(r'\s*\+/-\s*nan', '', regex=True)
            model_scores = model_scores.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            # Print the scores
            f.write(model_scores.to_string(col_space=25))
            f.write("\n\n")

            # Add Standard classification report
            report = classification_report(
                y_true, y_pred, target_names=["Control", "Case"]
            )

            # Print results
            f.write(f"Classification Report for {name}:\n")
            f.write(report)
            f.write("\n")

    # Create the confusion matrix plot
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    fig.suptitle("Confusion Matrix")

    # Iterate over models and plot confusion matrices
    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False,
            xticklabels=["Control", "Case"],
            yticklabels=["Control", "Case"],
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    # Save the confusion matrix plot
    os.makedirs(os.path.dirname(output_confusion_matrix), exist_ok=True)
    plt.savefig(output_confusion_matrix, bbox_inches='tight', dpi=300)
    plt.close()

    # Create the ROC curve plot
    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y_true, model.predict_proba(X_test)[:, 1])
        auc_score = roc_auc_score(y_true, model.predict_proba(X_test)[:, 1])
        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {auc_score:.2f})",
            linewidth=2,
        )
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.suptitle('ROC Curve', fontsize=16)
    plt.title('Final prediction results')
    plt.legend(loc='lower right')

    # Save the ROC curve plot
    os.makedirs(os.path.dirname(output_roc_curve), exist_ok=True)
    plt.savefig(output_roc_curve, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--count_train_file", type=str, required=True)
    parser.add_argument("--count_test_file", type=str, required=True)
    parser.add_argument("--metadata_test_file", type=str, required=True)
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--model_file_random_forest", type=str, required=True)
    parser.add_argument("--model_file_xgboost", type=str, required=True)
    parser.add_argument("--training_scores_file", type=str, required=True)
    parser.add_argument("--output_clasification_report", type=str, required=True)
    parser.add_argument("--output_confusion_matrix", type=str, required=True)
    parser.add_argument("--output_roc_curve", type=str, required=True)
    args = parser.parse_args()

    run_prediciton(
        count_train_file=args.count_train_file,
        count_test_file=args.count_test_file,
        metadata_test_file=args.metadata_test_file,
        config_file=args.config_file,
        model_file_random_forest=args.model_file_random_forest,
        model_file_xgboost=args.model_file_xgboost,
        training_scores_file=args.training_scores_file,
        output_clasification_report=args.output_clasification_report,
        output_confusion_matrix=args.output_confusion_matrix,
        output_roc_curve=args.output_roc_curve,
    )
