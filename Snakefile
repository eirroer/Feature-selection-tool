# Snakefile
configfile: "config/config.yml"

include: "rules/data_exploration.smk"
include: "rules/preprocessing.smk"
include: "rules/run_feature_selection.smk"
include: "rules/run_prediction.smk"

rule all:
    input:
        # Final predictions - random forest
        "results_final_prediction/random_forest/random_forest_classification_report.csv",
        "results_final_prediction/random_forest/random_forest_confusion_matrix.png",
        "results_final_prediction/random_forest/random_forest_roc_curve.png",

        # Final predictions - xgboost
        "results_final_prediction/xgboost_classification_report.csv",
        "results_final_prediction/xgboost_confusion_matrix.png",
        "results_final_prediction/xgboost_roc_curve.png",

        # Cross-validation outputs
        "results/random_forest/random_forest_feature_importance.csv",
        "results/xgboost/xgboost_feature_importance.csv",
        "results/random_forest/random_forest_feature_importance.png",
        "results/xgboost/xgboost_feature_importance.png",
        "results/random_forest/random_forest_hyperparameters.csv",
        "results/xgboost/xgboost_hyperparameters.csv",
        "results/random_forest/random_forest_model.pkl",
        "results/xgboost/xgboost_model.pkl",
        "results/all_best_model_scores_CV.csv",
        "results/all_roc_curves_CV.png",

        # PCA plot
        "results/pca_plot.png",

rule run_prediciton:
    input:
        # Final predictions - random forest
        "results_final_prediction/random_forest/random_forest_classification_report.csv",
        "results_final_prediction/random_forest/random_forest_confusion_matrix.png",
        "results_final_prediction/random_forest/random_forest_roc_curve.png",

        # Final predictions - xgboost
        "results_final_prediction/xgboost_classification_report.csv",
        "results_final_prediction/xgboost_confusion_matrix.png",
        "results_final_prediction/xgboost_roc_curve.png",

rule run_cross_validation:
    input:
        feature_importances=expand("results/{model}/{model}_feature_importance.csv", model=["random_forest", "xgboost"]),
        feature_importance_plots=expand("results/{model}/{model}_feature_importance.png", model=["random_forest", "xgboost"]),
        hyperparameters=expand("results/{model}/{model}_hyperparameters.csv", model=["random_forest", "xgboost"]),
        models=expand("results/{model}/{model}_model.pkl", model=["random_forest", "xgboost"]),
        all_model_scores="results/all_best_model_scores_CV.csv",
        all_roc_curves="results/all_roc_curves_CV.png",
        # pca_plot="results/pca_plot.png",


# rule get_final_results:
#     input:
#         classification_report="results_final_prediction/classification_report.txt",
#         confusion_matrix="results_final_prediction/confusion_matrix.png",
#         roc_curve="results_final_prediction/roc_curve_all.png"
#     output:
#         "results_final_prediction/classification_report.txt",
#         "results_final_prediction/confusion_matrix.png",
#         "results_final_prediction/roc_curve_all.png",


# rule run_cross_validation:
#     input:
#         expand("results/{model}/{model}_feature_importance.csv", model=["random_forest", "xgboost"]),
#         expand("results/{model}/{model}_feature_importance.png", model=["random_forest", "xgboost"]),
#         expand("results/{model}/{model}_hyperparameters.csv", model=["random_forest", "xgboost"]),
#         expand("results/{model}/{model}_model.pkl", model=["random_forest", "xgboost"]),
#         merged_model_scores="results/all_best_model_scores_CV.csv",
#         merged_roc_curves="results/all_roc_curves_CV.png",
#         pca_plot="results/pca_plot.png"
#     output:
#         cross_val_scores="results/all_best_model_scores_CV.csv",
#         all_roc_curves="results/all_roc_curves_CV.png",




# rule run_prediction:
#     input:
#         threshold_filter_data="data/threshold_filtered_count_train_data.csv",
#         count_test_data="data/count_test_data.csv",
#         metadata_test_data="data/metadata_test_data.csv",
#         config_file="config/config.yml",
#         random_forest_model="results/random_forest/random_forest_model.pkl",
#         xgboost_model="results/xgboost/xgboost_model.pkl",
#         cross_val_scores="results/all_best_model_scores_CV.csv"
#     output:
#         classification_report="results_final_prediction/classification_report.txt",
#         confusion_matrix="results_final_prediction/confusion_matrix.png",
#         roc_curve="results_final_prediction/roc_curve_all.png"
#     log:
#         "logs/run_prediction.log"
#     params:
#         script="scripts/run_prediction.py",
#         config_file="config/config.yml"
#     run:
#         # Prepare the command to run the external Python script
#         cmd = [
#             "python", "{params.script}",
#             "--count_train_file", input.threshold_filter_data,
#             "--count_test_file", input.count_test_data,
#             "--metadata_test_file", input.metadata_test_data,
#             "--config_file", input.config_file,
#             "--model_file_random_forest", input.random_forest_model,
#             "--model_file_xgboost", input.xgboost_model,
#             "--training_scores_file", input.traning_scores,
#             "--output_clasification_report", output.classification_report,
#             "--output_confusion_matrix", output.confusion_matrix,
#             "--output_roc_curve", output.roc_curve
#         ]
#         # Log the command
#         shell_cmd = " ".join(cmd)
#         print(f"Running command: {shell_cmd}")
#         # Run the command and redirect stdout and stderr to the log file
#         # shell(shell_cmd + " > {log} 2>&1")
#         shell(shell_cmd)


# rule all:
#     input:
#         classification_report="results_final_prediction/classification_report.txt",
#         confusion_matrix="results_final_prediction/confusion_matrix.png",
#         roc_curve="results_final_prediction/roc_curve_all.png"