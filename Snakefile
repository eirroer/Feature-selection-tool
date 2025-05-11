# Snakefile
configfile: "config/config.yml"

include: "rules/preprocessing.smk"
include: "rules/run_feature_selection.smk"
include: "rules/run_prediction.smk"

rule all:
    input:
        # Final predictions
        "results_final_prediction/all_models_classification_report.txt",
        "results_final_prediction/all_models_confusion_matrix.png",
        "results_final_prediction/all_models_roc_curve.png"

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

rule run_prediction:
    input:
        all_models_classification_report="results_final_prediction/all_models_classification_report.txt",
        all_models_confusion_matrix="results_final_prediction/all_models_confusion_matrix.png",
        all_models_roc_curve="results_final_prediction/all_models_roc_curve.png"
        

rule run_cross_validation:
    input:
        feature_importances=expand("results/{model}/{model}_feature_importance.csv", model=["random_forest", "xgboost"]),
        feature_importance_plots=expand("results/{model}/{model}_feature_importance.png", model=["random_forest", "xgboost"]),
        hyperparameters=expand("results/{model}/{model}_hyperparameters.csv", model=["random_forest", "xgboost"]),
        models=expand("results/{model}/{model}_model.pkl", model=["random_forest", "xgboost"]),
        all_model_scores="results/all_best_model_scores_CV.csv",
        all_roc_curves="results/all_roc_curves_CV.png",
