# Snakefile
configfile: "config/config.yml"

include: "rules/preprocessing.smk"
# include: "rules/random_forest.smk"
# include: "rules/xgboost.smk"
include: "rules/run_feature_selection.smk"

rule all:
    input:
        all_RF_feature_importance=expand("results/random_forest/RF_{method}_feature_importance.csv", method=["gridsearch", "randomsearch"]),
        all_RF_feature_importance_plots=expand("plots/random_forest/RF_{method}_feature_importance.png", method=["gridsearch", "randomsearch"]),
        all_RF_models=expand("results/random_forest/RF_{method}_model.pkl", method=["gridsearch", "randomsearch"]),
        all_XGB_feature_importance=expand("results/xgboost/XGB_{method}_feature_importance.csv", method=["gridsearch", "randomsearch"]),
        all_XGB_feature_importance_plots=expand("plots/xgboost/XGB_{method}_feature_importance.png", method=["gridsearch", "randomsearch"]),
        all_XGB_models=expand("results/xgboost/XGB_{method}_model.pkl", method=["gridsearch", "randomsearch"]),
        all_XGB_hyperparameters=expand("results/xgboost/XGB_{method}_hyperparameters.csv", method=["gridsearch", "randomsearch"]),
        pca_plot="plots/pca_plot.png",