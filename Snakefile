# Snakefile
configfile: "config/config.yml"

include: "rules/preprocessing.smk"
# include: "rules/random_forest.smk"
# include: "rules/xgboost.smk"
include: "rules/run_feature_selection.smk"
include: "rules/run_prediction.smk"

rule all:
    input:
        expand("results/{model}/{model}_feature_importance.csv", model=["random_forest", "xgboost"]),
        expand("results/{model}/{model}_feature_importance.png", model=["random_forest", "xgboost"]),
        expand("results/{model}/{model}_hyperparameters.csv", model=["random_forest", "xgboost"]),
        expand("results/{model}/{model}_model.pkl", model=["random_forest", "xgboost"]),
        "results/pca_plot.png",
