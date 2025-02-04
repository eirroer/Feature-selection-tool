# Snakefile
configfile: "config/config.yml"

include: "rules/preprocessing.smk"
include: "rules/feature_selection.smk"

rule all:
    input:
        "results/random_forest_feature_importance.csv",
        "plots/pca_plot.png",
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv"