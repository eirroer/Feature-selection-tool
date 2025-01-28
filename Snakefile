# Snakefile
configfile: "config/config.yml"

include: "rules/preprocessing.smk"

rule all:
    input:
        pre_filtered_data="data/pre_filtered_normalized_count_train_data.csv"