# Snakefile
configfile: "config/config.yml"

include: "rules/preprocessing.smk"

rule all:
    input:
        normalized_count_train_data="data/normalized_count_train_data.csv"