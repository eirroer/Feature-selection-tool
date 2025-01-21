# Snakefile

configfile: "config.yaml"

rule all:
    input:
        expand("results/{method}_features.txt", method=[m["method"] for m in config["methods"]])

rule feature_selection:
    input:
        data="data/input_data.csv"
    output:
        features="results/{method}_features.txt"
    params:
        method="{method}",
        top_n="{top_n}"  # Assuming `top_n` is passed in the config
    script:
        "run_feature_selection.py"