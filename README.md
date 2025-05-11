# Feature Selection and Classification Pipeline for RNA-seq Data
*A reproducible Snakemake pipeline for RNA‑seq feature selection, normalisation and supervised classification*

This tool takes raw RNA‑seq count tables plus sample metadata, applies configurable pre‑filters, normalises counts (CPM, TMM, DESeq2 or VST), fits scikit‑learn models (Random Forest and/or XGBoost) with cross‑validated hyper‑parameter tuning, and puts out performance reports and ranked feature importance – orchestrated by Snakemake and configured through a YAML file.

---

## Overview

| Stage | Highlights |
|-------|------------|
| **Input** | Accepts *train + test* matrices **or** splits a single dataset internally; strict sample‑ID matching. |
| **Pre‑filtering** | Threshold, variance, percentile, correlation filters|
| **Normalisation** | CPM, TMM, DESeq2, VST|
| **Current Models** | Random Forest & XGBoost |
| **Hyperparameter search** | `GridSearchCV` *or* `RandomizedSearchCV`
| **Outputs** | Confusion matrix, ROC curve, per‑fold CV scores, classification report, feature‑importance CSV/PNG, pickled best model. |
| **Reproducibility** | Conda env pinned (Python 3.12); optional offline zip via `conda‑pack`. |
| **Scale‑out** | Laptop: `--cores 8`; HPC: SLURM/SGE profiles – Snakemake handles the DAG. |

---


## 🔧 Install Conda Environment (`fs-tool.tar.gz`) - Linux only

This pipeline is designed to run in a secure or offline environment, such as the Tjeneste for Sensitive Data (TSD) or a high-performance computing (HPC) cluster without internet access. To ensure reproducibility and compatibility without requiring online package installation, the environment is distributed as a pre-packed Conda archive.

To activate the environment:

```bash
tar -xzf fs-tool.tar.gz
source fs-tool/bin/activate
```


After installing the tool and activating the conda enviroment the tool can be run by one of the following rules:


Local (not advised for bigger searches)
 ```bash
   Snakemake --cores [CORES] [Rule]
 ```

In HPC environments that use a job scheduler such as SLURM, the pipeline can be submitted as a Snakemake-managed cluster job. An example cluster submission command using SLURM is:

```bash
    snakemake --cluster "sbatch --mem-per-cpu=16G --cpus-per-task=8 account=your_account --time=12:00:00" --jobs 10
```

This setup allows Snakemake to submit each rule as a separate SLURM job, automatically handling dependencies and parallelization. 

the tool can be run by one of the following higher level rules:

| Rule | Comment |
|-------|------------|
| **plot_pca** | Runs a PCA of the input training data with options for normalization and a ``color_by`` option in the ``config`` file|
|**run_cross_validation**|Runs hyperparameter search with parameters specified in the ``config`` file |
|**run_prediciton**|Runs prediciton of models|
|**all**|Runs everthing all at once.|


### 📦 Core Packages

The following packages are included in the pre-built environment (`fs-tool.tar.gz`) and are required to run the pipeline:

| **Package**     | **Version** | **Purpose**                                               |
|------------------|-------------|------------------------------------------------------------|
| `python`         | 3.12.9      | Base programming language                                  |
| `conda`          | 22.11       | Environment and dependency management                      |
| `snakemake`      | 7.32.4      | Workflow orchestration                                     |
| `scikit-learn`   | 1.6.1       | Machine learning models and evaluation                     |
| `pandas`         | 2.2.3       | Data manipulation and preprocessing                        |
| `numpy`          | 2.2.3       | Numerical computations                                     |
| `xgboost`        | 2.1.4       | Gradient boosting model                                    |
| `joblib`         | 1.4.2       | Model serialization and parallelism                        |
| `pyyaml`         | 6.0.2       | YAML configuration file parsing                            |
| `pydeseq2`       | 0.5.0       | DESeq2-inspired normalization in Python                    |
| `rnanorm`        | 2.2.0       | RNA-seq normalization methods                              |
| `matplotlib`     | 3.10.0      | Plotting and visualization                                 |
| `seaborn`        | 0.13.2      | Statistical data visualization                             |

### Input Handling

The pipeline expects two input files:

- A **raw count matrix** (genes/features × samples)
- A **metadata file** with sample annotations, including class labels

Two input modes are supported:

- **Separate training and test sets** (provided as distinct files)
- **Single dataset**, automatically split using `train_test_split` parameters defined in `config/config.yml`

#### Required Input Formats

| **File**        | **Format**                                     | **Notes**                       |
|------------------|------------------------------------------------|----------------------------------|
| Count Matrix     | Rows = samples, Columns = features             | CSV, numeric only                |
| Metadata         | Rows = samples, Columns = descriptors          | Must include a `condition` label |


## Outputs
```
results/
├── model_CV_scores/
│ ├── random_forest_best_model_scores.csv
│ ├── xgboost_best_model_scores.csv
│
├── random_forest/
│ ├── random_forest_model.pkl
│ ├── random_forest_feature_importance.csv
│ ├── random_forest_feature_importance.png
│ ├── random_forest_gridsearch.pkl
│ └── random_forest_hyperparameters.csv
│
├── xgboost/
│ ├── xgboost_model.pkl
│ ├── xgboost_feature_importance.csv
│ ├── xgboost_feature_importance.png
│ ├── xgboost_gridsearch.pkl
│ ├── xgboost_hyperparameters.csv
│ └── xgboost_roc_curve.png
│
├── all_best_model_scores_CV.csv
├── all_roc_curves_CV.png
└── pca_plot.png

results_final_prediction/
├── all_models_classification_report.txt
├── all_models_confusion_matrix.png
└── all_models_roc_curve.png

data/
├── count_test_data.csv
├── count_train_data.csv
├── metadata_test_data.csv
├── metadata_train_data.csv
├── normalized_count_train_data.csv
└── pre_filtered_count_data.csv

config/
└── config.yml

logs/
└── *.log (Snakemake logs per rule)
```