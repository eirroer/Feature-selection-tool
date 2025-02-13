# Feature-selection-tool
This snakemake based pipeline tool is developed for doing feature selection on RNA counts data! It allows for plug-and-play analysis of the input data, with options for configuring the parameters used in the analysis.

## Install conda environment
Conda environment with the required dependecies can be set up using the environment.yaml file with the following command:

```
conda env create -f environment.yml
```
 
## Run the tool
Run with snakemake command.

to run all models (with all available cores):
```
snakemake --cores all
```


## Input file format
Input files need to be formatted correctly for the tool to work. 

### Count file structure
The count file(s) needs to have samples in the coloumns and features as the index. Index needs to be named 'ID'.

#### Example of required format

| ID              | sample1 | sample10 | sample100 | sample1000 | sample1001 |
|-----------------|---------|----------|-----------|------------|------------|
| hsa-let-7a-2-3p |    0    |    0     |     0     |     0      |     0      |
| hsa-let-7a-3p   |    0    |    6     |    13     |     7      |    22      |
| hsa-let-7a-5p   |  15064  |   978    |   18281   |   3254     |   6308     |
| hsa-let-7b-3p   |    0    |    12    |     0     |     0      |     0      |

### Metadata file structure
The metadata file(s) needs to have the samples as the index. Index needs to be named 'sample'. The only required column needs to named 'condition'. 

#### Example of required format
| sample    | sex | condition | histology | cancertype | stage   |
|-----------|-----|-----------|-----------|------------|---------|
| sample367 | M   | C         | control   | control    | Control |
| sample45  | M   | C         | control   | control    | Control |
| sample254 | M   | C         | control   | control    | Control |
| sample779 | M   | C         | control   | control    | Control |

## Setup of config file
The tool allows a list of configurable parameters to be tuned to the users liking. These parameters are found in the ```config.yaml```file. The files has a set of default parameters.
