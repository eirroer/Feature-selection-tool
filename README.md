# Feature-selection-tool
 
## How to run the tool

Run with the following command
```
python prototype.py <count file> <metadata file>
python prototype.py ../../mock_data_lc\lc_mirna_counts.csv ../../mock_data_lc\lc_dataset.csv
```

## Input file format

For the tool to work properly the input files needs to sturctured in a spesific way given below

### Count file structure

| ID              | sample1 | sample10 | sample100 | sample1000 | sample1001 |
|-----------------|---------|----------|-----------|------------|------------|
| hsa-let-7a-2-3p |    0    |    0     |     0     |     0      |     0      |
| hsa-let-7a-3p   |    0    |    6     |    13     |     7      |    22      |
| hsa-let-7a-5p   |  15064  |   978    |   18281   |   3254     |   6308     |
| hsa-let-7b-3p   |    0    |    12    |     0     |     0      |     0      |

### Metadata file structure

| sex | condition | histology | cancertype | stage   | sample     |
|-----|-----------|-----------|------------|---------|------------|
| M   | C         | control   | control    | Control | sample367  |
| M   | C         | control   | control    | Control | sample45   |
| M   | C         | control   | control    | Control | sample254  |
| M   | C         | control   | control    | Control | sample779  |

## Setup of config file
To customize and select what feature selection methods to be run, you can make edits in the config file in the same folder named ```config.yaml```