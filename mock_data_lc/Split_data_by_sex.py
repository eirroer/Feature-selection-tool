import pandas as pd
import os

def split_data_by_sex():
    count_file = "lc_mirna_counts.csv"
    metadata_file = "lc_dataset.csv"

    count_data = pd.read_csv(count_file, delimiter=";", index_col=0, header=0).T
    # count_data.index.name = "SampleID"
    metadata = pd.read_csv(metadata_file, delimiter=";", index_col=0, header=0)

    print(count_data.head())
    print(metadata.head())
    print(metadata.columns)

    # split both datasets into male (M) and female (F) based on sex in metadata
    # get metdata of all samples where sex is F

    female_counts = count_data.loc[metadata.index[metadata['sex'] == 'F'].tolist()]
    male_counts = count_data.loc[metadata.index[metadata['sex'] == 'M'].tolist()]
    female_metadata = metadata[metadata['sex'] == 'F']
    male_metadata = metadata[metadata['sex'] == 'M']

    print(count_data.shape)
    print(female_counts.shape)
    print(male_counts.shape)
    print()
    print(metadata.shape)
    print(female_metadata.shape)
    print(male_metadata.shape)

    print(female_metadata.head())


    # Save the split data
    output_dir = "dataset_male_female"
    os.makedirs(output_dir, exist_ok=True)

    female_counts.to_csv(f'{output_dir}/{os.path.splitext(count_file)[0]}_female.csv', sep=",", index=True, header=True)
    female_metadata.to_csv(f'{output_dir}/{os.path.splitext(metadata_file)[0]}_female.csv', sep=",", index=True, header=True)
    male_counts.to_csv(f'{output_dir}/{os.path.splitext(count_file)[0]}_male.csv', sep=",", index=True, header=True)
    male_metadata.to_csv(f'{output_dir}/{os.path.splitext(metadata_file)[0]}_male.csv', sep=",", index=True, header=True)
    

if __name__ == "__main__":
    split_data_by_sex()


