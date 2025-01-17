import pandas as pd


class InputFormatChecker:
    count_file: str = None
    meta_file: str = None
    count_test_file: str = None
    meta_test_file: str = None
    count_data: pd.DataFrame = None
    meta_data: pd.DataFrame = None
    count_test_data: pd.DataFrame = None
    meta_test_data: pd.DataFrame = None

    def __init__(
                self,
                count_file: str,
                meta_file: str,
                count_test_file: str = None,
                meta_test_file: str = None,
                ):
        self.count_file = count_file
        self.meta_file = meta_file
        self.count_test_file = count_test_file
        self.meta_test_file = meta_test_file

        self.count_data = None 
        self.meta_data = None
        self.count_test_data = None
        self.meta_test_data = None

    def read_datafile(self, path:str):
        try:
            data = pd.read_csv(
                path, delimiter=";", index_col=0, header=0
            )
        except FileNotFoundError:
            raise ValueError(f"The file {path} does not exist. Please check the file and try again.")
        except pd.errors.ParserError:
            raise ValueError(f"The file {path} is not in the correct format. Please check the file and try again.")
        except pd.errors.EmptyDataError:
            raise ValueError(f"The file {path} is empty. Please check the file and try again.")
        except pd.errors.DtypeWarning:
            raise ValueError(f"The file {path} contains mixed data types. Please check the file and try again.")
        except pd.errors.ParserWarning:
            raise ValueError(f"The file {path} contains mixed data types. Please check the file and try again.")
        return data

    def read_mandatory_files(self):
        self.count_data = self.read_datafile(self.count_file)
        self.meta_data = self.read_datafile(self.meta_file)
        return True

    def read_optional_files(self):
        self.count_test_data = self.read_datafile(self.count_test_file)
        self.meta_test_data = self.read_datafile(self.meta_test_file)
        return True

    def is_both_optional_files_provided(self):
        if not self.count_test_file and not self.meta_test_file:
            print("No optional test files provided. The data will be split into training and test set.")
            return False
        if self.count_test_file and not self.meta_test_file:
            raise ValueError("If you provide a count test file, you must also provide a meta test file.")
        if self.meta_test_file and not self.count_test_file:
            raise ValueError("If you provide a meta test file, you must also provide a count test file.")
        return True

    def check_mandatory_data_not_empty(self):
        if self.count_data.empty:
            raise ValueError("The count data is empty. Please check the file and try again.")
        if self.meta_data.empty:
            raise ValueError("The meta data is empty. Please check the file and try again.")
        return True

    def check_mandatory_data_not_null(self):
        if self.count_data.isnull().values.any():
            raise ValueError("The count data contains null values. Please check the file and try again.")
        if self.meta_data.isnull().values.any():
            raise ValueError("The meta data contains null values. Please check the file and try again.")
        return True

    def check_mandatory_data_is_unique(self):
        if not self.count_data.index.is_unique:
            raise ValueError("The count data has duplicate IDs. Please make sure the IDs are unique.")
        if not self.meta_data.index.is_unique:
            raise ValueError("The meta data has duplicate IDs. Please make sure the IDs are unique.")
        return True

    def check_optional_data_is_unique(self):
        if not self.count_test_data.index.is_unique:
            raise ValueError("The count test data has duplicate IDs. Please make sure the IDs are unique.")
        if not self.meta_test_data.index.is_unique:
            raise ValueError("The meta test data has duplicate IDs. Please make sure the IDs are unique.")
        return True

    def check_optional_data_not_null(self):
        if self.count_test_data.isnull().values.any():
            raise ValueError("The count test data contains null values. Please check the file and try again.")
        if self.meta_test_data.isnull().values.any():
            raise ValueError("The meta test data contains null values. Please check the file and try again.")
        return True

    def check_optional_data_not_empty(self):
        if self.count_test_data.empty:
            raise ValueError("The count test data is empty. Please check the file and try again.")
        if self.meta_test_data.empty:
            raise ValueError("The meta test data is empty. Please check the file and try again.")
        return True

    def meta_data_contains_sample_index(self):
        if self.meta_data.index.name != "sample":
            raise ValueError("The meta data does not have an index column named 'sample'. Please add this coloumn to the meta data. It should contain the sample IDs.")
        return True

    def mandatory_files_contains_same_index(self):
        # sort the the meta data by sample to match the order of the count data
        self.meta_data = self.meta_data.sort_values(by="sample")

        # transpose the count data to have the samples as rows
        self.count_data = self.count_data.T

        # check if the IDs in the count data and meta data match
        if not self.count_data.index.equals(self.meta_data.index):
            raise ValueError(
                "The IDs in the count data and meta data do not match. Please make sure the samples in the count data and meta data has matching IDs. "
            )
        return True

    def optional_files_contains_same_index(self):
        # sort the the meta data by sample to match the order of the count data
        self.meta_test_data = self.meta_test_data.sort_values(by="sample")

        # transpose the count data to have the samples as rows
        self.count_test_data = self.count_test_data.T

        # check if the IDs in the count data and meta data match
        if not self.count_test_data.index.equals(self.meta_test_data.index):
            raise ValueError(
                "The IDs in the count test data and meta test data do not match. Please make sure the samples in the count test data and meta test data has matching IDs. "
            )
        return True

    def training_and_test_data_indexes_is_unique(self):
        # count_data and count_test_data should not have overlapping indexes
        count_overlapping_indexes = self.count_data.index.intersection(self.count_test_data.index)
        if not count_overlapping_indexes.empty:
            print(f"Overlapping count data indexes: {count_overlapping_indexes}")
            raise ValueError(
                "The count data and count test data should not have overlapping indexes. Please make sure the samples in the count data and count test data are unique."
            )

        # meta_data and meta_test_data should not have overlapping indexes
        meta_overlapping_indexes = self.meta_data.index.intersection(self.meta_test_data.index) 
        if not meta_overlapping_indexes.empty:
            print(f"Overlapping metadata indexes: {meta_overlapping_indexes}")
            raise ValueError(
                "The meta data and meta test data should not have overlapping indexes. Please make sure the samples in the meta data and meta test data are unique."
            )
        return True

    def run_format_check(self):
        self.read_mandatory_files()
        self.check_mandatory_data_not_empty()
        self.check_mandatory_data_not_null()
        self.check_mandatory_data_is_unique()
        self.meta_data_contains_sample_index()
        self.mandatory_files_contains_same_index()

        if self.is_both_optional_files_provided():
            self.read_optional_files()
            self.check_optional_data_not_empty()
            self.check_optional_data_not_null()
            self.check_optional_data_is_unique()
            self.optional_files_contains_same_index()
            self.training_and_test_data_indexes_is_unique()

        # TODO check if meta data contains the correct columns: condition, batch, etc.

        return True
