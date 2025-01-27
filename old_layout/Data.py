import pandas as pd


class Data:
    """A class to represent the data object. Holds the count data, meta data and the targets."""

    count_data: pd.DataFrame = None
    meta_data: pd.DataFrame = None
    count_test_data: pd.DataFrame = None
    meta_test_data: pd.DataFrame = None

    targets: pd.Series = None

    def __init__(
        self,
        count_train_data: pd.DataFrame,
        meta_train_data: pd.DataFrame,
        count_test_data: pd.DataFrame = None,
        meta_test_data: pd.DataFrame = None,
        train_targets: pd.Series = None,
        test_targets: pd.Series = None,
    ):
        self.count_train_data = count_train_data
        self.meta_train_data = meta_train_data
        self.count_test_data = count_test_data
        self.meta_test_data = meta_test_data

        self._set_targets()

    def _set_targets(self):
        self.train_targets = self.meta_train_data["condition"].apply(lambda x: 0 if x == "C" else 1)
        self.test_targets = self.meta_test_data["condition"].apply(lambda x: 0 if x == "C" else 1)

    def __str__(self):
        return f"Data object with {self.counts.shape[0]} samples and {self.counts.shape[1]} features."
