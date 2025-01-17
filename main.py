import typer
from typing_extensions import Annotated, Optional

from InputFormatChecker import InputFormatChecker
from DataPreprocessor import DataPreprocessor
from AnalysisTool import AnalysisTool


def run(
    count_file: Annotated[str, typer.Argument()],
    meta_file: Annotated[str, typer.Argument()],
    count_test_file: Annotated[
        Optional[str],
        typer.Option(
            help="Optional holdout test of the RNA counts, if not provided count_file will be split into training and test set."
        ),
    ] = None,
    meta_test_file: Annotated[
        Optional[str],
        typer.Option(
            help="Optional holdout test of the metadata, if not provided meta_file will be split into training and test set."
        ),
    ] = None,
):
    print(f"files given: \n-->{count_file} \n-->{meta_file}")

    # check if the files are correct are given in correct format
    input_checker = InputFormatChecker(count_file, meta_file, count_test_file, meta_test_file)
    input_checker.run_format_check()

    # read and preprocess the data
    data = DataPreprocessor.preprocess(count_file, meta_file, count_test_file, meta_test_file)

    # create an analysis tool object
    analysis_tool = AnalysisTool(data)
    analysis_tool.analyze()


if __name__ == "__main__":
    typer.run(run)
