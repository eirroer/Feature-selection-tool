
import typer
import time
import yaml
import warnings
import logging
from typing_extensions import Annotated, Optional
from rich.progress import Progress, SpinnerColumn, TextColumn

from Data import Data
from FeatureSelector import FeatureSelector
from InputFormatChecker import InputFormatChecker
from DataPreprocessor import DataPreprocessor

# Ignore warnings from imported TMM module
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="rnanorm.methods.between_sample")

logging.basicConfig(
            filename='application.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

class AnalysisTool:
    '''
    A class to represent the analysis tool. Holds the data and methods for analyzing the data.    
    '''
    data: Data = None
    feature_selector: FeatureSelector = None
    
    def run(
            self,
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
        """
        Run the analysis tool with the given files.

        Args:
            count_file (str): Path to the count data file.
            meta_file (str): Path to the meta data file.
            count_test_file (str, optional): Path to the test count data file. Defaults to None.
            meta_test_file (str, optional): Path to the test meta data file. Defaults to None.
        """
        # print(f"files given: \n-->{count_file} \n-->{meta_file}")


        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:

            # check if the files are correct are given in correct format
            check_format_task = progress.add_task(description="Checking file formats...", total=1)
            input_checker = InputFormatChecker(count_file, meta_file, count_test_file, meta_test_file)
            input_checker.run_format_check()
            progress.update(check_format_task, completed=1)

            read_config_task = progress.add_task(description="Reading configuration...", total=1)
            config_data = self.read_config_file(config_file="config.yaml")
            progress.update(read_config_task, completed=1)

            # read and preprocess the data
            preprocess_task = progress.add_task(description="Preprocessing data...", total=1)
            dataPreProcessor = DataPreprocessor(count_file_path=count_file,
                                                meta_file_path=meta_file,
                                                count_test_file=count_test_file,
                                                meta_test_file= meta_test_file,
                                                config_data=config_data
                                                )
            self.data = dataPreProcessor.preprocess()
            progress.update(preprocess_task, completed=1)

            # analyze the data
            analyze_task = progress.add_task(description="Analyzing data...", total=1)
            self.feature_selector = FeatureSelector(self.data)
            self.analyze(config_data=config_data)
            progress.update(analyze_task, completed=1)

            time.sleep(5)


    def read_config_file(self, config_file: str = "config.yaml"):
        """
        Read the config file and return the data.

        Args:
            config_file (str, optional): Path to the config file. Defaults to "config.yaml".

        Returns:
            dict: Data from the config file.
        """
        with open(config_file, 'r') as file:
            config_data = yaml.safe_load(file)
        return config_data

    def analyze(self, config_data: dict):
        """
        Analyze the data object using the methods specified in the config file.

        Args:
            config_data (dict): Data from the config file.
        """

        logging.info(f'Methods found in config file: {config_data['feature_selection_methods']}')
        
        # Run the selected methods
        for method_name in config_data['feature_selection_methods']:
            if hasattr(self.feature_selector, method_name):  # Check if the method exists
                method = getattr(self.feature_selector, method_name)  # Get the method in feature_selector object by given name found in the config file
                method()  # Call the method
            else:
                logging.info(f"Method '{method_name}' does not exist in the analysis tool.")


if __name__ == "__main__":
    analysis_tool = AnalysisTool()
    typer.run(analysis_tool.run)