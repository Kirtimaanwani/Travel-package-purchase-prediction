import os, sys
import pandas as pd

from src.travel.exception import TravelException
from src.travel.logger import logging

from src.travel.constant.training_pipeline import SCHEMA_FILE_PATH
from src.travel.entity.config_entity import DataValidationConfig
from src.travel.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

from src.travel.utils.main_utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp



class DataValidation:

    def __init__(
                self,
                data_ingestion_artifact:DataIngestionArtifact,
                data_validation_config:DataValidationConfig):  
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)  # underscore for schema_config is used for protected variable
        except Exception as e:
            raise TravelException(e,sys)
    

    @staticmethod
    def read_data(file_path)->pd.DataFrame:

        f"""
        Reading data from [{file_path}] and returning its DataFrame
        """
        try:
            logging.info(f"Reading data from [{file_path}] and returning its DataFrame")
            return pd.read_csv(file_path)
        except Exception as e:
            raise TravelException(e,sys)
    

    def validate_number_of_columns(self, dataframe: pd.DataFrame)->bool:
        try:
            logging.info("validating number of columns")
            number_of_columns = len(self._schema_config["columns"])

            if len(dataframe.columns) == number_of_columns:
                return True
            return False
        except Exception as e:
            raise TravelException(e,sys)
    

    def is_numerical_column_exists(self, dataframe: pd.DataFrame)->bool:
        try:
            logging.info("validating Numerical columns")
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns

            numerical_column_present = True
            missing_numerical_columns = []
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present = False
                    missing_numerical_columns.append(num_column)
            logging.info(f"Missing numerical columnes [{missing_numerical_columns}]")
            return numerical_column_present

        except Exception as e:
            raise TravelException(e,sys)


    def is_categorical_column_exists(self, dataframe: pd.DataFrame)->bool:
        try:
            logging.info("validating Categorical columns")
            categorical_columns = self._schema_config["categorical_columns"]
            dataframe_columns = dataframe.columns

            categorical_column_present = True
            missing_categorical_columns = []
            for cat_column in categorical_columns:
                if cat_column not in dataframe_columns:
                    categorical_column_present = False
                    missing_categorical_columns.append(cat_column)
            logging.info(f"Missing categorical columnes [{missing_categorical_columns}]")
            return categorical_column_present

        except Exception as e:
            raise TravelException(e,sys)
    



    def detect_data_drift(self,base_df,current_df,threshold:float=0.05)->bool:
        """
        Detecting drift with the help of Kolmogorov-Smirnov Test or KS Test
        """
        try:
            status=True
            report ={}
            for column in base_df.columns:
                d1 = base_df[column]
                d2  = current_df[column]
                is_same_dist = ks_2samp(d1,d2)
                if threshold <= float(is_same_dist.pvalue):
                    is_found=False
                else:
                    is_found = True 
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            # Creating directory
            dir_path = os.path.dirname(drift_report_file_path)
            write_yaml_file(file_path=drift_report_file_path, content=report)


            return status
        
        except Exception as e:
            raise TravelException(e,sys)


    def initiate_data_validation(self)-> DataValidationArtifact:
        """
        Initiating data validation
        """
        try:
            logging.info("Initiating data validation")
            logging.info("Creating folder for data validation")
            os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
            os.makedirs(self.data_validation_config.invalid_data_dir, exist_ok=True)


            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            
            # Reading data from Train and Test file Locations
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)

            # Validate Number of Columns
            error_message = ""
            status = self.validate_number_of_columns(dataframe=train_dataframe)
            if not status:
                error_message = f"{error_message}, [Train DataFrame] does not contain all columns"
            
            status = self.validate_number_of_columns(dataframe=test_dataframe)
            if not status:
                error_message = f"{error_message}, [Test DataFrame] does not contain all columns"

            # Validate Numerical Columns
            status = self.is_numerical_column_exists(dataframe= train_dataframe)
            if not status:
                error_message = f"{error_message}, [Train DataFrame] does not contain all Numerical columns, check in logs for which columns are missing"

            status = self.is_numerical_column_exists(dataframe= test_dataframe)
            if not status:
                error_message = f"{error_message} [Test DataFrame] does not contain all Numerical columns, check in logs for which columns are missing"

            # Validate Categorical Columns
            status = self.is_categorical_column_exists(dataframe= train_dataframe)
            if not status:
                error_message = f"""{error_message}, [Train DataFrame] does not contain all Categorical columns,
                                                                 check in logs for which columns are missing """

            status = self.is_categorical_column_exists(dataframe= test_dataframe)
            if not status:
                error_message = f"""{error_message}, [Train DataFrame] does not contain all Categorical columns,
                                                                 check in logs for which columns are missing """
            
            # Raising error if there is any validation error
            if len(error_message) > 0 :
                logging.info("Data Validation error occured , saving files to invalid folder")
                train_dataframe.to_csv(self.data_validation_config.invalid_train_file_path)
                test_dataframe.to_csv(self.data_validation_config.invalid_test_file_path)
                raise Exception(error_message)

            # Checking Data DRIFT
            # self.detect_data_drift(base_df=train_dataframe, current_df=test_dataframe)
            # NOTE: 1Since data is still not transformed so we cannot check distribution of train and test so after transformation this step should be taken care of.
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path)
            # creating data valiidation artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path = self.data_validation_config.valid_train_file_path,
                valid_test_file_path = self.data_validation_config.valid_test_file_path,
                invalid_train_file_path = self.data_validation_config.invalid_train_file_path,
                invalid_test_file_path = self.data_validation_config.invalid_train_file_path,
                drift_report_file_path = None,
            )
            
            logging.info(f"Got data validation Artifact as [{data_validation_artifact}]\n\n")
            
            # return artifacts that are created 
            return data_validation_artifact 
        except Exception as e:
            raise TravelException(e, sys)