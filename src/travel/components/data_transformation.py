import sys, os

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.travel.constant.training_pipeline import TARGET_COLUMN
from src.travel.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.travel.entity.config_entity import DataTransformationConfig
from src.travel.exception import TravelException
from src.travel.logger import logging


class DataTransformation:
    def __init__(self, 
                    data_validation_artifact: DataValidationArtifact, 
                        data_transformation_config: DataTransformationConfig
                            ):
        try:
            pass

        except Exception as e:
            raise TravelException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logging.info("reading csv and returning DataFrame ")
            return pd.read_csv(file_path)

        except Exception as e:
            raise SensorException(e, sys)


    @classmethod
    def get_data_transformer_object(cls)-> Pipeline:
        """
        :creates a pipeline object:
        :return: Pipeline object
        """
        try:
            pass
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_data_transformation(self, )->DataTransformationArtifact:
        try:
            pass
        except Exception as e:
            raise SensorException(e, sys)
            