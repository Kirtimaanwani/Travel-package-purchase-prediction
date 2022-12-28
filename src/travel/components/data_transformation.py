import sys, os

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, FunctionTransformer, OrdinalEncoder

from src.travel.constant.training_pipeline import TARGET_COLUMN, FEATURE_NAMES
from src.travel.constant.training_pipeline import SCHEMA_FILE_PATH
from src.travel.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from src.travel.entity.config_entity import DataTransformationConfig
from src.travel.exception import TravelException
from src.travel.logger import logging

from src.travel.utils.main_utils import read_yaml_file, save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, 
                    data_validation_artifact: DataValidationArtifact, 
                        data_transformation_config: DataTransformationConfig
                            ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self.feature_names_after_transformation = None
        except Exception as e:
            raise TravelException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            logging.info("reading csv and returning DataFrame ")
            return pd.read_csv(file_path)

        except Exception as e:
            raise TravelException(e, sys)

    @staticmethod
    def categorical_correction(X, y=None):
        df = X.copy()
        df = pd.DataFrame(df)
        df[3].replace("Fe Male", "Female", inplace=True)
        return df


    
    def get_data_transformer_object(cls, numerical_columns, categorical_columns)-> Pipeline:
        """
        :creates a pipeline object:
        :return: Pipeline object
        """
        try:
            # categorical_correction = FunctionT
            # simple_imputer_numerical_column = SimpleImputer(strategy="median")
            # simple_imputer_categorical_column = SimpleImputer(strategy="mode")
            # numeric_features = self._schema_config["numerical_columns"]
            # categorical_features = self._schema_config["categorical_columns"]

            # category_correction = FunctionTransformer(DataTransformation.categorical_correction)
            
            numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median'))
                                                        ,('scaler', RobustScaler())
                                                                        ])
            categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent'))
                                                        ,('category_correction', FunctionTransformer(DataTransformation.categorical_correction))
                                                        ,('encoder', OrdinalEncoder())
                                                                        ])
            preprocessor = ColumnTransformer(transformers=[('numeric', numeric_transformer, numerical_columns)
                                                          ,('categorical', categorical_transformer, categorical_columns)
                                                                        ]) 

            return preprocessor

        except Exception as e:
            raise TravelException(e, sys)


    def initiate_data_transformation(self, )->DataTransformationArtifact:
        try:
            logging.info("Reading data for train dataset")
            train_df = DataTransformation.read_data(
                                                    file_path= self.data_validation_artifact.valid_train_file_path
                                                        )
            self.all_column_names = train_df.columns
            logging.info("Reading data for test dataset")
            test_df = DataTransformation.read_data(
                                                    file_path= self.data_validation_artifact.valid_test_file_path
                                                        )

            logging.info("getting transformer object")    
            numeric_features = self._schema_config["numerical_columns"]
            numeric_features = train_df[numeric_features].select_dtypes(include='number').columns
            
            categorical_features = self._schema_config["categorical_columns"]
            categorical_features = list(filter(lambda item: item!=TARGET_COLUMN ,categorical_features))
            
            train_df[categorical_features] = train_df[categorical_features].astype("category")   
            categorical_features = train_df.select_dtypes(include="category").columns  

            # print(numeric_features, categorical_features)      
            # return                       
            preprocessor = self.get_data_transformer_object(numeric_features, categorical_features)

        # for training data frame
            logging.info("splitting training data as input_feature_train_df and target_feature_train_df")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]


        # for testng data frame
            logging.info("splitting test data as input_feature_test_df and target_feature_test_df")        

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]


        # Transforming according to preprocessor object
            logging.info("fitting of transformer object on input_feature_train_df")
            preprocessor_object = preprocessor.fit(input_feature_train_df)     

            logging.info("Transform on input_feature_train_df and input_feature_test_df")
            # for training data frame
            transformed_input_feature_train_df = preprocessor_object.transform(input_feature_train_df)
            # for test data frame
            transformed_input_feature_test_df = preprocessor_object.transform(input_feature_test_df)
            # print(preprocessor_object.named_steps['categorical'].get_feature_names())
            
            self.feature_names_after_transformation = list(preprocessor_object.transformers_[0][2]) + list(preprocessor_object.transformers_[1][2])
            

        # Using SMOTETomek 
            logging.info("Using SMOTETomek on train and test data")
            smt = SMOTETomek(sampling_strategy= "minority")
            # on training data frame
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                                                                                    transformed_input_feature_train_df, 
                                                                                        target_feature_train_df)
            # on test data frame
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                                                                                    transformed_input_feature_test_df, 
                                                                                        target_feature_test_df)

        # concatinating input feature and target feature
            # for training data frame
            train_arr = np.c_[
                                input_feature_train_final, 
                                np.array(target_feature_train_final)
                             ]
            # for testing data frame
            test_arr = np.c_[
                                input_feature_test_final, 
                                np.array(target_feature_test_final)
                             ]
            # #NOTE # Lets try to save this and see 
            # columns = FEATURE_NAMES.append(TARGET_COLUMN)
            # pp = pd.DataFrame(train_arr, columns)
            # os.mkdir(self.data_transformation_config.data_transformation_dir)
            # os.mkdir(self.data_transformation_config.data_trans_transformed_object_dir)
            # pp.to_csv(self.data_transformation_config.transformed_object_file_path)



        # saving numpy array
            # for training data frame
            logging.info("saving numpy array of train and test data")
            save_numpy_array_data(
                                    file_path=self.data_transformation_config.transformed_train_file_path, 
                                    array=train_arr
                                    )
            # for testing data frame
            save_numpy_array_data(
                                    file_path=self.data_transformation_config.transformed_test_file_path, 
                                    array=test_arr
                                    )

        # saving preprocessor object
            logging.info("saving preprocessor object")
        
            save_object(
                        file_path=self.data_transformation_config.transformed_object_file_path, 
                        obj=preprocessor_object)
                
            
            # Creating Data Trabsforation Artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
            logging.info(f"Data Transformation Artifact:[{data_transformation_artifact}]\n\n")
            return data_transformation_artifact




        except Exception as e:
            raise TravelException(e, sys)
            