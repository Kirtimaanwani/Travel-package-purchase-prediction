import os
from datetime import datetime
from src.travel.constant import training_pipeline
from src.travel.logger import logging


class TrainingPipelineConfig:

    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR, timestamp)
        self.timestamp: str = timestamp


class DataIngestionConfig:
    
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        logging.info("Creating DataIngestionConfig")

        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_INGESTION_DIR_NAME
        )

        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
            training_pipeline.FILE_NAME
        )
        
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME

        self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME
    
        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TRAIN_FILE_NAME
        )

        self.test_file_path: str = os.path.join(
            self.data_ingestion_dir,
            training_pipeline.DATA_INGESTION_INGESTED_DIR,
            training_pipeline.TEST_FILE_NAME
        )

        self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO


class DataValidationConfig:
    
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        logging.info("Creating DataValidationConfig")

        self.data_validation_dir: str = os.path.join( training_pipeline_config.artifact_dir, 
                                                    training_pipeline.DATA_VALIDATION_DIR_NAME)
        
        self.valid_data_dir: str = os.path.join(self.data_validation_dir,
                                                    training_pipeline.DATA_VALIDATION_VALID_DIR)

        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, 
                                                    training_pipeline.DATA_VALIDATION_INVALID_DIR)
        
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, 
                                                    training_pipeline.TRAIN_FILE_NAME)

        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, 
                                                    training_pipeline.TEST_FILE_NAME)

        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, 
                                                    training_pipeline.TRAIN_FILE_NAME)
        
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, 
                                                    training_pipeline.TEST_FILE_NAME)

        # self.drift_report_file_path: str = os.path.join(
        #     self.data_validation_dir,
        #         training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
        #             training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
        # )


class DataTransformationConfig:
    
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        logging.info("Creating DataTransformationConfig")

        self.data_transformation_dir: str = os.path.join(
                                                    training_pipeline_config.artifact_dir, 
                                                        training_pipeline.DATA_TRANS_DIR_NAME
                                                            )

        self.transformed_train_file_path: str = os.path.join(
                                                        self.data_transformation_dir,
                                                            training_pipeline.DATA_TRANS_TRANSFORMED_DATA_DIR,
                                                                training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy")
                                                                    )

        self.transformed_test_file_path: str = os.path.join(
                                                        self.data_transformation_dir,
                                                            training_pipeline.DATA_TRANS_TRANSFORMED_DATA_DIR,
                                                                training_pipeline.TEST_FILE_NAME.replace("csv", "npy")
                                                                    )
        self.data_trans_transformed_object_dir:str = os.path.join(
                                                self.data_transformation_dir,
                                                    training_pipeline.DATA_TRANS_TRANSFORMED_OBJECT_DIR
                                                            )
        self.transformed_object_file_path: str = os.path.join(
                                                        self.data_transformation_dir,
                                                            training_pipeline.DATA_TRANS_TRANSFORMED_OBJECT_DIR,
                                                                training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
                                                                    )

                                         