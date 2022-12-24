import sys
from src.travel.exception import TravelException
from src.travel.logger import logging


from src.travel.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig
from src.travel.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

from src.travel.components.data_ingestion import DataIngestion
from src.travel.components.data_validation import DataValidation


class TrainPipeline:

    def __init__(self):
        
        self.training_pipeline_config = TrainingPipelineConfig()


    def start_data_ingestion(self)->DataIngestionArtifact:
        try:
            
            # creating data Ingestion config
            logging.info("Starting dataIngestion")
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)

            # passing dataIngestion config to DataIngestion class (of data_ingestion components)
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)

            # Initiating data ingestion to get data_ingestion_artifact
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(f"Data ingestion complete {data_ingestion_artifact}\n\n")

            return data_ingestion_artifact
        except Exception as e:
            raise TravelException(e, sys)


    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)


            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
                                                data_validation_config=data_validation_config)
            
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        
        except Exception as e:
            raise TravelException(e, sys)


    def run_pipeline(self):
            try:
                data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
                data_validation_artifact: DataValidationArtifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
                
            except Exception as e:
                raise TravelException(e, sys)