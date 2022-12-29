import sys
from src.travel.exception import TravelException
from src.travel.logger import logging


from src.travel.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig, ModelEvaluationConfig, ModelPusherConfig
from src.travel.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataTransformationArtifact, ModelTrainerArtifact, ModelEvaluationArtifact, ModelPusherArtifact

from src.travel.components.data_ingestion import DataIngestion
from src.travel.components.data_validation import DataValidation
from src.travel.components.data_transformation import DataTransformation
from src.travel.components.model_trainer import ModelTrainer
from src.travel.components.model_evaluation import ModelEvaluation
from src.travel.components.model_pusher import ModelPusher


class TrainPipeline:
    is_pipeline_running=True
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


    def start_data_transformation(self, data_validation_artifact:DataValidationArtifact)->DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)

            data_transformation = DataTransformation(
                                                    data_validation_artifact=data_validation_artifact,
                                                        data_transformation_config=data_transformation_config
                                                            )
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        
        except Exception as e:
            raise TravelException(e, sys)


    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            logging.info("Starting model trainer")
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
        except Exception as e:
            raise TravelException(e, sys)    


    def start_model_evaluation(self,data_validation_artifact:DataValidationArtifact,
                                model_trainer_artifact:ModelTrainerArtifact,
                                )->ModelEvaluationArtifact:
        try:
            logging.info("starting model evaluation")
            model_eval_config = ModelEvaluationConfig(self.training_pipeline_config)
            model_eval = ModelEvaluation(model_eval_config, data_validation_artifact, model_trainer_artifact)
            model_eval_artifact = model_eval.initiate_model_evaluation()
            logging.info("returning model_eval_artifact\n\n")
            return model_eval_artifact
        except  Exception as e:
            raise  TravelException(e,sys)


    def start_model_pusher(self,model_eval_artifact:ModelEvaluationArtifact):
        try:
            logging.info("starting model pusher")
            model_pusher_config = ModelPusherConfig(training_pipeline_config=self.training_pipeline_config)
            model_pusher = ModelPusher(model_pusher_config, model_eval_artifact)
            model_pusher_artifact = model_pusher.initiate_model_pusher()
            logging.info("returning model_pusher_artifact\n\n")
            return model_pusher_artifact
        except  Exception as e:
            raise  TravelException(e,sys)


    def run_pipeline(self):
            try:
                data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()
                data_validation_artifact: DataValidationArtifact = self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
                data_transformation_artifact: DataTransformationArtifact = self.start_data_transformation(data_validation_artifact=data_validation_artifact)
                model_trainer_artifact: ModelTrainerArtifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)           
                model_eval_artifact: ModelEvaluationArtifact = self.start_model_evaluation(data_validation_artifact=data_validation_artifact, model_trainer_artifact=model_trainer_artifact)
                if not model_eval_artifact.is_model_accepted:
                    logging.info("Trained model is not better than the best model which is already exists")
                    raise Exception("Trained model is not better than the best model which is already exists, either add more data or do some better model tune or better split the data or etc.")

                model_pusher_artifact = self.start_model_pusher(model_eval_artifact)
            
                TrainPipeline.is_pipeline_running=False

            except Exception as e:
                raise TravelException(e, sys)