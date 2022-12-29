import os,sys
from src.travel.exception import TravelException
from src.travel.logger import logging
from src.travel.entity.artifact_entity import ModelPusherArtifact,ModelTrainerArtifact,ModelEvaluationArtifact
from src.travel.entity.config_entity import ModelEvaluationConfig,ModelPusherConfig
from src.travel.ml.metric.classification_metric import get_classification_score
from src.travel.utils.main_utils import save_object,load_object,write_yaml_file

import shutil


class ModelPusher:

    def __init__(self,
                model_pusher_config:ModelPusherConfig,
                model_eval_artifact:ModelEvaluationArtifact):

        try:
            self.model_pusher_config = model_pusher_config
            self.model_eval_artifact = model_eval_artifact
        except  Exception as e:
            raise TravelException(e,sys)


    def initiate_model_pusher(self,)->ModelPusherArtifact:
        try:
            trained_model_path = self.model_eval_artifact.trained_model_path
            
            logging.info("Creating model pusher dir to save model")

        #Creating model pusher dir to save model
            model_file_path = self.model_pusher_config.model_file_path
            os.makedirs(os.path.dirname(model_file_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=model_file_path)

        #saved model dir
            logging.info("creating saved model dir to save model")
            saved_model_path = self.model_pusher_config.saved_model_path
            os.makedirs(os.path.dirname(saved_model_path),exist_ok=True)
            shutil.copy(src=trained_model_path, dst=saved_model_path)
        
        #prepare artifact
            model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path, model_file_path=model_file_path)
            logging.info(f"Got model pusher artifact : {model_pusher_artifact}")

            return model_pusher_artifact
        except  Exception as e:
            raise TravelException(e, sys)