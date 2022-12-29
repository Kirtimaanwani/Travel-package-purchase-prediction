from src.travel.exception import TravelException
import os, sys
from src.travel.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
from src.travel.logger import logging


class TravelModel:
    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
            self.feature_importances_ = None
        except Exception as e:
            raise TravelException(e, sys)
    
    def predict(self,x):
        try:
            #with this predict function , directly do preprocessing and prediction-----in future if wanna to add more steps of feature engineering , it can be added here
           
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict_proba(x_transform)
            self.feature_importances_ = self.get_feature_importances()
            return y_hat
        except Exception as e:
            raise TravelException(e, sys)

    def get_feature_importances(self):
        return self.model.feature_importances_


class ModelResolver:
    def __init__(self,model_dir=SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir

        except Exception as e:
            raise TravelException(e, sys)

    def get_best_model_path(self,)->str:
        try:
            logging.info("getting best model file path")
            timestamps = list(map(int,os.listdir(self.model_dir)))
            latest_timestamp = max(timestamps)
            latest_model_path= os.path.join(self.model_dir,f"{latest_timestamp}",MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            raise TravelException(e, sys)

    def is_model_exists(self)->bool:
        try:
            logging.info("Checking if model is already exists or not")
            if not os.path.exists(self.model_dir):
                logging.info("saved model  directory not found")
                return False

            timestamps = os.listdir(self.model_dir)
            if len(timestamps)==0:
                logging.info("timestamp folder not found in saved models")
                return False
            
            latest_model_path = self.get_best_model_path()

            if not os.path.exists(latest_model_path):
                logging.info("Latest Model not found")
                return False

            logging.info("Model Exists")
            return True
        except Exception as e:
            raise TravelException(e, sys)


  