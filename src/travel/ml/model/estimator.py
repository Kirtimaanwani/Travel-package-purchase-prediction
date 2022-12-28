from src.travel.exception import TravelException
import os, sys
# from src.travel.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
from src.travel.logger import logging


class SensorModel:

    def __init__(self, preprocessor, model):
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise TravelException(e, sys)
    
    def predict(self,x):
        try:
            #with this predict function , directly do preprocessing and prediction-----in future if wanna to add more steps of feature engineering , it can be added here
           
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise TravelException(e, sys)
