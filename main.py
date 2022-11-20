from src.travel.exception import TravelException
import sys, os  
from src.travel.logger import logging
from src.travel.pipeline.training_pipeline import TrainPipeline



if __name__=="__main__":
    try:
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()

    except Exception as e: 
        raise TravelException(e, sys)