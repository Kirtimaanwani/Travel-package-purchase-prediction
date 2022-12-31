from src.travel.entity.artifact_entity import ClassificationMetricArtifact
from src.travel.exception import TravelException
from src.travel.logger import logging
from sklearn.metrics import f1_score,precision_score,recall_score
import os,sys

def get_classification_score(y_true,y_pred)->ClassificationMetricArtifact:
    try:
        logging.info("Getting classification score")

        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score=precision_score(y_true,y_pred)

        classsification_metric  =  ClassificationMetricArtifact(f1_score=model_f1_score,
                                                                    precision_score=model_precision_score, 
                                                                        recall_score=model_recall_score)
        logging.info("got ClassificationMetricArtifact")
        return classsification_metric
    except Exception as e:
        raise TravelException(e,sys)