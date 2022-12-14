
import os,sys
from src.travel.exception import TravelException
from src.travel.logger import logging

from src.travel.entity.config_entity import ModelEvaluationConfig
from src.travel.entity.artifact_entity import DataValidationArtifact,ModelTrainerArtifact,ModelEvaluationArtifact

from src.travel.ml.metric.classification_metric import get_classification_score
from src.travel.ml.model.estimator import TravelModel
from src.travel.utils.main_utils import save_object, load_object, write_yaml_file

from src.travel.ml.model.estimator import ModelResolver
from src.travel.constant.training_pipeline import TARGET_COLUMN
import pandas  as  pd


class ModelEvaluation:


    def __init__(self,
                    model_eval_config:ModelEvaluationConfig,
                        data_validation_artifact:DataValidationArtifact,
                            model_trainer_artifact:ModelTrainerArtifact):

        try:
            self.model_eval_config = model_eval_config
            self.data_validation_artifact = data_validation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise TravelException(e,sys)

    
    def initiate_model_evaluation(self)->ModelEvaluationArtifact:
        try:
            logging.info("Initiating model evaluation")
            valid_train_file_path = self.data_validation_artifact.valid_train_file_path
            valid_test_file_path = self.data_validation_artifact.valid_test_file_path

        # valid train and test file dataframe
            train_df = pd.read_csv(valid_train_file_path)
            test_df = pd.read_csv(valid_test_file_path)

        # concatinating train and test and doing TargetValueMapping

            df = pd.concat([train_df,test_df])
            y_true = df[TARGET_COLUMN]
            df.drop(TARGET_COLUMN,axis=1,inplace=True)

            train_model_file_path = self.model_trainer_artifact.trained_model_file_path
            model_resolver = ModelResolver()
            is_model_accepted=True
        
        # Returning trained model if any other model or its directory is not present to compare
            if not model_resolver.is_model_exists():
                model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=None, 
                    best_model_path=None, 
                    trained_model_path=train_model_file_path, 
                    train_model_metric_artifact=self.model_trainer_artifact.test_metric_artifact, 
                    best_model_metric_artifact=None)

                
                logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact
            

        # if model exists, then compare it with trained model

            latest_model_path = model_resolver.get_best_model_path()
            latest_model = load_object(file_path=latest_model_path)
            train_model = load_object(file_path=train_model_file_path)

            # Predicting with both models
            y_trained_pred, y_trained_pred_prob = train_model.predict(df)
            y_latest_pred, y_latest_pred_prob  = latest_model.predict(df)

            # Getting Score with both model
            trained_metric = get_classification_score(y_true, y_trained_pred)
            latest_metric = get_classification_score(y_true, y_latest_pred)

        # Checking accuracy i.e f1 score , it should have difference more than setted threshold
            improved_accuracy = (trained_metric.f1_score - latest_metric.f1_score)
            if self.model_eval_config.change_threshold < improved_accuracy:
                is_model_accepted=True
            else:
                is_model_accepted=False

            model_evaluation_artifact = ModelEvaluationArtifact(
                    is_model_accepted=is_model_accepted, 
                    improved_accuracy=improved_accuracy, 
                    best_model_path=latest_model_path, 
                    trained_model_path=train_model_file_path, 
                    train_model_metric_artifact=trained_metric, 
                    best_model_metric_artifact=latest_metric)

            model_eval_report = model_evaluation_artifact.__dict__
            print(model_eval_report)

        # Saving the evaluation report
            os.makedirs(self.model_eval_config.model_evaluation_dir, exist_ok=True)
            write_yaml_file(self.model_eval_config.report_file_path, model_eval_report, replace=True)
            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}\n\n")

            return model_evaluation_artifact        
        except Exception as e:
            raise TravelException(e,sys)




