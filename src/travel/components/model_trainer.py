from src.travel.utils.main_utils import load_numpy_array_data, save_object, load_object
from src.travel.exception import TravelException
from src.travel.logger import logging
import os, sys
from src.travel.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.travel.entity.config_entity import ModelTrainerConfig

from sklearn.ensemble import RandomForestClassifier
from src.travel.ml.metric.classification_metric import get_classification_score
from src.travel.ml.model.estimator import SensorModel

from src.travel.constant.training_pipeline import MODEL_TRAINER_TUNED_PARAMETERS

class ModelTrainer:

    def __init__(self,
                    model_trainer_config: ModelTrainerConfig,
                         data_transformation_artifact: DataTransformationArtifact):
        
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise TravelException(e, sys)


    def train_model(self, x_train, y_train):
        try:
            logging.info("performing RandomForestClassifier fit on x_train and y_train")
            rf_clf = RandomForestClassifier(**MODEL_TRAINER_TUNED_PARAMETERS)

            rf_clf.fit(x_train, y_train)
            logging.info("RandomForestClassifier fit done")
            
            logging.info("returning RandomForestClassifier object")
            return rf_clf
        except Exception as e:
            raise TravelException(e, sys)
    

    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            logging.info("Initiating Model Trainer for training the model...")
            
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            logging.info("loading train and test array for training...")
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)

            logging.info("Creating [x_train, y_train, x_test, y_test]")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
                                    )   

            logging.info("going to train_model and Getting Object for trained_model...")
            model = self.train_model(x_train=x_train, y_train=y_train)

            logging.info("going to predict on x_train to get y_train_pred")
            y_train_pred = model.predict(x_train)
            
            logging.info("going to get classification score metric on y_train and y_train_pred")
            classification_train_metric =  get_classification_score(y_true=y_train,
                                                                        y_pred=y_train_pred)

            if (classification_train_metric.f1_score  <=  self.model_trainer_config.expected_accuracy):
                raise Exception("Trained model is not good to provide expected accuracy!")


            logging.info("Performing similar steps to get classification score metric on y_test and y_test_pred...")
            y_test_pred = model.predict(x_test)
            classification_test_metric =   get_classification_score(y_true=y_test, 
                                                                       y_pred=y_test_pred)

        #Overfitting and Underfitting
            diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)
            
            if (diff > self.model_trainer_config.overfitting_underfitting_threshold):
                raise Exception("""Model is not good 
                            try to do more experimentation since model is crossing Overfitting_Underfitting_Threshold!""")

        # Creating pipeline model so that it can be used for prediction
            logging.info("loading preprocessor object...")
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            logging.info("creating directory...")
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)

            logging.info("creating object for SensorModel(preprocessor=preprocessor,model=model)...")
            sensor_model = SensorModel(preprocessor=preprocessor,model=model)


            logging.info("Saving object for SensorModel(preprocessor=preprocessor,model=model) in created directory...")
            save_object(self.model_trainer_config.trained_model_file_path, obj=sensor_model)

        # Creating model_trainer Artifact
            model_trainer_artifact   =  ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path, 
                                                                train_metric_artifact=classification_train_metric,
                                                                    test_metric_artifact=classification_test_metric)
            logging.info(f"Model trainer artifact: {model_trainer_artifact}\n\n")
            print(f"Model trainer artifact: {model_trainer_artifact}\n\n")
            return model_trainer_artifact

        except Exception as e:
            raise TravelException(e, sys)