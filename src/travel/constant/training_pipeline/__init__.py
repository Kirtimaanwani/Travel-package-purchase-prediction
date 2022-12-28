import os

TARGET_COLUMN :str = "ProdTaken"
FEATURE_NAMES:str = ['Age','TypeofContact','CityTier','DurationOfPitch','Occupation',
                        'Gender','NumberOfPersonVisiting','NumberOfFollowups','ProductPitched',
                        'PreferredPropertyStar','MaritalStatus','NumberOfTrips','Passport','PitchSatisfactionScore'
                        ,'OwnCar','NumberOfChildrenVisiting','Designation','MonthlyIncome']
FILE_NAME: str = "travel.csv"
ARTIFACT_DIR: str = "artifact"
PIPELINE_NAME: str = "travel"
TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yml")
SCHEMA_DROP_COLS = "drop_columns"  # used to write columns which are to be droopped in schema.yaml file
MODEL_FILE_NAME = "model.pkl"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

""" 
Data Ingestion related constant start with DATA_INGESTION variable Name
"""
DATA_INGESTION_DATABASE_NAME:str = "finance1"
DATA_INGESTION_COLLECTION_NAME:str = "data_sample" # same as ""from sensor.constant.database import COLLECTION_NAME""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2  # test_size


"""
Data Validation related constant start with DATA_VALIDATION_variable_Name
"""
DATA_VALIDATION_DIR_NAME:str = "data_validation"
DATA_VALIDATION_VALID_DIR:str = "validated"
DATA_VALIDATION_INVALID_DIR:str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "drift_report.yaml"

"""
Data Transformation related constants starts with DATA_TRANS_variable_Name
"""
DATA_TRANS_DIR_NAME:str = "data_transformation"
DATA_TRANS_TRANSFORMED_DATA_DIR:str = "transformed"
DATA_TRANS_TRANSFORMED_OBJECT_DIR:str = "transformed_object"


"""
Model Trainer related constants starts with MODEL_TRAINER_var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME : str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float =  0.8
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.1
MODEL_TRAINER_TUNED_PARAMETERS : dict = {'criterion': 'gini', 
                                'max_depth': int(17.45980664953067), 
                                'min_samples_leaf': int(1.2653869952885435), 
                                'min_samples_split': int(2.0835728200367796), 
                                'n_estimators': int(151.27017362733017)}
# params_tuned =  {'max_depth': 17.45980664953067, 'min_samples_leaf': 1.2653869952885435, 'min_samples_split': 2.0835728200367796, 'n_estimators': 151.27017362733017}


