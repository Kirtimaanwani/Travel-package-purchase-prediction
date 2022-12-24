import os, sys

TARGET_FEATURE :str = "ProdTaken"
FILE_NAME: str = "travel.csv"
ARTIFACT_DIR: str = "artifact"
PIPELINE_NAME: str = "travel"
TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yml")
SCHEMA_DROP_COLS = "drop_columns"  # used to write columns which are to be droopped in schema.yaml file

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
