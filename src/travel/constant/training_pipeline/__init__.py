

TARGET_FEATURE :str = "ProdTaken"
FILE_NAME: str = "travel.csv"
ARTIFACT_DIR: str = "artifact"
PIPELINE_NAME: str = "travel"
TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = "test.csv"

""" 
Data Ingestion related constant start with DATA_INGESTION variable Name
"""
DATA_INGESTION_DATABASE_NAME:str = "finance1"
DATA_INGESTION_COLLECTION_NAME:str = "data_sample" # same as ""from sensor.constant.database import COLLECTION_NAME""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2  # test_size



