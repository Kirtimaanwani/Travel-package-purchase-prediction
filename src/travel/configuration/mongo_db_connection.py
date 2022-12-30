import os
import pymongo
from src.travel.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME
import certifi
ca = certifi.where()

# from credential.CREDENTIAL import MONGO_DB_URL
MONGO_DB_URL = os.getenv("MONGO_DB_URL")

class MongoDBClient:
    client = None
    def __init__(self, database_name=DATA_INGESTION_DATABASE_NAME):
        try:
            if MongoDBClient.client is None:
                MongoDBClient.client = pymongo.MongoClient(MONGO_DB_URL, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
        except Exception as e:
            raise 
