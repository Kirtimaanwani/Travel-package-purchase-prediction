import sys
from typing import Optional

import numpy as np
import pandas as pd

from src.travel.configuration.mongo_db_connection import MongoDBClient
from src.travel.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME
from src.travel.exception import TravelException


class TravelData:
    """
    This class will help to export mongo db record as pandas DataFrame
    """

    def __init__(self):
        try:
            self.mongo_client = MongoDBClient(database_name=DATA_INGESTION_DATABASE_NAME)

        except Exception as e:
            raise TravelException(e, sys)


    def export_collection_as_dataframe(self, collection_name:str, database_name :Optional[str] =None) -> pd.DataFrame:
        try:
            """
            export entire collection as dataframe
            return pd.dataframe of collection 
            """
            if database_name is None: 
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            
            df = pd.DataFrame(list(collection.find()))
            
            if "_id" in df.columns.to_list():
                df = df.drop(columns=[ "_id" ], axis=1)
            
            df.replace({"na": np.nan}, inplace=True)  # "na" in mongodb record converts into np.nan

            return df


        except Exception as e:
            raise TravelException(e, sys)