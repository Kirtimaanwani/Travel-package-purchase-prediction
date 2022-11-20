import sys, os
from src.travel.exception import TravelException
from src.travel.logger import logging
from src.travel.entity.config_entity import DataIngestionConfig
from src.travel.entity.artifact_entity import DataIngestionArtifact
from pandas import DataFrame
from src.travel.data_access.data_travel import TravelData




class DataIngestion:

    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:     
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise TravelException(e, sys)


    def export_data_into_feature_store(self) -> DataFrame:
            """
            Export mongo db collection records as DataFrame into feature store
            """
            try:   
                logging.info("Exporting data from mongo database to feature store...")  
                sensor_data = TravelData()
                dataframe = sensor_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)  # here not provided  database name since its already taken in mongo_db_connection 
                feature_store_file_path = self.data_ingestion_config.feature_store_file_path

                # now Creating folder for feature store
                logging.info("Creating folder for feature store")
                dir_path = os.path.dirname(feature_store_file_path)
                os.makedirs(dir_path, exist_ok=True)
                
                logging.info("Exporting csv in feature store")
                dataframe.to_csv(feature_store_file_path, index=False, header=False)
                
                return dataframe

            except Exception as e:
                raise TravelException(e, sys) 



    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Initiating data ingestion...")
            dataframe = self.export_data_into_feature_store()

            data_ingestion_artifact = DataIngestionArtifact(feature_store_file_path=self.data_ingestion_config.feature_store_file_path)
            
            logging.info("Getting data_ingestion_artifact")
            return data_ingestion_artifact

        except Exception as e:
            raise TravelException(e, sys)