import yaml
from src.travel.exception import TravelException
import os, sys
from src.travel.logger import logging
import numpy as np
import dill


def write_yaml_file(file_path: str, content: object, replace: bool = False)->None:
    logging.info(f"Writting yaml file at [{file_path}], replace: [{replace}], from main_utils class")

    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise TravelException(e, sys)


def read_yaml_file(file_path: str) -> dict:
    logging.info(f"Reading yaml file from [{file_path}] from main_utils class")
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file) 
    except Exception as e:
        raise TravelException(e, sys)



def save_numpy_array_data(file_path: str, array: np.array):

    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    
    try:
        logging.info(f"saving numpy array data at [{file_path}] from main_utils class")
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    
    except Exception as e:
        raise TravelException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        logging.info(f"Loading numpy array data from [{file_path}] from main_utils class")
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise TravelException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method of Main_Utils class")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Exited the save_object method of MainUtils class")
    except Exception as e:
        raise TravelException(e, sys) 


def load_object(file_path: str, ) -> object:
    try:
        logging.info("Entered the load_object method of Main_Utils class")
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            logging.info("Exiting the load_object method of MainUtils class")
            return dill.load(file_obj)
        logging.info("Exit from load_object method of Main_Utils class FAILED")
    except Exception as e:
        raise TravelException(e, sys) 
