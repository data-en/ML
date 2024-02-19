import os 
import sys 
from src1.mlProject.exception import CustomException
from src1.mlProject.logger import logging
import pandas as pd 
from src1.mlProject.utils import read_sql_data

from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv')
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # reading the data from my sql 
            df = read_sql_data() # raw data

            logging.info("Reading completed")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            # Converting above raw data imported from mysql into csv file 
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) # saving that csv file into ingestion_config folder into raw_data_path and saying to not save the index also
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            df.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            df.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        
        except Exception as e:
            raise CustomException(e,sys)
