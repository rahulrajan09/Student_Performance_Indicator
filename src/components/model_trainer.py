import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,AdaBoostRegressor,RandomForestRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        
    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test data input")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "Random Forest":RandomForestRegressor(),
                "Decision Tress":DecisionTreeRegressor(),
                "Gradiest Boosting":GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbours Regressor":KNeighborsRegressor(),
                "Support Vector Regressor":SVR(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(),
                "Adaboost Regressor":AdaBoostRegressor()
            }
            
            
            model_report:dict=evaluate_model(
                X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models
            )
            
            
            best_model_score= max(sorted(model_report.values()))
            
            best_model_name= list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model=models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No best model Found")
            
            logging.info("Best found model on both train and test data")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info("model file saved as pickle")
            
            predicted = best_model.predict(X_test)
            r2score=r2_score(y_test,predicted)
            
            return r2score
            
            
            
        except Exception as e:
            raise CustomException(e,sys)