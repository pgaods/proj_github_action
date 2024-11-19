# Databricks notebook source
# train.py

import numpy as np
import pandas as pd
import sys, os
from datetime import datetime
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

class MyLogisticRegression:
    def __init__(self, dataset, train_test_split_ratio, eval_metric, random_seed):
        self.data = dataset
        self.train_test_split_ratio = train_test_split_ratio
        self.eval_metric = eval_metric
        self.random_seed = random_seed

    def preprocess_data(self):
        logger.info("Starting data preprocessing...")
        self.data['age'].fillna(self.data['age'].median(), inplace=True)
        self.data['embarked'].fillna(self.data['embarked'].mode()[0], inplace=True)
        self.data['fare'].fillna(self.data['fare'].median(), inplace=True)
        self.data = pd.get_dummies(self.data, columns=['sex', 'embarked'], drop_first=True) # converting categorical variables to dummies 
        self.X = self.data[["sex_male", "embarked_Q", "age", "fare", "pclass", "sibsp", "parch"]] # using a few features to train the model
        self.y = self.data['survived']
        logger.info("Data preprocessing completed.")

    def train_and_evaluate(self):
        logger.info("Starting model training and evaluation pipeline...")
        self.preprocess_data()  # ensuring X is created
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.train_test_split_ratio, random_state=self.random_seed)
        
        model = LogisticRegression(max_iter=400)
        model.fit(X_train, y_train)
        
        y_hat = model.predict(X_test)
        if self.eval_metric == "accuracy":
            eval_metric = accuracy_score(y_test, y_hat) # using accuracy
        else:
            raise ValueError("Invalid Evaluation Metrics")
        
        logger.info(f"Model training and evaluation pipeline completed with {self.eval_metric}: {eval_metric}")
        return eval_metric