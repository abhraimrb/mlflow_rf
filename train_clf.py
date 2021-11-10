import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

from sklearn.model_selection import train_test_split


import mlflow
import mlflow.sklearn




if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    loan_approval_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cust_data_processed.csv")
    data = pd.read_csv( loan_approval_path )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    X_train = train.drop(["loan_approval_status"], axis=1)
    X_test = test.drop(["loan_approval_status"], axis=1)
    y_train = train[["loan_approval_status"]]
    y_test = test[["loan_approval_status"]]

    

    with mlflow.start_run():
        n_estimators = 200
        criterion = 'gini'
        min_samples_split = 5
        min_samples_leaf = 2           
        model = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf
        model.fit(X_train, y_train)
        predictions =  model.predict(X_test)
        predictions_proba = model.predict_proba(X_test)
        
        test_accuracy = accuracy_score(y_test, predictions)
        test_precision_score = precision_score(y_test, predictions)
        test_recall_score = recall_score(y_test, predictions)
        test_f1_score = f1_score(y_test, predictions)
        auc_score = roc_auc_score(y_test, predictions_proba[:,1])
        metrics = {"Test_accuracy": test_accuracy, "Test_precision_score": test_precision_score,
                   "Test_recall_score":test_recall_score,"Test_f1_score":test_f1_score, "auc score":auc_score}
    
  # Log the value of the metric from this run.
        mlflow.log_metrics(metrics )
   
    
    #mlflow.set_tag("Classifier", "Random Forest")
    
        mlflow.set_tag("Classifier", "RF-tuned parameters-wo autolog")
       
        mlflow.sklearn.log_model(model, "RF-tuned parameters-wo autolog")
        
        

        
