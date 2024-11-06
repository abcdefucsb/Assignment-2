import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler
import math

def main():

    # load data with default settings
    # You will need to pick the features you want to use! 
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['NSCLC', 'GENDER', 'Penicillin V Potassium 250 MG',
       'Penicillin V Potassium 500 MG',
       'Computed tomography of chest and abdomen',
       'Plain chest X-ray (procedure)', 'Diastolic Blood Pressure',
       'Body Mass Index', 'Body Weight', 'Body Height',
       'Systolic Blood Pressure', 'Low Density Lipoprotein Cholesterol',
       'High Density Lipoprotein Cholesterol', 'Triglycerides',
       'Total Cholesterol', 'Documentation of current medications',
       'Fluticasone propionate 0.25 MG/ACTUAT / salmeterol 0.05 MG/ACTUAT [Advair]',
       '24 HR Metformin hydrochloride 500 MG Extended Release Oral Tablet',
       'Carbon Dioxide', 'Hemoglobin A1c/Hemoglobin.total in Blood', 'Glucose',
       'Potassium', 'Sodium', 'Calcium', 'Urea Nitrogen', 'Creatinine',
       'Chloride', 'AGE_DIAGNOSIS']
                                                       , split_percent=0.8, split_state=42)

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform (X_val)
    
    print(X_train.shape, X_val.shape, y_val.shape, y_train.shape)


    """
    # for testing purposes once you've added your code
    # CAUTION & HINT: hyperparameters have not been optimized

    log_model = logreg.LogisticRegression(num_feats=1, max_iter=10, tol=0.01, learning_rate=0.00001, batch_size=12)
    log_model.train_model(X_train, y_train, X_val, y_val)
    log_model.plot_loss_history()
            
    """
    log_model = logreg.LogisticRegression(num_feats=27, max_iter=1000, tol=0.01, learning_rate=0.1, batch_size=12)
    log_model.train_model(X_train, y_train, X_val, y_val)
    log_model.plot_loss_history()
    ones=np.ones((X_val.shape[0],1))
    print(ones.shape)
    X_val=np.hstack([X_val,ones])
    result=np.dot(X_val,log_model.W_history[-1])
    for i in range(0,result.shape[0]):
            result[i]=(1)/(1+(math.e)**(-result[i]))
            if result[i]<0.5:
                 result[i]=0
            else:
                 result[i]=1
    print(result)
    
   

    
    



if __name__ == "__main__":
    main()
