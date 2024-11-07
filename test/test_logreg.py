"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
import numpy as np
import pandas as pd
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler

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

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform (X_val)
    
print(X_train.shape, X_val.shape, y_val.shape, y_train.shape)


def test_updates():
	# Check that your gradient is being calculated correctly
	# What is a reasonable gradient? Is it exploding? Is it vanishing? 
	
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training
	# What is a reasonable loss?
       log_model = logreg.LogisticRegression(num_feats=27, max_iter=1000, tol=0.01, learning_rate=0.1, batch_size=12)
       log_model.train_model(X_train, y_train, X_val, y_val)
       assert np.linalg.norm(log_model.gradient_history[0])>np.linalg.norm(log_model.gradient_history[133])


       assert log_model.loss_history_val[0]>log_model.loss_history_val[133]

      

       #pass

def test_predict():
	# Check that self.W is being updated as expected 
 	# and produces reasonable estimates for NSCLC classification
       log_model = logreg.LogisticRegression(num_feats=27, max_iter=1000, tol=0.01, learning_rate=0.1, batch_size=12)
       log_model.train_model(X_train, y_train, X_val, y_val)
       assert np.linalg.norm(log_model.W_history[0])!=np.linalg.norm(log_model.W_history[66])
       assert np.linalg.norm(log_model.W_history[67])!=np.linalg.norm(log_model.W_history[133]) 
       
	# What should the output should look like for a binary classification task?

	# Check accuracy of model after training
       #result=np.dot(X_train,log_model.W_history[-1])
       # for i in range(0,len(y_val)):

       #pass