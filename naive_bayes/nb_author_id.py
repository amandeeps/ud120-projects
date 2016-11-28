#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
cf = GaussianNB()

t0 = time()
cf.fit(features_train, labels_train)
print "Training Time", round(time() - t0, 3), "s"

t1 = time()
test_results = cf.predict(features_test)
print "Prediction Time", round(time() - t1, 3), "s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, test_results)
print "Accuracy Score: ", round(accuracy, 3)
#########################################################

#Results
#Training Time 1.081 s
#Prediction Time 0.154 s
#Accuracy Score:  0.973
