#!/usr/bin/python

"""
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:
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

#########################################################

from sklearn.svm import SVC

# clf = SVC(kernel='linear')
clf = SVC(kernel='rbf', C=10000.0)

# this version trains on the full training set ...
clf.fit(features_train, labels_train)

# this version takes a sub-index (slice) and trains much faster ...
# features_train_1 = features_train[:len(features_train)/100]
# labels_train_1 = labels_train[:len(features_train)/100]
# clf.fit(features_train_1, labels_train_1)

accuracy = clf.score(features_test, labels_test)
print "Accuracy: ", round(accuracy, 3)

# predictions for elements 10, 26, 50
print "Predictions: "

pred = clf.predict(features_test)
print "Email 10: ", pred[10]
print "Email 26: ", pred[26]
print "Email 50: ", pred[50]

pred_chris = filter (lambda l: l==1, pred)
print "No. of Chris emails:", len(pred_chris)
