#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the
### visualization code (prettyPicture) to show you the decision boundary



from time import time

print "Size of training set: ", len(features_train)
print "No. of features: ", len(features_train[0])

# classifiers

# Random Forest
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(
#   # criterion='entropy',
#   min_samples_split=20,
#   n_estimators=20
# )
# Result: ~ 0.924 accuracy

# Gradient Boosting Classifier
# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier(
#   min_samples_split=4,
#   n_estimators=200,
#   learning_rate=0.02
# )
# Result: ~ 0.92 accuracy


# Adaboost classifier
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(
  # n_estimators=100,
  # learning_rate=0.2
)
# Result: ~ 0.924 accuracy. No improvement over default by tuning.

# train
print "Training classifier ..."
t_fit_0 = time()
clf.fit(features_train, labels_train)
t_fit_1 = time()
print "Training complete."
print "Training time: ", round (t_fit_1-t_fit_0, 2), "sec."

# accuracy
acc = clf.score(features_test, labels_test)
print "Accuracy: ", round(acc, 3)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

# move show to the end, as it seems to grab the thread and prevent execution
# of any code appearing _below_ it!
plt.show()
