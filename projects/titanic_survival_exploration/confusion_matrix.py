# In this exercise, we'll use the Titanic dataset as before, train two classifiers and
# look at their confusion matrices. Your job is to create a train/test split in the data
# and report the results in the dictionary at the bottom.

import numpy as np
import pandas as pd

# Load the dataset
from sklearn import datasets

X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    X, y, test_size=0.25, random_state=0)

clf1 = DecisionTreeClassifier()
clf1.fit(features_train,labels_train)
confusion1 = confusion_matrix(labels_test, clf1.predict(features_test))
print "Confusion matrix for this Decision Tree:\n", confusion1

clf2 = GaussianNB()
clf2.fit(features_train,labels_train)
confusion2 = confusion_matrix(labels_test, clf2.predict(features_test))
print "GaussianNB confusion matrix:\n",confusion2

#TODO: store the confusion matrices on the test sets below

confusions = {
    "Naive Bayes": confusion2,
    "Decision Tree": confusion1
}

"""Note that the scikit-learn module uses the following transposed form instead:
TN FP

FN TP"""