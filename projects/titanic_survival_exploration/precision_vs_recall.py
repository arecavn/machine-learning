# As with the previous exercises, let's look at the performance of a couple of classifiers
# on the familiar Titanic dataset. Add a train/test split, then store the results in the
# dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')

X = X._get_numeric_data()
y = X['Survived']
del X['Age'], X['Survived']


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.naive_bayes import GaussianNB

# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    X, y, test_size=0.25, random_state=0)

clf1 = DecisionTreeClassifier()
clf1.fit(features_train,labels_train)
r1 = recall(labels_test,clf1.predict(features_test))
p1 = precision(labels_test,clf1.predict(features_test))
print "Decision Tree recall: {:.2f} and precision: {:.2f}".format(r1,p1)

clf2 = GaussianNB()
clf2.fit(features_train,labels_train)
r2 = recall(labels_test,clf2.predict(features_test))
p2 = precision(labels_test,clf2.predict(features_test))
print "GaussianNB recall: {:.2f} and precision: {:.2f}".format(r2,p2)

results = {
    "Naive Bayes Recall": r2,
    "Naive Bayes Precision": p2,
    "Decision Tree Recall": r1,
    "Decision Tree Precision": p1
}