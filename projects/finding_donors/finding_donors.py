# Machine learning - Finding donors - Python version of i
# Ninhsth

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualization code visuals.py
# import visuals as vs

# Pretty display for notebooks
# %matplotlib inline

# Load the Census dataset
data = pd.read_csv("census.csv")

# Success - Display the first record
display(data.head(n=1))
# data.head(n=1)


# TODO: Total number of records
n_records = data.shape[0]

# TODO: Number of records where individual's income is more than $50,000
n_greater_50k = data[data.income == '>50K'].shape[0]

# TODO: Number of records where individual's income is at most $50,000
n_at_most_50k = data[data.income == '<=50K'].shape[0]

# TODO: Percentage of individuals whose income is more than $50,000
greater_percent = (n_greater_50k*100.0)/n_records

# Print the results
print "Total number of records: {}".format(n_records)
print "Individuals making more than $50,000: {}".format(n_greater_50k)
print "Individuals making at most $50,000: {}".format(n_at_most_50k)
print "Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent)

# Split the data into features and target label
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualize skewed continuous features of original data
# vs.distribution(data)

# Log-transform the skewed features
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# Visualize the new log distributions
# vs.distribution(features_raw, transformed = True)

# TODO: One-hot encode the 'features_raw' data using pandas.get_dummies()
import pandas
features = pandas.get_dummies(features_raw)

# TODO: Encode the 'income_raw' data to numerical values
income = income_raw.apply(lambda x: 1 if x == '>50K' else 0)

# Print the number of features after one-hot encoding
encoded = list(features.columns)
print "{} total features after one-hot encoding.".format(len(encoded))

# Uncomment the following line to see the encoded feature names
print encoded
print income.head()

# Import train_test_split
from sklearn.cross_validation import train_test_split

# Split the 'features' and 'income' data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0)

# Show the results of the split
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


### Question 1 - Naive Predictor Performace
# TODO: Calculate accuracy
accuracy = float(n_greater_50k)/n_records

# TODO: Calculate F-score using the formula above for beta = 0.5
beta=0.5
precision = float(n_greater_50k)/n_records
recall = float(n_greater_50k)/n_greater_50k
fscore = float(((1+pow(beta,2))*(precision*recall)))/((pow(beta,2)*precision) + recall)

# Print the results
print "Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore)

# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # TODO: Fit the learner to the training data using slicing with 'sample_size'
    # print "X_train.iloc[:sample_size,:] ", X_train.iloc[:sample_size,:]
    # exit
    start = time() # Get start time
    learner.fit(X_train[:sample_size],y_train[:sample_size])
    end = time() # Get end time

    # TODO: Calculate the training time
    results['train_time'] = end - start

    # TODO: Get the predictions on the test set,
    #       then get predictions on the first 300 training samples
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[0:300])
    end = time() # Get end time

    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start

    # TODO: Compute accuracy on the first 300 training samples
    results['acc_train'] = accuracy_score(y_train[0:300], predictions_train)

    # TODO: Compute accuracy on test set
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # TODO: Compute F-score on the the first 300 training samples
    results['f_train'] = fbeta_score(y_train[0:300],predictions_train, beta=beta)

    # TODO: Compute F-score on the test set
    results['f_test'] = fbeta_score(y_test,predictions_test,beta=beta)

    # Success
    print "{} trained on {} samples.".format(learner.__class__.__name__, sample_size)

    # Return the results
    return results


# TODO: Import the three supervised learning models from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

# TODO: Initialize the three models
a_random_state = 12
clf_A = LogisticRegression(random_state=a_random_state)
clf_B = SVC(random_state=a_random_state)
clf_C = GradientBoostingClassifier(random_state=a_random_state)

# clf_A.fit(X_train[:362],y_train[:362])
# exit

# TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
samples_1 = int(round(X_train.shape[0]*0.01))
samples_10 = int(round(X_train.shape[0]*0.1))
samples_100 = X_train.shape[0]

print "shape: ", X_train.shape
print "number of samples for 1%, 10%, and 100% of the training data: ", samples_1, samples_10, samples_100

# Collect results on the learners
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    print "********** ", clf_name, "***********"
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = \
            train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
# vs.evaluate(results, accuracy, fscore)

########## Implementation: Model Tuning
# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
# TODO: Initialize the classifier
a_random_state = 12
clf = GradientBoostingClassifier(random_state=a_random_state)

# TODO: Create the parameters list you wish to tune
#Note: Avoid tuning the max_features parameter of your learner if that parameter is available!???
# parameters = {'learning_rate':[0.05,0.1,0.2],'n_estimators':[100,150,200],'max_depth':[3,5,10],'min_samples_split':[5,10,15], 'subsample': [0.85,0.9,1]}
# parameters = {'learning_rate':[0.05,0.1,0.2],'n_estimators':[100,150,200],'max_depth':[3,4,5],'min_samples_split':[5,10,15]}
parameters = {'learning_rate':[0.05,0.1,0.15],'n_estimators':[100,110,120],'max_depth':[3,4,5]}

# TODO: Make an fbeta_score scoring object
scorer = make_scorer(fbeta_score,beta=0.5)

# TODO: Perform grid search on the classifier using 'scorer' as the scoring method
print "start GridSearchCV: ", time()
grid_obj = GridSearchCV(clf,param_grid=parameters,scoring=scorer)
print "end GridSearchCV: ", time()

# TODO: Fit the grid search object to the training data and find the optimal parameters
start_time = time()
print "start grid_fit: ", start_time
grid_fit = grid_obj.fit(X_train,y_train)
end_time = time()
print "end grid_fit: ", end_time
print "total grid_fit time:", (end_time - end_time)

# Get the estimator
best_clf = grid_fit.best_estimator_
print "best_clf: ", best_clf
# Make predictions using the unoptimized and model
start_time = time()
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)
end_time = time()
print "total predict time:", (end_time - end_time)


# Report the before-and-afterscores
print "Unoptimized model\n------"
print "Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions))
print "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5))
print "\nOptimized Model\n------"
print "Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions))
print "Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5))

importances = clf.feature_importances_
print "Importances: ", importances

# best_clf:  GradientBoostingClassifier(init=None, learning_rate=0.1, loss='deviance',
#                                       max_depth=5, max_features=None, max_leaf_nodes=None,
#                                       min_samples_leaf=1, min_samples_split=2,
#                                       min_weight_fraction_leaf=0.0, n_estimators=100,
#                                       presort='auto', random_state=12, subsample=1.0, verbose=0,
#                                       warm_start=False)
# total predict time: 0.0
# Unoptimized model
# ------
# Accuracy score on testing data: 0.8630
# F-score on testing data: 0.7395
#
# Optimized Model
# ------
# Final accuracy score on the testing data: 0.8697
# Final F-score on the testing data: 0.7504
# Importances:  [  1.16469775e-01   9.91755987e-02   1.34916342e-01   1.26455315e-01
#                  7.21541423e-02   1.66359331e-02   9.34690192e-03   6.38015423e-04
#                  8.05510079e-03   1.67999764e-02   1.08438914e-03   0.00000000e+00
#                  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#                  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#                  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.37883635e-03
#                  0.00000000e+00   3.26517192e-04   3.25506872e-03   0.00000000e+00
#                  0.00000000e+00   8.77978395e-03   1.20482775e-01   2.00985445e-03
#                  3.46956926e-03   0.00000000e+00   4.43095706e-04   9.10298332e-04
#                  0.00000000e+00   0.00000000e+00   3.35374145e-02   2.57506735e-02
#                  1.09804445e-02   8.81954412e-03   2.37198816e-02   0.00000000e+00
#                  2.05977580e-02   8.16303819e-03   1.39362827e-02   1.09140659e-02
#                  0.00000000e+00   1.60864241e-02   1.17322494e-03   1.62452115e-03
#                  4.14704615e-03   0.00000000e+00   2.49572872e-02   0.00000000e+00
#                  8.39425321e-04   1.99669804e-03   0.00000000e+00   5.40884110e-03
#                  5.95860078e-03   1.53004005e-02   2.37016644e-03   4.76091688e-03
#                  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#                  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#                  0.00000000e+00   0.00000000e+00   5.61976776e-04   0.00000000e+00
#                  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#                  0.00000000e+00   0.00000000e+00   0.00000000e+00   0.00000000e+00
#                  1.14867092e-03   9.23558776e-04   0.00000000e+00   3.95552826e-03
#                  0.00000000e+00   0.00000000e+00   0.00000000e+00   1.22542204e-04
#                  0.00000000e+00   4.94664026e-04   0.00000000e+00   0.00000000e+00
#                  3.95978967e-03   0.00000000e+00   0.00000000e+00   5.51907168e-04
#                  4.45141787e-03   0.00000000e+00   0.00000000e+00]

########## Question 6 - Feature Relevance Observation
# TODO: Import a supervised learning model that has 'feature_importances_'
from sklearn.ensemble import GradientBoostingClassifier
# TODO: Train the supervised model on the training set
model = GradientBoostingClassifier()
model.fit(X_train,y_train)

# TODO: Extract the feature importances
importances = model.feature_importances_

# Plot
# vs.feature_plot(importances, X_train, y_train)