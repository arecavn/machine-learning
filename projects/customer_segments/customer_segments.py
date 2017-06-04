# Machine learning - Customer segments - Python version
# Ninhsth

##### Getting Started
# Import libraries necessary for this project
# matplotlib.animation.BackendError: The current backend is 'MacOSX'
# and may go into an infinite loop with blit turned on.  Either
# turn off blit or use an alternate backend, for example, like
#     'TKAgg', using the following prepended to your source code:

import matplotlib
matplotlib.use('TKAgg')

import numpy as np
import pandas as pd
#from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
#import visuals as vs

# Pretty display for notebooks
#%matplotlib inline

##### override display function
def display (parameters):
    print parameters


# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
    print data.head(2)
except:
    print "Dataset could not be loaded. Is the dataset missing?"


###### Data Exploration
# Display a description of the dataset
display(data.describe())

##### Implementation: Selecting Samples
# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [1,16,203]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

##### Implementation: Feature Relevance
# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
selected_field = "Grocery"
new_data = data.drop([selected_field], axis = 1, inplace = False)
# display(new_data.head(2))
new_output=data[selected_field]

# TODO: Split the data into training and testing sets using the given feature as the target
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(new_data, new_output, test_size = 0.25, random_state = 11)

# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.metrics import fbeta_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=12)
regressor.fit(X_train,y_train)
predict = regressor.predict(X_test)

# TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test,y_test)
print "Score of the prediction for ", selected_field, ": "
display(score)

display(regressor.feature_importances_)

# import matplotlib.pyplot as plt
# def plot_heatmap(percentiles_data, ax=None, figsize=(10,5), title="Heatmap"):
#     fig, ax = plt.subplots(figsize=figsize) if ax is None else (None, ax)
#     ax.set_title(title)
#     ax.imshow(percentiles_data, cmap=plt.cm.Greys, interpolation='nearest')
#     ax.set_xticks(np.arange(len(percentiles_data.columns.values)))
#     ax.set_xticklabels(percentiles_data.columns.values, rotation=90)
#     ax.set_yticks(np.arange(len(percentiles_data.index.values)))
#     ax.set_yticklabels(percentiles_data.index.values)
#     for i, feature in enumerate(percentiles_data):
#         for j, _ in enumerate(percentiles_data[feature]):
#             ax.text(i, j, "{:0.2f}".format(percentiles_data.iloc[j,i]), verticalalignment='center', \
#                                                                                         horizontalalignment='center', color=plt.cm.Reds(1-percentiles_data.iloc[j,i]), fontweight='bold')

    # plt.show()

# samples_percentiles = data.rank(pct=True).loc[indices]
# plot_heatmap(samples_percentiles, figsize=(7,5), title="Percentiles")

##### Visualize Feature Distributions
# Produce a scatter matrix for each pair of features in the data
# pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

##### Implementation: Feature Scaling
# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
log_samples = np.log(samples)

# Produce a scatter matrix for each pair of newly-transformed features
# pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

print "Display the log-transformed sample data:"
display(log_samples)

print "\n\n*************************************** Implementation: Outlier Detection"
outliers_all=pd.DataFrame()
# For each feature find the data points with extreme high or low values
for idx, feature  in enumerate(log_data.keys()):
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature],25)

    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature],75)

    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = Q3-Q1

    # Display the outliers
    print "Data points considered outliers for the feature '{}' ".format(feature)
    print "Valid range: {} - {}: ".format(Q1 - step, Q3 + step)
    print "(Q1: {} Q3: {} step: {} Min: {} Max: {})".format(Q1, Q3, step, np.min(log_data[feature]), np.max(log_data[feature]))
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])

    ## Additional work
    print "Added to array"
    indices_i = pd.DataFrame(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index.values)
    outliers_all=outliers_all.append(indices_i)


# print "outliers_all: ", outliers_all.sort_index()

# OPTIONAL: Select the indices for data points you wish to remove
outliers  = [154,97,128]

# Remove the outliers, if any were specified
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

##### Implementation: PCA
# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(good_data)
# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
# pca_results = vs.pca_results(good_data, pca)

# Generate PCA results plot
# pca_results = vs.pca_results(good_data, pca)
# Display sample log-data after having a PCA transformation applied
# print pd.DataFrame(np.round(pca_samples, 4))
# print pca_results.index.values
# display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components = 2)
pca.fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

# TODO: Apply your clustering algorithm of choice to the reduced data
from sklearn.mixture import GMM
from sklearn.metrics import silhouette_score

def clustering(i_components):
    clusterer = GMM(n_components = i_components)
    clusterer.fit(reduced_data)

    # TODO: Predict the cluster for each data point
    preds = clusterer.predict(reduced_data)

    # TODO: Find the cluster centers
    centers = clusterer.means_

    # TODO: Predict the cluster for each transformed sample data point
    sample_preds = clusterer.predict(pca_samples)

    # TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
    score = silhouette_score(reduced_data, preds)
    print("{} clusters has {} score".format(i_components, score))
    return preds, centers, sample_preds

for l_clusters in range(2,15):
    clustering(l_clusters)

# Display the results of the clustering from implementation
preds, centers, sample_preds = clustering(2)
# vs.cluster_results(reduced_data, preds, centers, pca_samples)

# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers )

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

# Compare with mean and median
print "True Centers vs Mean"
display(true_centers - data.mean().round())
print "True Centers vs Median"
display(true_centers - data.median().round())

# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred