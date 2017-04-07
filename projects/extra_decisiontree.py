# # Import libraries necessary for this project
# import numpy as np
# import pandas as pd
# from sklearn.cross_validation import ShuffleSplit
#
# # Import supplementary visualizations code visuals.py
# import visuals as vs
#
# # Pretty display for notebooks
# %matplotlib inline
#
# # Load the Boston housing dataset
# data = pd.read_csv('housing.csv')
# prices = data['MEDV']
# # Drop column MEDV (axis=1 denotes that we are referring to a column, not a row)
# features = data.drop('MEDV', axis = 1)
# # Success
# print "Boston housing dataset has {} data points with {} variables each.".format(*data.shape)
#
# from IPython.display import Image
# from sklearn.externals.six import StringIO
# import pydot
# from sklearn import tree
#
# clf = DecisionTreeRegressor(max_depth=4)
# clf = clf.fit(X_train, y_train)
# dot_data = StringIO()
# tree.export_graphviz(clf, out_file=dot_data,
#                      feature_names=X_train.columns,
#                      class_names="PRICES",
#                      filled=True, rounded=True,
#                      special_characters=True)
# graph = pydot.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png())
