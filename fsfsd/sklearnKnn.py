import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split

# read the csv file into our data variable
data = pd.read_csv('C:\\Users\\prafu\\Desktop\\fsfsd\\heart.csv')


# delete the unwanted id column
#data.drop(['id'], 1, inplace=True)

# get our attributes and classes in place
X = np.array(data.drop(['target'], 1))
y = np.array(data['target'])

# split data into training and testing sections
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)


# initialize our classifier
knn = neighbors.KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=None, n_neighbors=5, p=2,weights='uniform')

# fit the classifier with the training data
knn.fit(X_train, y_train)

# calculating accuracy with test data
accuracy = knn.score(X_test, y_test)

# let's make a prediction
new_tests = np.array([[63,1,3,145,233,1,0,150,0,2.3,0,0,1], [37,1,2,130,250,0,1,187,0,3.5,0,0,2], [41,0,1,130,204,0,0,172,0,1.4,2,0,2]])
new_tests = new_tests.reshape(len(new_tests), -1)
prediction = knn.predict(new_tests)

# print out details
print( "Accuracy: ", (accuracy*100),'%')

#print( "Predictions:")
#for pred in prediction:
	#if pred == 1:
		#print( pred)
	#else: print( pred)
'''
n_neighbors : int, optional (default = 5)
Number of neighbors to use by default for kneighbors queries.

weights : str or callable, optional (default = ‘uniform’)
weight function used in prediction. Possible values:

‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
[callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.
algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional


Algorithm used to compute the nearest neighbors:

‘ball_tree’ will use BallTree
‘kd_tree’ will use KDTree
‘brute’ will use a brute-force search.
‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
Note: fitting on sparse input will override the setting of this parameter, using brute force.


leaf_size : int, optional (default = 30)
Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.

p : integer, optional (default = 2)
Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

metric : string or callable, default ‘minkowski’
the distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric. See the documentation of the DistanceMetric class for a list of available metrics.

metric_params : dict, optional (default = None)
Additional keyword arguments for the metric function.

n_jobs : int or None, optional (default=None)
The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details. Doesn’t affect fit method.

'''
