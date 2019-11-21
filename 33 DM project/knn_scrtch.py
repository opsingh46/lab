#includes
import numpy as np
from collections import Counter
import pandas as pd
import os
#import PIL.Image

#knn class
class KNeighborsClassifieR(object):

	def __init__(self):
		pass
    #"training" function
	def fit(self, X, y):
		self.X_train = X
		self.y_train = y

    #predict function, output of this function is lis to
	def predict(self, X_test, k=5):
		distances = self.compute_distances(self.X_train, X_test)
		vote_results = []
		for i in range(len(distances)):
			votesOneSample = []
			for j in range(k):
				votesOneSample.append(distances[i][j][1])
			vote_results.append(Counter(votesOneSample).most_common(1)[0][0])
		
		return vote_results
    

	#For each sample and every item in test set algorithm is making tuple in distance list
	#this is how list looks =>> distances = [[[distance, class],[distance, class],[distance, class],[distance, class]]]
	#distances and sort
	def compute_distances(self, X, X_test):
		distances = []
		for i in range(X_test.shape[0]):
			euclidian_distances = np.zeros(X.shape[0])
			oneSampleList = []
			for j in range(len(X)):
				euclidian_distances[j] = np.sqrt(np.sum(np.square(np.array(X_test[i]) - np.array(X[j]))))
				oneSampleList.append([euclidian_distances[j], self.y_train[j]])
			#drugi deo je klasa za element iz train seta za koji smo racunali u ovom krugu
			distances.append(sorted(oneSampleList))
		return distances

#to check how much did algo predict right
def accuracy(y_tes, y_pred):
    correct = 0
    for i in range(len(y_pred)):
        if(y_tes[i] == y_pred[i]):
            correct += 1
    return (correct/len(y_tes))*100


classifier = KNeighborsClassifieR()
    
def run():
    # Importing the dataset
    dataset = pd.read_csv('breast-cancer-wisconsin.data')
    dataset.replace('?', -9999, inplace=True)
    dataset = dataset.applymap(np.int64)
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
#     from sklearn.preprocessing import StandardScaler
#     sc = StandardScaler()
#     X_train = sc.fit_transform(X_train)
#     X_test = sc.transform(X_test)
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("My KNN accuracy: ",accuracy(y_test, y_pred),'%')
run()
# let's make a prediction
new_tests = np.array([[10, 10, 2, 3, 10, 2, 1, 8, 44], [10, 1, 12, 3, 1, 12, 1, 8, 12], [3, 1, 1, 3, 1, 12, 1, 2, 1]])
new_tests = new_tests.reshape(len(new_tests), -1)
prediction = classifier.predict(new_tests)

print( "Predictions:")
for pred in prediction:
	if pred == 2:
		print( pred, "Benign")
	else: print( pred, "Malignant")





