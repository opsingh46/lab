#includes
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
#to check how much did algo predict right
def accuracy(y_tes, y_pred):
    correct = 0
    for i in range(len(y_pred)):
        if(y_tes[i] == y_pred[i]):
            correct += 1
    return (correct/len(y_tes))*100

def skLearnKNN():
    # Importing the dataset
    dataset = pd.read_csv('breast-cancer-wisconsin.data')
    dataset.replace('?', 0, inplace=True)
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
    print("Sklearn accuracy: ", accuracy(y_test, y_pred),'%')

skLearnKNN()


# let's make a prediction
new_tests = np.array([[10, 10, 2, 3, 10, 2, 1, 8, 44], [10, 1, 12, 3, 1, 12, 1, 8, 12], [3, 1, 1, 3, 1, 12, 1, 2, 1]])
new_tests = new_tests.reshape(len(new_tests), -1)
prediction = classifier.predict(new_tests)

print( "Predictions:")
for pred in prediction:
	if pred == 2:
		print( pred, "Benign")
	else: print( pred, "Malignant")


