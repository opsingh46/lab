import csv
import random
from math import sqrt


def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rt') as csvfile:
	    lines = csv.reader(csvfile,delimiter=',')
	    next(lines)
	    
	               
	    dataset = list(lines)
	    for x in range(len(dataset)-1):
	        for y in range(13):
	            dataset[x][y] = float(dataset[x][y])
	           
	        if random.random() < split:
	            trainingSet.append(dataset[x])
	        else:
	            testSet.append(dataset[x])
	


import math
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return sqrt(distance)


import operator 
def getNeighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key=operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors



import operator
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]



def getAccuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] in predictions[x]: 
			correct = correct + 1
			
	return (correct/float(len(testSet))*100) 

def main():
	# prepare data
	trainingSet=[]
	testSet=[]
	split = 0.73
	loadDataset('C:\\Users\\prafu\\Desktop\\data mining lab\\fsfsd\\heart.csv', split, trainingSet, testSet)
	
	# generate predictions
	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) +'%')
	
	
main()
