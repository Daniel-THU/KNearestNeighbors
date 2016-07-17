# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 10:09:59 2016

@author: liulei
"""

'''
45 minutes KNN
'''
import numpy as np


dataSet = np.array([ [1.0,1.1,0],[1.0,1.0,1],[0,0,20],[0,0.1,3] ])
labels = np.array(['CAT','CAT','DOG','DOG'])

def KNN(newInput, dataSet,labels, k):

#矩阵里求欧式距离
    numSamples = dataSet.shape[0]    
    diff = np.tile(newInput, (numSamples,1))-dataSet
    squaredDiff = diff ** 2 # squared for the subtract  
    squaredDist = np.sum(squaredDiff, axis = 1) # sum is performed by row  
    distance = squaredDist ** 0.5 
    
    # argsort() returns the indices that would sort an array in a ascending order  
    sortDistIndices = np.argsort(distance)
    
    classCount = {}
    for i in xrange(k):
        voteLabel = labels[sortDistIndices[i]]
        classCount.setdefault(voteLabel,0)        
        #计数
        classCount[voteLabel] = classCount[voteLabel]+1
        print classCount
     
    ranking = sorted(classCount.iteritems(), key = lambda x: x[1])
    ranking.reverse()
    maxIndex = ranking[0][0]
        
#    maxCount =0
#    for key, value in classCount.items():
#        if value > maxCount:
#            maxCount =value
#            maxIndex =key
    return maxIndex



testX = np.array([1.2, 1.0, 1])
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(dataSet, labels)

#testX = np.array([1.2, 1.0, 1])
print knn.predict(np.array([0,0,1]))
