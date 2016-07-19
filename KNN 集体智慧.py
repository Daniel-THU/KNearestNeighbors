# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 13:56:00 2016

@author: liulei
"""

from random import random, randint
import math

def wineprice(rating, age):
    peak_age = rating - 50
    
    price = rating/2
    if age > peak_age:
        price = price*(5-(age-peak_age))
    else:
        price = price*(5*((age+1)/peak_age))
    return price


def wineset1():
    rows=[]
    for i in range(300):
        rating = random()*50 + 50
        age = random()*50
        
        price = wineprice(rating, age)
        price *= (random()*0.4 + 0.8)
        
        rows.append({'input':(rating, age),
                     'result':price})
    return rows

data =  wineset1()

#定义相似度
def eucliden(v1, v2):
    d=0.0
    for i in range(len(v1)):
        d+=(v1[i] - v2[i])**2
    return math.sqrt(d)

def getdistances(data, vec1):
    distancelist = []
    for i in range(len(data)):
        vec2 = data[i]['input']
        distancelist.append((eucliden(vec2,vec1),i))
        distancelist.sort()
    return distancelist


# KNN 算法
def knnestimate(data, vec1, k=5):
    dlist = getdistances(data, vec1)
    avg =0.0
    
    for i in range(k):
        idx = dlist[i][1]
        avg+=data[idx]['result']
    avg = avg/k
    
    return avg

#  加权 KNN

#交叉验证
def dividedata(data, test = 0.05):
    trainset = []
    testset= []
    
    for row in data:
        if random()<test:
            testset.append(row)
        else:
            trainset.append(row)
    return trainset, testset

#预测

def testAlgo(algf, trainset, testset):
    error = 0.0
    for row in testset:
        guess = algf(trainset, row['input'])
        error += (row['result'] - guess)**2
    return error/len(testset)
    
def crossvalidate(algf, data, trials  =100, test = 0.05):
    error =0.0
    for i in range(trials):
        trainset,testset= dividedata(data, test)
        error += testAlgo(algf, trainset,testset)
    return error/trials

# 将新的观测数据加入到 dataset中进行预测
'''
KNN 的优劣
'''
    
    
    
    
    
    
    
