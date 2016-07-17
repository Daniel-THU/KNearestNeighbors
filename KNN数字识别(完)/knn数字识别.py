# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:27:10 2016

@author: liulei
"""
import numpy as np
import os
import time
import operator

#  KNN核心算法
def classify(inputPoint,dataSet,labels,k):
  dataSetSize = dataSet.shape[0]	 #已知分类的数据集（训练集）的行数
  #先tile函数将输入点拓展成与训练集相同维数的矩阵，再计算欧氏距离
  diffMat = np.tile(inputPoint,(dataSetSize,1))-dataSet  #样本与训练集的差值矩阵
  sqDiffMat = diffMat ** 2					#差值矩阵平方
  sqDistances = sqDiffMat.sum(axis=1)		 #计算每一行上元素的和
  distances = sqDistances ** 0.5			  #开方得到欧拉距离矩阵
  sortedDistIndicies = distances.argsort()	#argsort 返回由小到大的下标值，按distances中元素进行升序排序后得到的对应下标的列表
  #选择距离最小的k个点
  classCount = {}  #字典的用法
  for i in range(k):
    voteIlabel = labels[ sortedDistIndicies[i] ]
    classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
  #classCount字典的第2个元素（即按k个点中，类别出现的次数）从大到小排序，选取最大的那个 为所求
  # 字典输出的是预测的label类别
  #items() 以列表返回可遍历的元组
  sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse = True) #从大到小，key：用列表元素的某个属性和函数进行作为关键字
  return sortedClassCount[0][0]

#数据操作
#先把文本中矩阵格式的数字转化为一个向量格式
def img2vector(filename):
    returnVect=[]
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect.append(int(lineStr[j]))
    return returnVect
    
# 训练集  文件名的第一个数字是 此文本表示的实际数字
def classnumCut(fileName):
    fileStr =fileName.split('.')[0]
    classNumStr=int(fileStr.split('_')[0])
    return classNumStr

#训练集向量 
def trainingDataSet():
    hwLabels=[]
    traingingFileList=os.listdir('/Users/liulei/Desktop/digits/trainingDigits') #获取目录内容
    m=len(traingingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr= traingingFileList[i]
        hwLabels.append(classnumCut(fileNameStr))
        trainingMat[i,:]=img2vector('/Users/liulei/Desktop/digits/trainingDigits/%s' % fileNameStr)
    return hwLabels, trainingMat
    
def handwritingTest():
    hwLabels, trainingMat=trainingDataSet() #构建训练集
    testFileList =os.listdir('/Users/liulei/Desktop/digits/testDigits')  #获取测试集
    errorCount =0.0
    mTest=len(testFileList)
    t1=time.time()
    
    for i in range(mTest):
        fileNameStr=testFileList[i]
        classNumStr=classnumCut(fileNameStr)
        vectorUnderTest = img2vector('/Users/liulei/Desktop/digits/testDigits/%s' % fileNameStr)
        classifierResult = classify(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
    
    if (classifierResult != classNumStr): 
        errorCount += 1.0
    print "\nthe total number of tests is: %d" % mTest			   #输出测试总样本数
    print "the total number of errors is: %d" % errorCount		   #输出测试错误样本数
    print "the total error rate is: %f" % (errorCount/float(mTest))  #输出错误率
    t2 = time.time()
    print "Cost time: %.2fmin, %.4fs."%((t2-t1)//60,(t2-t1)%60)	  #测试耗时

if __name__ == "__main__":
  handwritingTest()
        