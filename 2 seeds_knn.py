# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 13:54:35 2016

@author: liulei


seeds_knn


"""
import numpy as np

def load_dataset(dataset_name):
    '''
   获取数据 并分类返回data多维数组 和label一维数组
    Returns
    -------
    data : numpy ndarray
    labels : list of str
    '''
    data = []
    labels = []
    with open('/Users/liulei/Desktop/机器学习系统设计/1400OS_Code/1400OS_02_Codes/data/{0}.tsv'.format(dataset_name)) as ifile:
        for line in ifile:
            tokens = line.strip().split('\t')
            data.append([float(tk) for tk in tokens[:-1]])
            labels.append(tokens[-1])
    data = np.array(data)
    labels = np.array(labels)
    return data, labels


"""
KNN 算法
"""
def learn_model(k, features, labels):
    return k, features.copy(), labels.copy()


def plurality(xs):
    from collections import defaultdict
    
    