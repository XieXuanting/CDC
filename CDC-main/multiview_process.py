#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: 2022-11-19 11:38:33
 |---| LastEditors: panern
 |---| LastEditTime: 2022-11-19 20:30:59
 |---| Description: 
 |---| 
 |---| Copyright (c) 2022 by WuJi, All Rights Reserved. 
 
=========================================================================
'''
#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: 2022-11-19 11:38:33
 |---| LastEditors: panern
 |---| LastEditTime: 2022-11-19 11:38:37
 |---| Description: 
 |---| 
 |---| Copyright (c) 2022 by WuJi, All Rights Reserved. 
 
=========================================================================
'''
from utils import dimension_reduction
import numpy as np
from graph_filtering import LowPassFilter_sparse



'''
=========================================================================
 
 |---| description: construct the data matrix of each view
 |---| param {*} X: the attribute matrix
 |---| param {*} A: the adjacency matrix
 |---| param {*} k: the filter order
 |---| return {*} H: the filtered multi-view data matrix
=========================================================================
'''

def multi_view_processing(X, A, Dr, k=2, dims=100):
    
    
    try:
        
        lenX = len(X)
        lenA = len(A)
        H = []

        if lenX > lenA:
            for x in X:
                if x.shape[1] >= dims:
                    x = dimension_reduction(x, dims)
                Htmp = LowPassFilter_sparse(x, A[0], Dr[0], k1=k)
                H.append(Htmp)
        else:
            for i in range(lenA-lenX):
                X.append(X[0])
            for (x, a, d) in zip(X, A, Dr):
                if x.shape[1] >= dims:
                    x = dimension_reduction(x, dims)

                Htmp = LowPassFilter_sparse(x, a, d, k1=k)
                H.append(Htmp)
        
        assert len(H) == max(lenX, lenA)
        return H
        
    except Exception as e:
        print("Error: {}".format(e))
        return
    