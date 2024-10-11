#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: 2022-11-24 17:04:08
 |---| LastEditors: panern
 |---| LastEditTime: 2022-12-07 12:25:28
 |---| Description: 
 |---| 
 |---| Copyright (c) 2022 by WuJi, All Rights Reserved. 
 
=========================================================================
'''

from sklearn.neighbors import kneighbors_graph


def knn_graph(X, k=6): #~ K should be k+1, because the first neighbor is itself
    A = kneighbors_graph(X, k, mode='connectivity', include_self=True, n_jobs=-1)
    return A

if __name__ == '__main__':
    from data_loader import datasets
    X, A, gnd = datasets['non-graph']("YTF-31")
    knn_graph(X, k=6)
    pass