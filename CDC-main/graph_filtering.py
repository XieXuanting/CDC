#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: 2022-10-28 11:31:31
 |---| LastEditors: panern
 |---| LastEditTime: 2022-10-31 13:06:27
 |---| Description: 
 |---| 
 |---| Copyright (c) 2022 by WuJi, All Rights Reserved. 
 
=========================================================================
'''
import numpy as np
from utils import normalize_matrix
from scipy.sparse import csr_matrix


'''
=========================================================================
 
 |---| description: low-pass filter for small attributed graphs
 |---| param {*} X: raw feature matrix
 |---| param {*} A: the adjacency matrix
 |---| param {*} k1: the order of the filter
 |---| param {*} p: the parameter of the filter
 |---| return {*} H_low: the filtered feature matrix
 
=========================================================================
'''
def LowPassFilter(X, A, k1, p=0.5):
    
    # A = normalize_matrix(A+I)
    I = np.eye(A.Ahape[0])
    S = A + I
    S = normalize_matrix(S)
    
    #laplacian matrix
    L = I - S
    
    #filtered matrix with order k1
    FM = I - p * L
    FMk = np.linalg.matrix_power(FM, k1)
    H_low = FMk.dot(X)
    
    return H_low


'''
=========================================================================
 
 |---| description: low-pass filter for large attributed graphs with sparse adjacency matrix
 |---| param {*} X: raw feature matrix
 |---| param {*} A: the adjacency matrix
 |---| param {*} Dr: the degree vector
 |---| param {*} k1: the order of the filter
 |---| param {*} p: the parameter of the filter
 |---| return {*} H_low: the filtered feature matrix
 
=========================================================================
'''
def LowPassFilter_sparse(X, A, Dr, k1, p=0.5):
    N = X.shape[0]
    row_I = np.arange(N)
    col_I = np.arange(N)
    data_I = np.ones(N)
    
    # Sparse identity matrix
    I = csr_matrix((data_I, (row_I, col_I)), shape=(N, N))

    # Sparse degree matrix
    D = np.array(Dr)
    D = D + 1
    D = np.power(D, -0.5)
    D = csr_matrix((D, (row_I, col_I)), shape=(N, N))

    # self-loop and normalize
    S = A + I
    S = csr_matrix(S)
    # normalize
    S = D * S * D
    
    # filtered kernel
    # I - p * L_S = I - p * (I - S) = (1-p)I + pS
    F_M = (1-p) * I + p * S

    # filtered matrix with order k1
    H_low = F_M * X
    f_order = k1-1
    while f_order > 0:
        H_low = F_M * H_low
        f_order -= 1
    # print("Filtering Done!")
    return H_low

