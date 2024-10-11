#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: 2022-11-01 15:48:26
 |---| LastEditors: panern
 |---| LastEditTime: 2022-11-01 15:48:33
 |---| Description: efficient single-view clustering
 
 |---| 
 |---| Copyright (c) 2022 by WuJi, All Rights Reserved. 
 
=========================================================================
'''
import numpy as np
from scipy.linalg import solve_sylvester
import warnings




warnings.filterwarnings('ignore')



def Effecient_clustering(H, B, alpha=1, beta=1, epochs=100, threshold=1e-5):
    Im = np.eye(B.shape[0])
    loss_last = 1e16

    for epoch in range(epochs):
        BBt = B.dot(B.T)
        BHt = B.dot(H.T)

        #update Si
        tmp1 = BBt + (alpha + beta) * Im
        tmp2 = np.linalg.inv(tmp1).dot(BHt)
        S = tmp2 * (1 + beta)
            
        #Compute the value of objective function
        loss_SE = np.linalg.norm(H.T - B.T.dot(S), 'fro') ** 2
        loss_L2 = np.linalg.norm(S, 'fro') ** 2
        loss_inv = np.linalg.norm(B.dot(H.T) - S, 'fro') ** 2
        loss_total = loss_SE + alpha * loss_L2 + beta * loss_inv
        
        if loss_last - loss_total < threshold * loss_last:
            break
        else:
            loss_last = loss_total
            
        #update B
        SSt = S.dot(S.T)
        HtH = H.T.dot(H)
        SH = S.dot(H)
    
        B = solve_sylvester(SSt, beta * HtH, SH*(1+beta))
        
        
        
    return S, B



def Effecient_clustering_selection(H, B, alpha=1, beta=1, epochs=100, threshold=1e-5):
    Im = np.eye(B.shape[0])
    BBt = B.dot(B.T)
    BHt = B.dot(H.T)
    #update S
    tmp1 = BBt + (alpha + beta) * Im
    tmp2 = np.linalg.inv(tmp1).dot(BHt)
    S = tmp2 * (1 + beta)
    return S, B

def Effecient_clustering_Nosim_Fixed(H, B, beta=1, epochs=100, threshold=1e-5):
    Im = np.eye(B.shape[0])
    BBt = B.dot(B.T)
    BHt = B.dot(H.T)
        
    #update S
    tmp1 = BBt + beta * Im
    S = np.linalg.inv(tmp1).dot(BHt)
   
    
    return S, B

def Effecient_clustering_Nosim(H, B, beta=1, epochs=100, threshold=1e-5):
    Im = np.eye(B.shape[0])
    for epoch in range(epochs):
        BBt = B.dot(B.T)
        BHt = B.dot(H.T)

        #update Si
        tmp1 = BBt + beta * Im
        S = np.linalg.inv(tmp1).dot(BHt)

        #Compute the value of objective function
        loss_SE = np.linalg.norm(H.T - B.T.dot(S), 'fro') ** 2
        loss_L2 = np.linalg.norm(S, 'fro') ** 2
        loss_total = loss_SE  + beta * loss_L2
        if loss_last - loss_total < threshold * loss_last:
            break
        else:
            loss_last = loss_total
            
        #update B
        SSt = S.dot(S.T)
        SH = S.dot(H)
        B = np.linalg.inv(SSt).dot(SH)
    
    return S, B

