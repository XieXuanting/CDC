#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: 2022-11-01 15:48:26
 |---| LastEditors: panern
 |---| LastEditTime: 2022-11-01 15:48:33
 |---| Description: 
 
 |---| 
 |---| Copyright (c) 2022 by WuJi, All Rights Reserved. 
 
=========================================================================
'''
import numpy as np
from scipy.linalg import solve_sylvester
import warnings


warnings.filterwarnings('ignore')


'''
=========================================================================
 
 |---| description: Compute the anchor graph and adaptive anchors
 |---| param {*} H: the list of the data matrix of each view
 |---| param {*} anchor_num: the number of anchors
 |---| param {*} alpha: the parameter of the balance between the SE and BG
 |---| param {*} beta: the parameter of the balance between the SE and L2
 |---| param {*} eps: to avoid the denominator is zero
 |---| param {*} threshold: the threshold of the convergence of the algorithm
 |---| return {*} 
    B: the list of the anchors of each view
    S: the anchor graph
=========================================================================
'''
def Effecient_multi_view_clustering(H, B, alpha=1, beta=1, eps = 1e-5,  threshold = 1e-5, epochs = 100):
    
    
    V = len(H)

    Im = np.eye(B[0].shape[0])
    
    #! initialize the weight of each view
    omiga = [ 1/V ] * V
    # B = B1.copy()
    # # initialize anchors
    # B = []
    # for v in range(V):
    #     if H[v].shape[0] > 30000:
    #         B_tmp = sampling_minikmeans(H[v], anchor_num)[0]
    #         B.append(B_tmp)
    #     else:
    #         B_tmp = sampling_kmeans(H[v], anchor_num)[0]
    #         B.append(B_tmp)
        
    
    loss_last = 1e16
    for epoch in range(epochs):
        
    
        BBt = []
        BHt = []
        for v in range(V):
            BBt_v = B[v].dot(B[v].T)
            BHt_v = B[v].dot(H[v].T)

            BBt.append(BBt_v)
            BHt.append(BHt_v)
            
        
            
        
        #update S
        tmp1 = 0
        tmp2 = 0
        for v in range(V):
            omiga_v_square = omiga[v] ** 2
            tmp1 += omiga_v_square * (BBt[v]+alpha*Im)
            tmp2 += omiga_v_square * BHt[v]
            
        S = np.linalg.inv(tmp1+ beta * Im).dot(tmp2*(1+alpha))
        
        #! Compute the value of objective function
        loss_view = 0
        for v in range(V):
            loss_SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            loss_BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
            omiga_v_square = omiga[v] ** 2
            loss_view += omiga_v_square * (loss_SE + alpha * loss_BG)
            
        loss_L2 = np.linalg.norm(S, 'fro') ** 2
        loss_total = loss_view + beta * loss_L2 
        
        # ! observe the convergence of the algorithm
        # print('Epoch: {}, loss: {}'.format(epoch+1, loss_total))
        
        #! break the loop if the loss converges
        if loss_last - loss_total < threshold * loss_last:
            break
        else:
            loss_last = loss_total
            
        #! update B
        SSt = S.dot(S.T)
        for v in range(V):
            HtH_v = H[v].T.dot(H[v])
            SH_v = S.dot(H[v])
            B[v] = solve_sylvester(SSt, alpha * HtH_v, SH_v*(1+alpha))
        
        
        #! update omiga (called lambda in paper)
        Const_loss = np.zeros(V)
        for v in range(V):
            SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
            Const_loss[v] += SE + alpha * BG
        
        Total = [1/CL for CL in Const_loss]
        Total_sum = np.array(Total).sum()
        for v in range(V):
            omiga[v] = (Total[v]) / (Total_sum + eps)
        
        # Total = [CL for CL in Const_loss]
        # Total_sum = np.array(Total).sum()
        # for v in range(V):
        #     omiga[v] = (Total[v]) / (Total_sum + eps)
        
        
    return S, omiga, B



def Effecient_multi_view_clustering_selection(H, B, alpha=1, beta=1, eps = 1e-5,  threshold = 1e-5, epochs = 100):
    
    
    V = len(H)
    
    epochs = 100
    
    Im = np.eye(B[0].shape[0])
    
    #! initialize the weight of each view
    omiga = [ 1/V ] * V
    
   
    BBt = []
    BHt = []
    
    for v in range(V):
        BBt_v = B[v].dot(B[v].T)
        BHt_v = B[v].dot(H[v].T)

        BBt.append(BBt_v)
        BHt.append(BHt_v)
            
            
    loss_last = 1e16
    
    
    
    for epoch in range(epochs):
        
        #update S
        tmp1 = 0
        tmp2 = 0
        for v in range(V):
            omiga_v_square = omiga[v] ** 2
            tmp1 += omiga_v_square * (BBt[v]+alpha*Im)
            tmp2 += omiga_v_square * BHt[v]
            
        S = np.linalg.inv(tmp1+ beta * Im).dot(tmp2*(1+alpha))
        
        #! Compute the value of objective function
        loss_view = 0
        for v in range(V):
            loss_SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            loss_BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
            omiga_v_square = omiga[v] ** 2
            loss_view += omiga_v_square * (loss_SE + alpha * loss_BG)
            
        loss_L2 = np.linalg.norm(S, 'fro') ** 2
        loss_total = loss_view + beta * loss_L2 
        
        #! observe the convergence of the algorithm
        # print('Epoch: {}, loss: {}'.format(epoch+1, loss_total))
        
        #! break the loop if the loss converges
        if loss_last - loss_total < threshold * loss_last:
            break
        else:
            loss_last = loss_total
            
        
        #! update omiga (called lambda in paper)
        Const_loss = np.zeros(V)
        for v in range(V):
            SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            BG = np.linalg.norm(BHt[v] - S, 'fro') ** 2
            Const_loss[v] += SE + alpha * BG
        
        Total = [1/CL for CL in Const_loss]
        Total_sum = np.array(Total).sum()
        for v in range(V):
            omiga[v] = (Total[v]) / (Total_sum + eps)
        
     
        
        
    return S, omiga, B


def Effecient_multi_view_clustering_NoSim_Fixed(H, B,  beta=1, eps = 1e-5,  threshold = 1e-5, epochs = 100):
    
    
    V = len(H)
    
    epochs = 100
    
    Im = np.eye(B[0].shape[0])
    
    #! initialize the weight of each view
    omiga = [ 1/V ] * V
    
   
    BBt = []
    BHt = []
    
    for v in range(V):
        BBt_v = B[v].dot(B[v].T)
        BHt_v = B[v].dot(H[v].T)

        BBt.append(BBt_v)
        BHt.append(BHt_v)
            
            
    loss_last = 1e16
    
    
    
    for epoch in range(epochs):
        
        #update S
        tmp1 = 0
        tmp2 = 0
        for v in range(V):
            omiga_v_square = omiga[v] ** 2
            tmp1 += omiga_v_square * (BBt[v])
            tmp2 += omiga_v_square * BHt[v]
            
        S = np.linalg.inv(tmp1+ beta * Im).dot(tmp2)
        
        #! Compute the value of objective function
        loss_view = 0
        for v in range(V):
            loss_SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            omiga_v_square = omiga[v] ** 2
            loss_view += omiga_v_square * loss_SE
            
        loss_L2 = np.linalg.norm(S, 'fro') ** 2
        
        loss_total = loss_view + beta * loss_L2 
        
        #! observe the convergence of the algorithm
        # print('Epoch: {}, loss: {}'.format(epoch+1, loss_total))
        
        #! break the loop if the loss converges
        if loss_last - loss_total < threshold * loss_last:
            break
        else:
            loss_last = loss_total
            
        
        #! update omiga (called lambda in paper)
        Const_loss = np.zeros(V)
        for v in range(V):
            SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            Const_loss[v] += SE
        
        Total = [1/CL for CL in Const_loss]
        Total_sum = np.array(Total).sum()
        for v in range(V):
            omiga[v] = (Total[v]) / (Total_sum + eps)
        
     
        
        
    return S, omiga, B


def Effecient_multi_view_clustering_NoSim(H, B,  beta=1, eps = 1e-5,  threshold = 1e-5, epochs = 100):
    
    
    V = len(H)
    
    Im = np.eye(B[0].shape[0])
    
    #! initialize the weight of each view
    omiga = [ 1/V ] * V
    
    # # initialize anchors
    # B = []
    # for v in range(V):
    #     if H[v].shape[0] > 30000:
    #         B_tmp = sampling_minikmeans(H[v], anchor_num)[0]
    #         B.append(B_tmp)
    #     else:
    #         B_tmp = sampling_kmeans(H[v], anchor_num)[0]
    #         B.append(B_tmp)
        
    
    loss_last = 1e16
    for epoch in range(epochs):
        
    
        BBt = []
        BHt = []
        for v in range(V):
            BBt_v = B[v].dot(B[v].T)
            BHt_v = B[v].dot(H[v].T)

            BBt.append(BBt_v)
            BHt.append(BHt_v)
            
        
            
        
        #update S
        tmp1 = 0
        tmp2 = 0
        for v in range(V):
            omiga_v_square = omiga[v] ** 2
            tmp1 += omiga_v_square * BBt[v]
            tmp2 += omiga_v_square * BHt[v]
            
        S = np.linalg.inv(tmp1+ beta * Im).dot(tmp2)
        
        #! Compute the value of objective function
        loss_view = 0
        for v in range(V):
            loss_SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            omiga_v_square = omiga[v] ** 2
            loss_view += omiga_v_square * loss_SE 
            
        loss_L2 = np.linalg.norm(S, 'fro') ** 2
        loss_total = loss_view + beta * loss_L2 
        
        # ! observe the convergence of the algorithm
        # print('Epoch: {}, loss: {}'.format(epoch+1, loss_total))
        
        #! break the loop if the loss converges
        if loss_last - loss_total < threshold * loss_last:
            break
        else:
            loss_last = loss_total
            
        #! update B
        SSt = S.dot(S.T)
        for v in range(V):
            HtH_v = H[v].T.dot(H[v])
            SH_v = S.dot(H[v])
            B[v] = np.linalg.inv(HtH_v).dot(SH_v)
        
        
        #! update omiga (called lambda in paper)
        Const_loss = np.zeros(V)
        for v in range(V):
            SE = np.linalg.norm(H[v].T - B[v].T.dot(S), 'fro') ** 2
            Const_loss[v] += SE 
        
        Total = [1/CL for CL in Const_loss]
        Total_sum = np.array(Total).sum()
        for v in range(V):
            omiga[v] = (Total[v]) / (Total_sum + eps)
        
        # Total = [CL for CL in Const_loss]
        # Total_sum = np.array(Total).sum()
        # for v in range(V):
        #     omiga[v] = (Total[v]) / (Total_sum + eps)
        
        
    return S, omiga, B



