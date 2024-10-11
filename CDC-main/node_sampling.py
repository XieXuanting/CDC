#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: 2022-10-28 11:31:43
 |---| LastEditors: panern
 |---| LastEditTime: 2022-11-01 14:02:02
 |---| Description:
 |---| 
 |---| Copyright (c) 2022 by WuJi, All Rights Reserved. 
 
=========================================================================
'''

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

'''
=========================================================================
 
 |---| description: sampling the anchor points by k-means
 |---| param {*} X: feature matrix
 |---| param {*} anchor_num: the number of anchor points
 |---| return {*} inds: the index of anchor points
 
=========================================================================
'''
def sampling_kmeans(X, anchor_num=50):
    
    inds = []
    for x in X:
        KM = KMeans(n_clusters=anchor_num, random_state=1234).fit(x)
        anchor = KM.cluster_centers_
        inds.append(anchor)
    
    return inds

'''
=========================================================================
 
 |---| description: sampling the anchor points by MiniBatchKMeans
 |---| param {*} X: feature matrix
 |---| param {*} anchor_num: the number of anchor points
 |---| return {*} inds: the index of anchor points
 
=========================================================================
'''
def sampling_minikmeans(X, anchor_num=50):
    inds = []
    for x in X :
        KM = MiniBatchKMeans(init="k-means++", n_clusters=anchor_num, random_state=1234, batch_size=1000).fit(x)
        anchor = KM.cluster_centers_
        inds.append(anchor)

    return inds


def sampling_knp(X, anchor_num=50):
    from scipy.spatial.distance import euclidean
    inds = []
    for x in X:
        ind = []
        KM = KMeans(n_clusters=anchor_num, random_state=1234).fit(x)
        labels = KM.labels_
        centroids = KM.cluster_centers_
        numLabels = KM.n_clusters
        C_dis = [[] for i in range(numLabels)]
        C_index = [[] for i in range(numLabels)]
        for nn in range(x.shape[0]):
            for c_num in range(numLabels):
                if(labels[nn] == c_num):
                    C_dis[c_num].append(euclidean(x[nn], centroids[c_num]))
                    C_index[c_num].append(nn)
        
        
        for c_num in range(numLabels):
            anchorInd = np.argmin(C_dis[c_num])
            ind.append(C_index[c_num][anchorInd])
    
        ind = np.array(ind)
        assert ind.shape[0] == (anchor_num)
        inds.append(x[ind])
    
    return inds


def sampling_random(X, anchor_num=50):
    inds = []
    for x in X:
        N = x.shape[0]
        ind = np.random.choice([i for i in range(N)], anchor_num)
        inds.append(x[ind])
        
    return inds






def sampling_SFMC(H,  m=50):
    inds = []
    X = H[0]
    if len(H) > 1:
        for x in X[1:]:
            X = np.concatenate((X,x),axis=1)
    Xmin = np.min(X)
    X = X - Xmin
    X = np.sum(X, axis=1)
   
    
    ind = []
    m_tmp = m
    while m_tmp > 0:
        Xmax = np.max(X)
        X = X / Xmax
        ind_max = np.argmax(X)
        ind.append(ind_max)
        X = np.multiply(X, 1-X)
        m_tmp -= 1
        
    assert len(ind) == m
    for h in H:
        inds.append(h[ind])
    return inds



'''
=========================================================================
 
 |---| description: this function is used to find the lower bound of the value in the array p
 |---| param {*} p: the probability array
 |---| param {*} rd: the random value
 |---| return {*}
    l: position of the lower bound
=========================================================================
'''
def lower_bound(p, rd):
    l = 0
    r = len(p) - 1
    while(l < r):

        mid = (l + r) // 2
        if(p[mid] > rd):
            r = mid
        else:
            l = mid + 1

    return l


'''
=========================================================================
 
 |---| description: this function is used to sample the nodes
 |---| param {*} A: the adjacency matrix
 |---| param {*} m: the number of nodes to be sampled
 |---| param {*} alpha: the parameter of the probability, fixed to 4
 |---| return {*}
    ind: the index of the sampled nodes
=========================================================================
'''
def node_sampling(Drs, m, alpha=4):
    D = Drs[0]
    if len(Drs) > 1:
        for dr in Drs[1:]:
            D = D + dr
    D = D**alpha
    Max = np.max(D)
    Min = np.min(D)
    D = (D- Min) / (Max - Min)
    tot = np.sum(D)
    p = D / tot
    for i in range(len(p) - 1):
        p[i + 1] = p[i + 1] + p[i]
    inds = []
    vis = [0] * len(D)
    while(m):
        while(1):
            # sd = 1
            # np.random.seed(m+sd)
            rd = np.random.rand()
            pos = lower_bound(p, rd)
            if(vis[pos] == 1):
                # sd += 1000
                continue
            else:
                vis[pos] = 1
                inds.append(pos)
                m = m - 1
                break
    return inds


def sampling_NodeDegree(X, Drs, anchor_num=50):
    inds = []
    for x in X:
        ind = node_sampling(Drs, anchor_num)
        indx = x[ind]
        inds.append(indx)
    
    return inds
