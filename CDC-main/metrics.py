#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Wuji
 |---| Date: 2022-11-19 11:31:29
 |---| LastEditors: panern
 |---| LastEditTime: 2022-11-19 11:31:43
 |---| Description: 
 |---| 
 |---| Copyright (c) 2022 by WuJi, All Rights Reserved. 
 
=========================================================================
'''
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans, MiniBatchKMeans
from Metrics_O2MAC import metric_all


'''
=========================================================================
 
 |---| description: Evaluate the clustering performance
 |---| param {*} S: the anchor graph
 |---| param {*} gnd: the ground truth
 |---| return {*}
    acc: the accuracy
    nmi: the normalized mutual information
    ari: the adjusted rand index
    f1: the f1 score
    pur: the purity
=========================================================================
'''
def evaluate_clustering(S, gnd):
    
    num_labels = len(np.unique(gnd))
    
    #! Y choose 30? Because the 31 is close to sqrt(1000). Th complexity of SVD is O(d^2N),I hope it's less than O(1000N)
    #! If number of labels is large, it's bearable that Tp=num_labels.
    Tp = max(num_labels, 30) 
    
    if S.shape[0] <= Tp:
        u, s, v = np.linalg.svd(
            S, full_matrices=False
            )
    else:
        u, s, v = sp.linalg.svds(
            S, k=Tp, which='LM'
            )

    #! Perform MiniBatchKMeans to reduce the space cost while keeping the accuracy.
    if S.shape[1] > 20000:
        kmeans = MiniBatchKMeans(init="k-means++", batch_size=1000, n_clusters=num_labels, random_state=20).fit(v.T)
    elif S.shape[1] > 1e6:
        kmeans = MiniBatchKMeans(init="k-means++", batch_size=500, n_clusters=num_labels, random_state=20).fit(v.T)
    else:
        #! Perform KMeans on smaller graph.
        kmeans = KMeans(init="k-means++",n_clusters=num_labels, random_state=20).fit(v.T)
        
    predict_labels = kmeans.predict(v.T)
    
    re = metric_all.clustering_metrics(predict_labels, gnd)
    acc, nmi, ari, f1, pur = re.evaluationClusterModelFromLabel()
        
    

    return acc, nmi, ari, f1, pur




def baseline_kmeans(X, gnd):
    
    num_labels = len(np.unique(gnd))
    
    #! Perform MiniBatchKMeans to reduce the space cost while keeping the accuracy.
    if X.shape[1] > 20000:
        kmeans = MiniBatchKMeans(init="k-means++", batch_size=1000, n_clusters=num_labels, random_state=20).fit(X)
    else:
        #! Perform KMeans on smaller graph.
        kmeans = KMeans(init="k-means++",n_clusters=num_labels, random_state=20).fit(X)
        
    predict_labels = kmeans.predict(X)
    
    re = metric_all.clustering_metrics(predict_labels, gnd)
    acc, nmi, ari, f1, pur = re.evaluationClusterModelFromLabel()

    return acc, nmi, ari, f1, pur
