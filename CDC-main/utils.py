#!python
# coding=utf-8
'''
=========================================================================

 |---| Author: Wuji
 |---| Date: 2022-11-18 18:59:28
 |---| LastEditors: panern
 |---| LastEditTime: 2022-12-02 17:04:58
 |---| Description:
 |---|
 |---| Copyright (c) 2022 by WuJi, All Rights Reserved.

=========================================================================
'''
# !python
# coding=utf-8
'''
=========================================================================

 |---| Author: Wuji
 |---| Date: 2022-10-28 11:39:58
 |---| LastEditors: panern
 |---| LastEditTime: 2022-10-31 13:07:52
 |---| Description: 
 |---| 
 |---| Copyright (c) 2022 by WuJi, All Rights Reserved. 

=========================================================================
'''

import numpy as np
from sklearn.decomposition import PCA, SparsePCA
from tqdm import  tqdm
'''
=========================================================================

 |---| description: normalize the adjacency matrix
 |---| param {*} A: adjacency matrix
 |---| return {*} A: normalized adjacency matrix

=========================================================================
'''


def normalize_matrix(A):
    D = np.sum(A, axis=1)

    D = np.power(D, -0.5)
    D[np.isinf(D)] = 0
    D[np.isnan(D)] = 0
    D = np.diagflat(D)
    A = D.dot(A).dot(D)

    return A


'''
=========================================================================

 |---| description: normalize the feature matrix
 |---| param {*} X: feature matrix
 |---| return {*} X: normalized feature matrix

=========================================================================
'''


def normalize_feature(X):
    X_new = [x / np.linalg.norm(x) for x in X]
    X_new = np.array(X_new)
    return X_new


'''
=========================================================================

 |---| description: normalize the feature matrix by max-min
 |---| param {*} X: feature matrix
 |---| return {*} X: normalized feature matrix

=========================================================================
'''


def normalize_F(X):
    X_new = (X - np.min(X)) / (np.max(X) - np.min(X))
    return X_new


'''
=========================================================================

 |---| description: PCA
 |---| param {*} X: feature matrix with dims=NxD
 |---| param {*} dim: dimension of the feature matrix after PCA
 |---| return {*} X_new: feature matrix with dims=Nxd after PCA

=========================================================================
'''


def dimension_reduction(X, dim=64, idd=0, dt="ACM"):
    try:
        X_new = np.load("./feature_matrix/{}_X_{}.npy".format(dt, idd))
    except Exception:
        try:
            # print(8 / 0)
            pca = PCA(n_components=dim, random_state=12345)
            X_new = pca.fit_transform(X)
        except:  # !! this for extra large-scale data
            done = 0
            batch_size = 1000000
            N = X.shape[0]
            while done <= 0:
                try:
                    X_new = []
                    # batch_size = 1000000
                    steps = int(N / batch_size) + 1
                    for step in range(steps):
                        try:
                            x_new = np.load("./feature_matrix/{}_X_{}_{}.npy".format(dt, idd, step))
                        except Exception:
                            x = X[step * batch_size: min((step + 1) * batch_size, N)]
                            pca = PCA(n_components=dim, random_state=12345)
                            x_new = pca.fit_transform(x)
                        finally:
                            assert x_new != None
                            X_new.append(x_new)
                    done += 1

                except:
                    batch_size = batch_size / 2
                    pass

    return X_new


def dimension_reduction_sparse(X, dim=64):
    pca = SparsePCA(n_components=dim, random_state=12345)
    X_new = pca.fit_transform(X)

    return X_new


if __name__ == "__main__":
    from data_loader import single_view_graphs

    X, _, _, _ = single_view_graphs("Products")
    X = dimension_reduction(X[0], dim=64, idd=0, dt="Products")
    print(X.shape)