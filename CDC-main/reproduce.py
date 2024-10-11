#!python
# coding=utf-8
'''
=========================================================================
 
 |---| Author: Anoymous
 |---| LastEditors: Anoymous
 |---| Description: Produce the results of the paper
 |---| 
 |---| Copyright (c) 2023 by CDC, All Rights Reserved. 
=========================================================================
'''


import numpy as np
import traceback
from multiview_process import multi_view_processing
from FMvSGC import Effecient_multi_view_clustering
from FSGC import Effecient_clustering
from data_loader import Graph_data
from metrics import evaluate_clustering, baseline_kmeans
from utils import dimension_reduction
from scipy.sparse import csr_matrix
import scipy.io as sio
from graph_filtering import LowPassFilter_sparse
from time import time
import matplotlib.pyplot as plt

from matplotlib.pyplot import MultipleLocator


reproduce_paras = {
        'data type': 'small',
        "data name": "ACM",
        'filter order': 2,
        'alpha': 1000,
        'beta': 0.001
        }

def run() :




            X, A, gnd = Graph_data[reproduce_paras['data type']](reproduce_paras['data name'])
            Dr = None


            H = multi_view_processing(X, A, k=int(reproduce_paras['filter order']), large_graph=(reproduce_paras['data type'] == "large"), Dr=Dr, Type="graph")

            V = min(len(H), 20)
            if V == 20:
                V = 1
            print("This is {} dataset with {} view(s)!".format(reproduce_paras['data name'], V))

            C = len(np.unique(gnd))
            Re_best = [0, 0, 0, 0, 0, 0]
            Paras_best = [0, 0, 0]
            for alpha in [reproduce_paras['alpha']] :
                for beta in [reproduce_paras['beta']] :
                    for m in [C, 10, 30, 50, 70, 100] :

                        if m < C :
                            continue
                        try :
                            time_begin = time()
                            if V != 1 :
                                S, omiga, B = Effecient_multi_view_clustering(H, m, alpha=alpha, beta=beta)
                            else :
                                S, B = Effecient_clustering(H, m, alpha=alpha, beta=beta)
                            if alpha == 1000 and beta == 0.001 and m == C:
                                np.save("./re/rebuttal/S_ACM.npy", S)
                            print("S shape: {}".format(S.shape))  
                            ac, nm, ari, f1, pur = evaluate_clustering(S, gnd)
                            time_end = time()
                            Time = np.fabs(time_end - time_begin)
                            # fg = plt.figure(figsize=(m, 3025))
                            # from mpl_toolkits.axes_grid1 import ImageGrid
                            # # plt.matshow(S.T)
                            # grid = ImageGrid(fg, 111,  # similar to subplot(111)
                            #                  nrows_ncols=(m, 1),  # creates 2x2 grid of axes
                            #                  axes_pad=0.01,  # pad between axes in inch.
                            #                  )
                            # grid[0].imshow(S.T)
                            # plt.show()
                            if ac > Re_best[0] :
                                Re_best[0] = ac
                                Re_best[1] = nm
                                Re_best[2] = ari
                                Re_best[3] = f1
                                Re_best[4] = pur
                                Re_best[5] = Time
                                Paras_best[0] = alpha
                                Paras_best[1] = beta
                                Paras_best[2] = m

                            print(
                                    'Alpha={} bete={} m={} ACC={:.4f}, F1={:.4f}, ARI={:.4f}, NMI={:.4f}, PUR={:.4f} Time={:.4f}'.format(
                                            alpha, beta, m, ac, f1, ari, nm, pur, Time
                                            )
                                    )
                        except Exception as e :
                            print(e)
                            traceback.print_exc()
                            continue

            print(
                "Best re: ACC={:.4f}, NMI={:.4f}, ARI={:.4f}, F1={:.4f}, PUR={:.4f} TIME={:.4f}".format(
                        Re_best[0], Re_best[1], Re_best[2], Re_best[3], Re_best[4], Re_best[5]
                        )
                )
            print("Best paras: alpha={}, beta={}, m={}".format(Paras_best[0], Paras_best[1], Paras_best[2]))


if __name__ == '__main__' :
    run()

