import numpy as np
from data_loader import datasets
from multiview_process import multi_view_processing
from node_sampling import sampling_kmeans, sampling_minikmeans
from FMvSGC import Effecient_multi_view_clustering
from FSGC import Effecient_clustering
from metrics import evaluate_clustering
import time


class CDC():
    
    def __init__(self, dataname="Citeseer"):
        self.datasets = datasets
        self.alphas = [0.001, 1, 10, 100, 1000, 10000]
        self.betas = [0.001, 1, 10, 100, 1000, 10000]
        self.data_name = dataname
        self.data_type = 'graph data'
        self.max_dims = 100
        self.algorithms = [Effecient_clustering, Effecient_multi_view_clustering]
        self.k = 2
        
        self.best_paras = {
            "alpha": 0,
            "beta": 0,
            "m": 0,
        }
        self.best_re = {
            "ACC": 0,
            "NMI": 0,
            "ARI": 0,
            "F1": 0,
            "PUR": 0,
            "Time": 0
        }
        
    def loadData(self):
        if self.data_name in ["ACM", "DBLP"]:
            X, As, Drs, gnd = self.datasets["multi-relational"](self.data_name)
            self.data_type = 'multi-relational '+ self.data_type
        elif self.data_name in ["Pubmed", "Citeseer", "Products", "Papers100M"]:
            X, As, Drs, gnd = self.datasets["single-view"](self.data_name)
            self.data_type = 'single-view ' + self.data_type
        elif self.data_name in ["AMAP", "AMAC"]:
            X, As, Drs, gnd = self.datasets["multi-attribute"](self.data_name)   
            self.data_type = 'multi-attribute '+self.data_type
        elif self.data_name in ["YTF-31", "YTF-400"]:
            self.data_type = "non-"+self.data_type
            X, As, Drs, gnd = self.datasets["non-graph"](self.data_name)
        else:
            print("No such dataset!PLZ recode data_loader.py!")
            return
    
        self.gnd = gnd
        self.C = len(np.unique(self.gnd))
        self.ms = [self.C, 10, 30, 50, 70, 100, 300 ,500]
        self.N = X[0].shape[0]
        return X, As, Drs, gnd

    def graphFiltering(self, k=4):
        X, As, Drs, _ = self.loadData()
        H = multi_view_processing(X=X, A=As, Dr=Drs,k=k, dims=self.max_dims)
        self.V = len(H)
        return H
        
        
    def initailizeB(self, m, H):
        B = []
        for v in range(self.V):
            if self.N > 30000:
                B_tmp = sampling_minikmeans(H, m)[0]
                B.append(B_tmp)
            else:
                B_tmp = sampling_kmeans(H, m)[0] #H[v]
                B.append(B_tmp)
        return B

    def train(self):
        H = self.graphFiltering()
        self.showData()
        for m in self.ms:
            if m < self.C:
                continue
            B = self.initailizeB(m=m, H=H)
            for alpha in self.alphas:
                for beta in self.betas:
                    if self.V == 1:
                        Z, _ = self.algorithms[0](H[0], B[0], alpha=alpha, beta=beta)
                    else:
                        Z, _, _ = self.algorithms[1](H, B, alpha=alpha, beta=beta)
                        
                    acc, nmi, ari, f1, pur = evaluate_clustering(Z, self.gnd)
                    print(
                            "m: {0: <3} alpha: {1: <5} beta: {2: <5} ACC: {3:.4f} NMI: {4:.4f} ARI: {5:.4f} F1: {6:.4f} PUR: {7:.4f}".format(
                                m, alpha, beta, acc, nmi, ari, f1, pur,
                                )
                            )       
                                     
                    if acc > self.best_re["ACC"]:
                        self.best_re["ACC"] = acc
                        self.best_re["NMI"] = nmi
                        self.best_re["ARI"] = ari
                        self.best_re["F1"] = f1
                        self.best_re["PUR"] = pur
                        self.best_paras["alpha"] = alpha
                        self.best_paras["beta"] = beta
                        self.best_paras["m"] = m
                            
    def reproduce(self):
        H = self.graphFiltering()
        B = self.initailizeB(m=self.best_paras["m"], H=H)
        time_begin = time.time()
        if self.V == 1 :
            Z, _ = self.algorithms[0](H[0].copy(), B[0].copy(), alpha=self.best_paras["alpha"], beta=self.best_paras["beta"])
        else :
            Z, _ = self.algorithms[1](H.copy(), B.copy(), alpha=self.best_paras["alpha"], beta=self.best_paras["beta"])

        acc, nmi, ari, f1, pur = evaluate_clustering(Z, self.gnd)
        time_end = time.time()
        Time = np.fabs(time_end - time_begin)
        self.best_re["Time"] = Time
        print(
            "k: {0: <3} m: {1: <3} alpha: {2: <5} beta: {3: <5} ACC: {4:.4f} NMI: {5:.4f} ARI: {6:.4f} F1: {7:.4f} PUR: {8:.4f} Time: {9:.2f}".format(
                    self.k, self.best_paras["m"], self.best_paras["alpha"], self.best_paras["beta"], acc, nmi, ari, f1, pur, Time
                    )
            )
        
    def showData(self):
        print("------------------------------------------------------------------------------")
        print("{} is {} with {} view(s) and {} nodes".format(self.data_name, self.data_type, self.V, self.N))
        print("------------------------------------------------------------------------------")

if __name__ == "__main__":
    dataname = "Citeseer"
    cdc = CDC(dataname=dataname)
    cdc.train()
    cdc.reproduce()