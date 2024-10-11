
from data_preprocess import load_dataset
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
from collections import Counter
from graph_construction import knn_graph
from ogb.nodeproppred import PygNodePropPredDataset
import h5py
import os
os.environ["OMP_NUM_THREADS"] = '2'

def Acm(dataname='ACM') :
    if dataname == "ACM" :
        dataset = "./Data/mat/" + 'ACM3025'
        data = sio.loadmat('{}.mat'.format(dataset))


        X = data['feature']
        A = data['PAP']
        B = data['PLP']

    if sp.issparse(X) :
        X = X.todense()
    As = []
    A = np.array(A)
    B = np.array(B)
    X = np.array(X)
    As.append(A)
    As.append(B)
    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)

    return X, As, gnd

def Dblp() :
    ## Load data
    dataset = "./Data/mat/" + 'DBLP4057_GAT_with_idx'
    data = sio.loadmat('{}.mat'.format(dataset))
    X = data['features']
    A = data['net_APTPA']
    B = data['net_APCPA']
    C = data['net_APA']
    if sp.issparse(X) :
        X = X.todense()
    As = []
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    As.append(A)
    As.append(B)
    As.append(C)
    X = np.array(X)
    # av.append(D)
    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)

    return X, As, gnd

def Imdb() :
    # Load data
    dataset = "./Data/mat/" + 'imdb5k'
    data = sio.loadmat('{}.mat'.format(dataset))

    X = data['feature']
    A = data['MAM']
    B = data['MDM']
    if sp.issparse(X) :
        X = X.todense()
    As = []
    X = np.array(X)
    A = np.array(A)
    B = np.array(B)

    As.append(A)
    As.append(B)
    # av.append(C)
    # av.append(D)
    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)
    gnd = np.squeeze(gnd)

    return X, As, gnd


def YoutubeFace():
    data = sio.loadmat('./data/YoutubeFace/YoutubeFace_sel_fea.mat')
    X = data['X'][:, 0]
    gnd = data['Y']
    return X, gnd[:,0]


def YTF400():
    data = h5py.File("data/mat/YTF400_data.mat", "r")
    X = []
    for rfs in data['fea']:
        rf = rfs[0]
        x = data[rf][:]
        x = x.T
        X.append(x)
    gnd = data['gt'][:]
    gnd = gnd[0, :]
    return X, gnd



def multi_relational_graphs(dataname="ACM"):
    if dataname == "ACM" :
        X, Adj, Gnd = Acm()
    elif dataname == "DBLP" :
        X, Adj, Gnd = Dblp()
    elif dataname == "IMDB" :
        X, Adj, Gnd = Imdb()
    else:
        print("No such dataset")
        return None, None, None, None
    As = []
    Drs = []
    for A in Adj:
        Dr = np.sum(A, axis=1)
        Drs.append(Dr)
        A = csr_matrix(A)
        As.append(A)
    X = [X]

    return X, As, Drs, Gnd


def single_view_graphs(dataname='Pubmed'):
    file_root = './data'
    if dataname == "Products":
        dataname = "ogbn-products"
        file_root = './data/ogb'
    if dataname == "Papers100M":
        dataname = "ogbn-papers100M"
        file_root = './data/ogb'

    large_graph_dataset = {
        "Pubmed": Planetoid,
        "Citeseer": Planetoid,
        "Cora": Planetoid,
        "ogbn-products": PygNodePropPredDataset,
        "ogbn-papers100M": PygNodePropPredDataset
    }
    if dataname in ["Texas", "Cornell", "Wisconsin"]:
        dataset = WebKB(root='./data/wiki/{}'.format(dataname), name="{}".format(dataname))
    elif dataname in ["Squirrel", "Chameleon"]:
        dataset = WikipediaNetwork(root='./data/webKB/{}'.format(dataname), name="{}".format(dataname))
    else:
        dataset = large_graph_dataset["{}".format(dataname)](root='{}/{}'.format(file_root, dataname.lower()), name=dataname)

    data = dataset[0]

    X = data.x.numpy()
    N = X.shape[0]
    gnd = data.y.numpy()

    # D = []
    row = data.edge_index[0].numpy()
    col = data.edge_index[1].numpy()

    degree = np.zeros(N)
    Ct = Counter(row)
    degree[list(Ct.keys())] = list(Ct.values())


    M = data.num_edges
    values = torch.ones(M)
    adj = csr_matrix((values, (row, col)), shape=(N, N))

    X = [X]
    adj = [adj]
    degree = [degree]
    if "ogbn" in dataname:
        gnd = gnd[:, 0]
    
    return X, adj, degree, gnd


def multi_attribute_graphs(dataname="AMAP"):

    Amazon = {
        "AMAP":'amazon_photos',
        "AMAC":'amazon_computers'
    }

    dataname = Amazon["{}".format(dataname)]

    X = []
    Amazon = load_dataset("./data/npz/{}.npz".format(dataname))

    Adj = sp.csr_matrix(Amazon.standardize().adj_matrix).A
    Attr = sp.csr_matrix(Amazon.standardize().attr_matrix).A

    Gnd = sp.csr_matrix(Amazon.standardize().labels).A
    Gnd = Gnd.T.squeeze()

    Attr = np.array(Attr)
    X.append(Attr)
    X.append(Attr.dot(Attr.T))
    A = np.array(Adj)
    Dr = np.sum(A, axis=1)
    Dr = [Dr]
    A = csr_matrix(A)
    A = [A]

    return X, A, Dr, Gnd


def non_graph_data(dataname="YTF-31", k=6):
    As = []
    Drs = []

    if dataname == "YTF-31":
        X, gnd = YoutubeFace()
    elif dataname == "YTF-400":
        X, gnd = YTF400()
    else:
        print("No such dataset")
        X, A, gnd = None, None, None
        return X, A, gnd


    for idd, x in enumerate(X):
        try:
            At = sp.load_npz("{}_5nn_{}.npz".format(dataname, idd))
            As.append(At)
        except Exception:
            print("5nn Graphs have not been constructed!")
            At = knn_graph(x, k=k)
            sp.save_npz("{}_5nn_{}.npz".format(dataname, idd), At)
            As.append(At)
        finally:
            Dr = [5 for i in range(x.shape[0])]
            Drs.append(Dr)

    # gnd = gnd[:,0]
    # print(gnd.shape)
    return X, As, Drs, gnd


        
datasets = {
    "single-view": single_view_graphs,
    "multi-relational": multi_relational_graphs,
    "multi-attribute": multi_attribute_graphs,
    "non-graph": non_graph_data,
}

if __name__ == "__main__" :
    # non_graph_data(dataname="YoutubeFace", k=6)
    # ogb_graphs(dataname="ogbn-papers100M")
    from graph_filtering import LowPassFilter_sparse
    from utils import dimension_reduction
    from Metrics_O2MAC import metric_all
    from sklearn.cluster import KMeans
    from utils import dimension_reduction
    X, As, Drs, gnd = multi_relational_graphs("ACM")
    for kk in [1, 2, 4, 6, 8, 10]:
        if X[0].shape[1] > 100:
            x = dimension_reduction(X[0], dt="ACM", dim=100)
        else:
            x = X[0].copy()
        H = LowPassFilter_sparse(x, As[0], Drs[0], k1=kk)
        pred = KMeans(n_clusters=len(np.unique(gnd)), random_state=1234).fit_predict(H)
        Re = metric_all.clustering_metrics(gnd, pred)
        ac, nm, ar, f, pur = Re.evaluationClusterModelFromLabel()
        print("k: {} ac: {}".format(kk, ac))