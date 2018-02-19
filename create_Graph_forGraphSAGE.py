import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
import os
import networkx as nx
from utils import *
import json
from networkx.readwrite import json_graph

 # 'cora', 'citeseer', 'pubmed'

if __name__=="__main__":
    data_name = 'cora'
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_original(data_name)

    G = nx.from_scipy_sparse_matrix(adj)
    train_index = np.where(train_mask)[0]
    val_index = np.where(val_mask)[0]
    test_index = np.where(test_mask)[0]
    y = y_train+y_val+y_test
    y = np.argmax(y,axis=1)


    for i in range(len(y)):
        if i in val_index:
            G.node[i]['val']=True
            G.node[i]['test']=False
            G.node[i]['train']=False
        elif i in test_index:
            G.node[i]['test']=True
            G.node[i]['val']=False
            G.node[i]['train'] = False
        elif i in train_index:
            G.node[i]['test']=False
            G.node[i]['val']=False
            G.node[i]['train'] = True
        else:
            G.node[i]['test'] = False
            G.node[i]['val'] = False
            G.node[i]['train'] = False


    data = json_graph.node_link_data(G)
    with open("cora/cora0-G.json","wb") as f:
        json.dump(data,f)
    classMap = {}
    idMap = {}
    for i in range(len(y)):
        classMap[i]=y[i]
        idMap[i] = i
    with open("cora/cora0-id_map.json","wb") as f:
        json.dump(idMap,f)
    with open("cora/cora0-class_map.json","wb") as f:
        json.dump(classMap,f)
    np.save(open("cora/cora0-feats.npy","wb"), features.todense())












