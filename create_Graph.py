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
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(data_name)

    G = nx.from_scipy_sparse_matrix(adj)

    val_index = np.where(val_mask)[0]
    test_index = np.where(test_mask)[0]
    y = y_train+y_val+y_test
    y = np.argmax(y,axis=1)


    for i in range(len(y)):
        if i in val_index:
            G.node[i]['val']=True
            G.node[i]['test']=False
        elif i in test_index:
            G.node[i]['test']=True
            G.node[i]['val']=False
        else:
            G.node[i]['test'] = False
            G.node[i]['val'] = False


    data = json_graph.node_link_data(G)
    with open("cora/cora-G.json","wb") as f:
        json.dump(data,f)
    classMap = {}
    idMap = {}
    for i in range(len(y)):
        classMap[i]=y[i]
        idMap[i] = i
    with open("cora/cora-id_map.json","wb") as f:
        json.dump(idMap,f)
    with open("cora/cora-class_map.json","wb") as f:
        json.dump(classMap,f)
    np.save(open("cora/cora-feats.npy","wb"), features.todense())












