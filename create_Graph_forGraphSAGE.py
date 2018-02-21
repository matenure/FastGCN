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

    train_attr, val_attr, test_attr = ({i: bool(m) for i, m in enumerate(mask)} for mask in (train_mask, val_mask, test_mask))

    nx.set_node_attributes(G, train_attr, 'train')
    nx.set_node_attributes(G, val_attr, 'val')
    nx.set_node_attributes(G, test_attr, 'test')
    
    data = json_graph.node_link_data(G)
    with open("%s/%s0-G.json" % (data_name, data_name), "wb") as f:
        json.dump(data,f)
    classMap = {}
    idMap = {}
    for i in range(len(y)):
        classMap[i]=y[i]
        idMap[i] = i

    with open("%s/%s0-id_map.json" % (data_name, data_name), "wb") as f:
        json.dump(idMap,f)
    with open("%s/%s0-class_map.json" % (data_name, data_name), "wb") as f:
        json.dump(classMap,f)

    np.save("%s/%s0-feats.npy" % (data_name, data_name), features.todense())
