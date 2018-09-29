#### Please first download original Reddit Graph Data: http://snap.stanford.edu/graphsage/reddit.zip
####


import json
from networkx.readwrite import json_graph
import scipy.sparse as sp
import numpy as np
import pickle as pkl


def loadRedditFromG(dataset_dir, inputfile):
    f= open(dataset_dir+inputfile)
    objects = []
    for _ in range(pkl.load(f)):
        objects.append(pkl.load(f))
    adj, train_labels, val_labels, test_labels, train_index, val_index, test_index = tuple(objects)
    feats = np.load(dataset_dir + "/reddit-feats.npy")
    return sp.csr_matrix(adj), sp.lil_matrix(feats), train_labels, val_labels, test_labels, train_index, val_index, test_index


def loadRedditFromNPZ(dataset_dir):
    adj = sp.load_npz(dataset_dir+"reddit_adj.npz")
    data = np.load(dataset_dir+"reddit.npz")

    return adj, data['feats'], data['y_train'], data['y_val'], data['y_test'], data['train_index'], data['val_index'], data['test_index']


def transferRedditData2AdjNPZ(dataset_dir):
    G = json_graph.node_link_graph(json.load(open(dataset_dir + "/reddit-G.json")))
    feat_id_map = json.load(open(dataset_dir + "/reddit-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.iteritems()}
    numNode = len(feat_id_map)
    print(numNode)
    adj = sp.lil_matrix((numNode, numNode))
    print("no")
    for edge in G.edges():
        adj[feat_id_map[edge[0]], feat_id_map[edge[1]]] = 1
    sp.save_npz("reddit_adj.npz", sp.coo_matrix(adj))


def transferRedditDataFormat(dataset_dir, output_file):
    G = json_graph.node_link_graph(json.load(open(dataset_dir + "/reddit-G.json")))
    labels = json.load(open(dataset_dir + "/reddit-class_map.json"))

    train_ids = [n for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]
    test_ids = [n for n in G.nodes() if G.node[n]['test']]
    val_ids = [n for n in G.nodes() if G.node[n]['val']]
    train_labels = [labels[i] for i in train_ids]
    test_labels = [labels[i] for i in test_ids]
    val_labels = [labels[i] for i in val_ids]
    feats = np.load(dataset_dir + "/reddit-feats.npy")
    ## Logistic gets thrown off by big counts, so log transform num comments and score
    feats[:, 0] = np.log(feats[:, 0] + 1.0)
    feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
    feat_id_map = json.load(open(dataset_dir + "reddit-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.iteritems()}

    train_index = [feat_id_map[id] for id in train_ids]
    val_index = [feat_id_map[id] for id in val_ids]
    test_index = [feat_id_map[id] for id in test_ids]
    np.savez(output_file, feats=feats, y_train=train_labels, y_val=val_labels, y_test=test_labels,
             train_index=train_index,
             val_index=val_index, test_index=test_index)


if __name__=="__main__":
    # transferRedditData2AdjNPZ("reddit")
    transferRedditDataFormat("reddit","reddit.npz")