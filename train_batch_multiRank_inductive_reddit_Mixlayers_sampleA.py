from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy.sparse as sp

from utils import *
from models import GCN_APPRO_Mix
import json
from networkx.readwrite import json_graph
import os

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_mix', 'Model string.')  # 'gcn', 'gcn_appr'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data


def iterate_minibatches_listinputs(inputs, batchsize, shuffle=False):
    assert inputs is not None
    numSamples = inputs[0].shape[0]
    if shuffle:
        indices = np.arange(numSamples)
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield [input[excerpt] for input in inputs]


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

    # train_feats = feats[[feat_id_map[id] for id in train_ids]]
    # test_feats = feats[[feat_id_map[id] for id in test_ids]]

    # numNode = len(feat_id_map)
    # adj = sp.lil_matrix(np.zeros((numNode,numNode)))
    # for edge in G.edges():
    #     adj[feat_id_map[edge[0]], feat_id_map[edge[1]]] = 1

    train_index = [feat_id_map[id] for id in train_ids]
    val_index = [feat_id_map[id] for id in val_ids]
    test_index = [feat_id_map[id] for id in test_ids]
    np.savez(output_file, feats = feats, y_train=train_labels, y_val=val_labels, y_test = test_labels, train_index = train_index,
             val_index=val_index, test_index = test_index)


def transferLabel2Onehot(labels, N):
    y = np.zeros((len(labels),N))
    for i in range(len(labels)):
        pos = labels[i]
        y[i,pos] =1
    return y

def construct_feeddict_forMixlayers(AXfeatures, support, labels, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['AXfeatures']: AXfeatures})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: AXfeatures[1].shape})
    return feed_dict

def main(rank1):



    # config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
    #                 inter_op_parallelism_threads = 1,
    #                 intra_op_parallelism_threads = 4,
    #                 log_device_placement=False)
    adj, features, y_train, y_val, y_test,train_index, val_index, test_index = loadRedditFromNPZ("data/")
    adj = adj+adj.T


    y_train = transferLabel2Onehot(y_train, 41)
    y_val = transferLabel2Onehot(y_val, 41)
    y_test = transferLabel2Onehot(y_test, 41)

    features = sp.lil_matrix(features)

    adj_train = adj[train_index, :][:, train_index]


    numNode_train = adj_train.shape[0]


    # print("numNode", numNode)



    if FLAGS.model == 'gcn_mix':
        normADJ_train = nontuple_preprocess_adj(adj_train)
        normADJ = nontuple_preprocess_adj(adj)
        # normADJ_val = nontuple_preprocess_adj(adj_val)
        # normADJ_test = nontuple_preprocess_adj(adj_test)

        num_supports = 2
        model_func = GCN_APPRO_Mix
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Some preprocessing
    features = nontuple_preprocess_features(features).todense()

    train_features = normADJ_train.dot(features[train_index])
    features = normADJ.dot(features)
    nonzero_feature_number = len(np.nonzero(features)[0])
    nonzero_feature_number_train = len(np.nonzero(train_features)[0])


    # Define placeholders
    placeholders = {
        'support': tf.sparse_placeholder(tf.float32) ,
        'AXfeatures': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features.shape[-1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feeddict_forMixlayers(features, support, labels, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    cost_val = []

    p0 = column_prop(normADJ_train)

    # testSupport = [sparse_to_tuple(normADJ), sparse_to_tuple(normADJ)]
    valSupport = sparse_to_tuple(normADJ[val_index, :])
    testSupport = sparse_to_tuple(normADJ[test_index, :])

    t = time.time()
    maxACC = 0.0
    # Train model
    for epoch in range(FLAGS.epochs):
        t1 = time.time()

        n = 0
        for batch in iterate_minibatches_listinputs([normADJ_train, y_train], batchsize=256, shuffle=True):
            [normADJ_batch, y_train_batch] = batch

            # p1 = column_prop(normADJ_batch)
            if rank1 is None:
                support1 = sparse_to_tuple(normADJ_batch)
                features_inputs = train_features
            else:
                distr = np.nonzero(np.sum(normADJ_batch, axis=0))[1]
                if rank1 > len(distr):
                    q1 = distr
                else:
                    q1 = np.random.choice(distr, rank1, replace=False, p=p0[distr]/sum(p0[distr]))  # top layer

                # q1 = np.random.choice(np.arange(numNode_train), rank1, p=p0)  # top layer

                support1 = sparse_to_tuple(normADJ_batch[:, q1].dot(sp.diags(1.0 / (p0[q1] * rank1))))
                if len(support1[1])==0:
                    continue

                features_inputs = train_features[q1, :]  # selected nodes for approximation
            # Construct feed dictionary
            feed_dict = construct_feeddict_forMixlayers(features_inputs, support1, y_train_batch,
                                            placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            n = n+1


        # Validation
        cost, acc, duration = evaluate(features, valSupport, y_val,  placeholders)
        cost_val.append(cost)

        if epoch > 20 and acc>maxACC:
            maxACC = acc
            saver.save(sess, "tmp/tmp_MixModel_sampleA_full.ckpt")

        # Print results
        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc), "time per batch=", "{:.5f}".format((time.time() - t1)/n))

        if epoch%5==0:
            # Validation
            test_cost, test_acc, test_duration = evaluate(features, testSupport, y_test,
                                                          placeholders)
            print("training time by far=", "{:.5f}".format(time.time() - t),
                  "epoch = {}".format(epoch + 1),
                  "cost=", "{:.5f}".format(test_cost),
                  "accuracy=", "{:.5f}".format(test_acc))

        if epoch > FLAGS.early_stopping and np.mean(cost_val[-2:]) > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            # print("Early stopping...")
            break

    train_duration = time.time() - t
    # Testing
    if os.path.exists("tmp/tmp_MixModel_sampleA_full.ckpt.index"):
        saver.restore(sess, "tmp/tmp_MixModel_sampleA_full.ckpt")
    test_cost, test_acc, test_duration = evaluate(features, testSupport, y_test,
                                                  placeholders)
    print("rank1 = {}".format(rank1), "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "training time=", "{:.5f}".format(train_duration),
          "epoch = {}".format(epoch+1),
          "test time=", "{:.5f}".format(test_duration))

def transferG2ADJ():
    G = json_graph.node_link_graph(json.load(open("reddit/reddit-G.json")))
    feat_id_map = json.load(open("reddit/reddit-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.iteritems()}
    numNode = len(feat_id_map)
    adj = np.zeros((numNode, numNode))
    newEdges0 = [feat_id_map[edge[0]] for edge in G.edges()]
    newEdges1 = [feat_id_map[edge[1]] for edge in G.edges()]

    # for edge in G.edges():
    #     adj[feat_id_map[edge[0]], feat_id_map[edge[1]]] = 1
    adj = sp.csr_matrix((np.ones((len(newEdges0),)), (newEdges0, newEdges1)), shape=(numNode, numNode))
    sp.save_npz("reddit_adj.npz", adj)


def test(rank1=None):
    # config = tf.ConfigProto(device_count={"CPU": 4}, # limit to num_cpu_core CPU usage
    #                 inter_op_parallelism_threads = 1,
    #                 intra_op_parallelism_threads = 4,
    #                 log_device_placement=False)
    adj, features, y_train, y_val, y_test, train_index, val_index, test_index = loadRedditFromNPZ("data/")
    adj = adj + adj.T

    y_train = transferLabel2Onehot(y_train, 41)
    y_test = transferLabel2Onehot(y_test, 41)

    features = sp.lil_matrix(features)


    numNode_train = y_train.shape[0]

    # print("numNode", numNode)



    if FLAGS.model == 'gcn_mix':
        normADJ = nontuple_preprocess_adj(adj)
        normADJ_test = normADJ[test_index, :]
        # normADJ_val = nontuple_preprocess_adj(adj_val)
        # normADJ_test = nontuple_preprocess_adj(adj_test)

        num_supports = 2
        model_func = GCN_APPRO_Mix
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Some preprocessing
    features = nontuple_preprocess_features(features).todense()

    features = normADJ.dot(features)


    # Define placeholders
    placeholders = {
        'support': tf.sparse_placeholder(tf.float32),
        'AXfeatures': tf.placeholder(tf.float32, shape=(None, features.shape[1])),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features.shape[-1], logging=True)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(features, support, labels, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feeddict_forMixlayers(features, support, labels, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    saver.restore(sess, "tmp/tmp_MixModel_sampleA.ckpt")

    cost_val = []

    p0 = column_prop(normADJ_test)


    t = time.time()

    if rank1 is None:
        support1 = sparse_to_tuple(normADJ_test)
        features_inputs = features
    else:
        distr = np.nonzero(np.sum(normADJ_test, axis=0))[1]
        if rank1 > len(distr):
            q1 = distr
        else:
            q1 = np.random.choice(distr, rank1, replace=False, p=p0[distr] / sum(p0[distr]))  # top layer

        # q1 = np.random.choice(np.arange(numNode_train), rank1, p=p0)  # top layer

        support1 = sparse_to_tuple(normADJ_test[:, q1].dot(sp.diags(1.0 / (p0[q1] * rank1))))


        features_inputs = features[q1, :]  # selected nodes for approximation

    test_cost, test_acc, test_duration = evaluate(features_inputs, support1, y_test,
                                                  placeholders)


    test_duration = time.time() - t
    print("rank1 = {}".format(rank1), "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc),
          "test time=", "{:.5f}".format(test_duration))

if __name__=="__main__":
    # main(None)
    main(None)
    # for k in [25, 50, 100, 200, 400]:
    #     main(k)