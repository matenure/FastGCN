from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
import scipy.sparse as sp
import os

from utils import *
from models import GCN_APPRO_Mix

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'pubmed', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn_mix', 'Model string.')  # 'gcn_mix', 'gcn_appr'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


def construct_feeddict_forMixlayers(AXfeatures, support, labels, placeholders):
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['AXfeatures']: AXfeatures})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['num_features_nonzero']: AXfeatures[1].shape})
    return feed_dict

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


def main(rank1):

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

    train_index = np.where(train_mask)[0]
    adj_train = adj[train_index, :][:, train_index]
    train_mask = train_mask[train_index]
    y_train = y_train[train_index]
    val_index = np.where(val_mask)[0]
    y_val = y_val[val_index]
    test_index = np.where(test_mask)[0]
    y_test = y_test[test_index]

    train_val_index = np.concatenate([train_index, val_index],axis=0)
    train_test_idnex = np.concatenate([train_index, test_index],axis=0)


    numNode_train = adj_train.shape[0]
    # print("numNode", numNode)


    if FLAGS.model == 'gcn_mix':
        normADJ_train = nontuple_preprocess_adj(adj_train)
        # normADJ = nontuple_preprocess_adj(adj)


        normADJ_val = nontuple_preprocess_adj(adj[train_val_index,:][:,train_val_index])
        normADJ_test = nontuple_preprocess_adj(adj[train_test_idnex,:][:,train_test_idnex])

        num_supports = 2
        model_func = GCN_APPRO_Mix
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Some preprocessing
    features = nontuple_preprocess_features(features).todense()

    train_features = normADJ_train.dot(features[train_index])
    val_features = normADJ_val.dot(features[train_val_index])
    test_features = normADJ_test.dot(features[train_test_idnex])

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
    valSupport = sparse_to_tuple(normADJ_val[len(train_index):, :])
    testSupport = sparse_to_tuple(normADJ_test[len(train_index):, :])

    t = time.time()
    maxACC = 0.0
    # Train model
    for epoch in range(FLAGS.epochs):
        t1 = time.time()

        n = 0
        for batch in iterate_minibatches_listinputs([normADJ_train, y_train], batchsize=1024, shuffle=True):
            [normADJ_batch, y_train_batch] = batch

            p1 = column_prop(normADJ_batch)
            if rank1 is None:
                support1 = sparse_to_tuple(normADJ_batch)
                features_inputs = train_features
            else:
                distr = np.nonzero(np.sum(normADJ_batch, axis=0))[1]
                if rank1 > len(distr):
                    q1 = distr
                else:
                    q1 = np.random.choice(distr, rank1, replace=False)  # top layer
                # q1 = np.random.choice(np.arange(numNode_train), rank1)  # top layer

                support1 = sparse_to_tuple(normADJ_batch[:, q1] * numNode_train / len(q1))

                features_inputs = train_features[q1, :]  # selected nodes for approximation
            # Construct feed dictionary
            feed_dict = construct_feeddict_forMixlayers(features_inputs, support1, y_train_batch,
                                            placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Training step
            outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
            n = n +1


        # Validation
        cost, acc, duration = evaluate(val_features, valSupport, y_val,  placeholders)
        cost_val.append(cost)

        # if epoch > 50 and acc>maxACC:
        #     maxACC = acc
        #     save_path = saver.save(sess, "tmp/tmp_MixModel.ckpt")

        # Print results
        # print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
        #       "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
        #       "val_acc=", "{:.5f}".format(acc), "time per batch=", "{:.5f}".format((time.time() - t1)/n))

        if epoch > FLAGS.early_stopping and np.mean(cost_val[-2:]) > np.mean(cost_val[-(FLAGS.early_stopping + 1):-1]):
            # print("Early stopping...")
            break


    train_duration = time.time() - t
    # Testing
    # if os.path.exists("tmp/pubmed_MixModel.ckpt"):
    #     saver.restore(sess, "tmp/pubmed_MixModel.ckpt")
    test_cost, test_acc, test_duration = evaluate(test_features, testSupport, y_test,
                                                  placeholders)
    print("rank1 = {}".format(rank1), "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "training time per epoch=", "{:.5f}".format(train_duration/(epoch+1)),
          "test time=", "{:.5f}".format(test_duration))

if __name__=="__main__":
    print("DATASET:", FLAGS.dataset)
    main(5)
    # for k in [25, 50, 100, 200, 400]:
    #     main(k)