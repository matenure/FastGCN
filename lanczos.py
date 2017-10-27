import numpy as np
from numpy.linalg import norm
from utils import load_data as dataload
import scipy.sparse as sparse
import pickle
from scipy.linalg import qr, svd

def lanczos(A,k,q):
    n = A.shape[0]
    Q = np.zeros((n,k+1))

    Q[:,0] = q/norm(q)

    alpha = 0
    beta = 0

    for i in range(k):
      if i == 0:
        q = np.dot(A,Q[:,i])
      else:
        q = np.dot(A, Q[:,i]) - beta*Q[:,i-1]
      alpha = np.dot(q.T, Q[:,i])
      q = q - Q[:,i]*alpha
      q = q - np.dot(Q[:,:i], np.dot(Q[:,:i].T, q)) # full reorthogonalization
      beta = norm(q)
      Q[:,i+1] = q/beta
      print(i)

    Q = Q[:,:k]

    Sigma = np.dot(Q.T, np.dot(A, Q))
    # A2 = np.dot(Q[:,:k], np.dot(Sigma[:k,:k], Q[:,:k].T))
    # return A2
    return Q, Sigma

def dense_RandomSVD(A,K):
    G = np.random.randn(A.shape[0],K)
    B = np.dot(A,G)
    Q,R =qr(B,mode='economic')
    M = np.dot(np.dot(Q, np.dot(np.dot(Q.T, A),Q)),Q.T)
    return M


if __name__=="__main__":
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = dataload('cora')
    print(adj.shape)
    adj = np.array(sparse.csr_matrix.todense(adj))
    # np.save("ADJ_cora.npy",adj)
    q = np.random.randn(adj.shape[0],)
    Q, sigma = lanczos(adj,100,q)
    r = 100
    A2 = np.dot(Q[:,:r], np.dot(sigma[:r,:r], Q[:,:r].T))

    # u,v,a = svd(adj)

    err = norm(adj-A2)/norm(adj)
    print(err)


# A = np.random.random((10000,10000))
# A = np.triu(A) + np.triu(A).T
# q = np.random.random((10000,))
# K = 100
# Q, sigma = lanczos(A,K,q)
# r = 100
# A2 = np.dot(Q[:,:r], np.dot(sigma[:r,:r], Q[:,:r].T))
# err = norm(A-A2)/norm(A)
# print(err)
