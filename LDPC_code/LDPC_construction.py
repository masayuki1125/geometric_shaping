import os
import numpy as np
import scipy
from scipy import sparse
import math 

class coding():
    
  def __init__(self,N,K):
    self.N=N
    self.K=K
    self.R=K/N
    self.max_itr=20
    
    #regular_LDPC
    #prepere constants
    self.Wc=2
    self.Wr=int(self.Wc/self.R)

    if (self.Wr-self.Wc)/self.Wr!=self.R:
        print("encoder rate error")

    self.H=self.generate_regular_H() 
    
    self.tG=self.HtotG()#[:,0:self.K]
    
    self.filename="regular_LDPC_code_{}_{}".format(self.N,self.K)
    #check
    if np.any(self.H.T.shape!=self.tG.shape):
        print("H or tGerror")
        print(self.H.shape)
        print(self.tG.shape)
    
    
    
  #interleave N sequence
  @staticmethod
  def interleave(N):

    interleaver_sequence=np.arange(N)
    np.random.shuffle(interleaver_sequence)
    return interleaver_sequence

  def generate_regular_H(self):
    '''
    #generate regular parity check matrix
    #-----------
    #Wr : row weight
    #Wc : column weight
    #N : length of codeword 
    '''

    if self.N*self.Wc%self.Wr!=0:
      print("constant err")
      exit()

    #generate sub_H matrix(Wc=1)
    sub_H=np.zeros(((self.N-self.K)//self.Wc,self.N),dtype=int)
    for i in range((self.N-self.K)//self.Wc):
        sub_H[i][self.Wr*i:self.Wr*(i+1)]=1

    H=sub_H

    #generate other sub_H matrix(Wc=1)
    for i in range(self.Wc+1):
      sub_H2=sub_H[:,self.interleave(self.N)]
      H=np.concatenate((H,sub_H2))
    
    H=H[:self.K,:]

    return H 

#from https://github.com/hichamjanati/pyldpc/blob/master/pyldpc/code.py 
  @staticmethod
  def binaryproduct(X, Y):
    """Compute a matrix-matrix / vector product in Z/2Z."""
    A = X.dot(Y)
    try:
      A = A.toarray()
    except AttributeError:
      pass
    return A % 2

  @staticmethod
  def gaussjordan(X, change=0):
    """Compute the binary row reduced echelon form of X.
    Parameters
    ----------
    X: array (m, n)
    change : boolean (default, False). If True returns the inverse transform
    Returns
    -------
    if `change` == 'True':
        A: array (m, n). row reduced form of X.
        P: tranformations applied to the identity
    else:
        A: array (m, n). row reduced form of X.
    """
    A = np.copy(X)
    m, n = A.shape

    if change:
      P = np.identity(m).astype(int)

    pivot_old = -1
    for j in range(n):
      filtre_down = A[pivot_old+1:m, j]
      pivot = np.argmax(filtre_down)+pivot_old+1

      if A[pivot, j]:
        pivot_old += 1
        if pivot_old != pivot:
          aux = np.copy(A[pivot, :])
          A[pivot, :] = A[pivot_old, :]
          A[pivot_old, :] = aux
          if change:
            aux = np.copy(P[pivot, :])
            P[pivot, :] = P[pivot_old, :]
            P[pivot_old, :] = aux

        for i in range(m):
          if i != pivot_old and A[i, j]:
            if change:
              P[i, :] = abs(P[i, :]-P[pivot_old, :])
            A[i, :] = abs(A[i, :]-A[pivot_old, :])

      if pivot_old == m-1:
        break

    if change:
      return A, P
    return A

  def HtotG(self,sparse=True):
    """Return the generating coding matrix G given the LDPC matrix H.
    Parameters
    ----------
    H: array (n_equations, n_code). Parity check matrix of an LDPC code with
        code length `n_code` and `n_equations` number of equations.
    sparse: (boolean, default True): if `True`, scipy.sparse format is used
        to speed up computation.
    Returns
    -------
    G.T: array (n_bits, n_code). Transposed coding matrix.
    """

    if type(self.H) == scipy.sparse.csr_matrix:
      self.H = self.H.toarray()
    n_equations, n_code = self.H.shape

    # DOUBLE GAUSS-JORDAN:

    Href_colonnes, tQ = self.gaussjordan(self.H.T, 1)

    Href_diag = self.gaussjordan(np.transpose(Href_colonnes))

    Q = tQ.T

    n_bits = n_code - Href_diag.sum()

    Y = np.zeros(shape=(n_code, n_bits)).astype(int)
    Y[n_code - n_bits:, :] = np.identity(n_bits)

    if sparse:
      Q = scipy.sparse.csr_matrix(Q)
      Y = scipy.sparse.csr_matrix(Y)

    tG = self.binaryproduct(Q, Y)
    
    return tG


if __name__=="__main__":
    N=1024
    #for k in range(2,N-1):
    myLDPC=coding(N,N//2)