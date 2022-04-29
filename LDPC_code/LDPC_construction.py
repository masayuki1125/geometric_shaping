import os
import numpy as np
import scipy
from scipy import sparse
import math 

class coding():
    
  def __init__(self,N,K,encoder_var=0):

    self.encoder_var=encoder_var #0:regular_LDPC 1:NR_LDPC(quasi cyclic)

    #self.ch=_AWGN() #difine channel
    
    self.N=N
    self.K=K
    self.R=K/N
    self.max_iter=50
    
    if self.encoder_var==0:#regular_LDPC
      #prepere constants
      self.Wc=2
      self.Wr=4

      if (self.Wr-self.Wc)/self.Wr!=self.R:
        print("encoder rate error")

      self.H=self.generate_regular_H()  
      self.tG=self.HtotG()   
      self.filename="regular_LDPC_code_{}_{}".format(self.N,self.K)
    
    elif self.encoder_var==1:#NR_LDPC
      self.Zc,filename,self.BG_num,Kb=self.generate_filename(self.K,self.R)
      print(filename)
      self.R=self.K/(self.N-2*self.Zc)
      self.H=self.generate_NR_H(Kb,self.Zc,filename)

      #redifine N and K

      self.K=Kb*self.Zc
      self.N=self.K+self.H.shape[0]
      print("N,K")
      print((self.N,self.K))
      #print("matrix shape")
      #print(self.H.shape)
      print("R")
      print(self.R)
      #modify H 
      #self.H=self.H[:self.K,:(self.H.shape[1]-self.H.shape[0]+self.K)]
  
      self.filename="NR_LDPC_code_{}_{}_{}".format(self.N,self.K,math.floor(self.R*10)/10)

    
    self.H=sparse.csr_matrix(self.H)

    #np.savetxt("tG",self.tG,fmt='%i')
    #np.savetxt("H",self.H.toarray(),fmt='%i')

  @staticmethod
  def generate_filename(K,R):

    #decide BG_num
    if K<=3824 and R<=0.67:
      BG_num=2
    elif K<=292:
      BG_num=2
    elif R<=0.25:
      BG_num=2
    else:
      BG_num=1

    #decide Kb
    if BG_num==1:
      Kb=22
    else:
      if K>640:
        Kb=10
      elif 560<K<=640:
        Kb=9
      elif 192<K<=560:
        Kb=8
      elif K<=192:
        Kb=6

    #decide Zc

    a=np.arange(2,16)
    j=np.arange(0,8)

    a,j=np.meshgrid(a,j)
    a=a.flatten()
    j=j.flatten()

    Zc_array=a*(2**j)
    MAX_Zc=384
    Zc_array=Zc_array[MAX_Zc>=Zc_array]
    #print(Zc_array)
    Zc=np.min(Zc_array[Zc_array>=K/Kb])

    #decide iLS
    i=list()
    i0=np.array([2,4,8,16,32,64,128,256])
    i1=np.array([3,6,12,24,48,96,192,384])
    i2=np.array([5,10,20,40,80,160,320])
    i3=np.array([7,14,28,56,112,224])
    i4=np.array([9,18,36,72,144,288])
    i5=np.array([11,22,44,88,176,352])
    i6=np.array([13,26,52,104,208])
    i7=np.array([15,30,60,120,240])
    i_list=[i0,i1,i2,i3,i4,i5,i6,i7]

    for count,i in enumerate(i_list):
      if np.any(i==Zc):
        iLS=count

    filename='NR_'+str(BG_num)+'_'+str(iLS)+'_'+str(Zc)+'.txt'

    return Zc,filename,BG_num,Kb

  @staticmethod
  def permute(a,Zc): #n*n単位行列をaだけシフトさせる
    if a==-1:
      tmp=np.zeros([Zc,Zc],dtype=int)
    else:
      tmp=np.identity(Zc,dtype=int)
      tmp=np.roll(tmp,a,axis=1)
    return tmp

  def generate_NR_H(self,Kb,Zc,filename):

    base_matrix=np.loadtxt(os.path.join('base_matrices', filename),dtype='int')
    
    if self.BG_num==1:
        tmp=22
    elif self.BG_num==2:
        tmp=10
    
    Mb=np.arange((self.N-self.K)//Zc)
    Nb=np.arange(tmp+len(Mb))
    #print(Zc)
    #print(Nb)
    #print(Mb)

    H=np.empty((0,Zc*len(Nb)),dtype=int)
    for i in Mb:
        matrix_row=np.empty((Zc,0),dtype=int)

        for j in Nb:
            tmp=self.permute(base_matrix[i,j],Zc)
            matrix_row=np.concatenate([matrix_row,tmp],axis=1)

        H=np.concatenate([H,matrix_row],axis=0)
        
    if H[H.shape[0]-1,H.shape[1]-1]!=1:
      print("H error")
        
    return H

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
  for k in range(2,N-1):
    myLDPC=coding(N,k)