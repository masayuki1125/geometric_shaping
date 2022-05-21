import numpy as np
import scipy
from scipy import sparse
import numpy as np
import math

from sympy import LC 

class decoding():
    def __init__(self,cd):
        self.N=cd.N
        self.K=cd.K
        self.H=cd.H
        self.H=sparse.csr_matrix(self.H) 
        #print(self.H)
        self.max_itr=cd.max_itr

        self.ii,self.jj=self.H.nonzero()
        self.m,self.n=self.H.shape

    def phi(self,mat):
        '''
        input: 2D matrix 
        output: 2D matrix (same as H)
        '''
        smat=mat.toarray()[self.ii, self.jj]

        #clipping operaiton
        smat[smat>10**2]=10**2
        smat[smat < 10**-5] = 10**-5

        smat=np.log((np.exp(smat) + 1) / (np.exp(smat) - 1))

        mat=sparse.csr_matrix((smat, (self.ii, self.jj)), shape=(self.m, self.n))

        return mat
  
    def make_alpha(self,mat):
        '''
        input: 2D matrix(same as H)
        output: 2D matrix (same as H)
        '''
        smat= mat.toarray()[self.ii, self.jj]

        salpha=np.sign(smat)
        alpha=sparse.csr_matrix((salpha, (self.ii, self.jj)), shape=(self.m, self.n))
        
        mask=(alpha-self.H).getnnz(axis=1)%2 #-1の数が奇数なら１、偶数なら０を出力
        mask=-2*mask+1
        #列ごとに掛け算する マイナスの列は１、プラスの列は０
        alpha=sparse.spdiags(mask, 0, self.m, self.m, 'csr').dot(alpha)
        return alpha

    def make_beta(self,mat):
        '''
        input: 2D array
        output: 2D matrix (same as H)
        '''
        smat= mat.toarray()[self.ii, self.jj]

        sbeta=np.abs(smat)
        beta=sparse.csr_matrix((sbeta, (self.ii, self.jj)), shape=(self.m, self.n))

        #leave-one-out operation
        beta=self.phi(beta)
        mask=beta.sum(axis=1).ravel()
        tmp=sparse.spdiags(mask, 0, self.m, self.m, 'csr').dot(self.H)
        beta=tmp-beta
        beta=self.phi(beta)

        return beta

    def sum_product(self,Lc):
        
        # initialization
        L_mat = self.H.dot(sparse.spdiags(Lc, 0, self.n, self.n, 'csr'))
        
        k=0 #itr counter

        while k < self.max_itr:
            ##horizontal operation from L_mat to L_mat 
            #calcurate alpha
            alpha=self.make_alpha(L_mat)
            #culcurate beta
            beta=self.make_beta(L_mat)

            L_mat=alpha.multiply(beta)

            ##vertical operation
            stmp=L_mat.sum(axis=0).ravel()
            stmp+=Lc
            tmp=self.H.dot(sparse.spdiags(stmp, 0, self.n, self.n, 'csr'))
            L_mat=tmp-L_mat

            ##check operation
            EX_info=L_mat.sum(axis=0)
            EST_Lc=Lc+EX_info
            EST_codeword=(np.sign(EST_Lc)+1)/2

            #convert from matrix class to array class
            EST_codeword=(np.asarray(EST_codeword)).flatten()
            if np.all(self.H.dot(EST_codeword)%2 == 0):
                break
            k+=1
        
        return EST_codeword ,EX_info

    def LDPC_decode(self,Lc):
        #Lcをプラスとマイナス逆にする
        Lc=-1*Lc
        
        EST_codeword ,*EX_info=self.sum_product(Lc)
        #EST_information=EST_codeword[(self.N-self.K):] #systematicじゃないので、情報ビットだけで測れない
        return EST_codeword ,np.asarray(EX_info).flatten()

    
if __name__=="__main__":
    from LDPC_construction import coding
    from LDPC_encode import encoding
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
    from channel.AWGN import _AWGN
    from modulation.modulation import QAMModem
    N=1024
    K=512
    EsNodB=3
    EsNo = 10 ** (EsNodB / 10)
    No=1/EsNo
    cd=coding(N,K)
    ec=encoding(cd)
    dc=decoding(cd)
    modem=QAMModem(4)
    ch=_AWGN()
    
    info,cwd=ec.LDPC_encode()
    TX_const=modem.modulate(cwd)
    RX_const=ch.add_AWGN(TX_const,No)
    Lc=modem.demodulate(RX_const,No)

    print(Lc)
    EST_cwd,*EX_info=dc.LDPC_decode(Lc)
    print(EST_cwd)
    print(EX_info)
    print(np.sum(cwd!=EST_cwd))
    