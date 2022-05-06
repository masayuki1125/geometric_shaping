import ray
import pickle
import sys
import numpy as np
#my module
from LDPC_code import LDPC_construction
from LDPC_code import LDPC_encode
from polar_code import polar_construction
from polar_code import polar_encode
from polar_code import polar_decode
from turbo_code import turbo_code
from modulation import modulation
from channel import AWGN

class Mysystem:
    def __init__(self,M,K):
        self.M=M
        self.K=K
        self.N=self.K*int(np.log2(self.M))
        
        self.BICM=False 
        
        if self.BICM==True:
            self.BICM_int=np.arange(self.N)
            np.random.shuffle(self.BICM_int)
            self.BICM_deint=np.argsort(self.BICM_int)
            #check
            a=np.arange(self.N)
            b=a[self.BICM_int]
            c=b[self.BICM_deint]
            if np.any(a!=c):
                print("BICM interleaver error!")
        #coding
        self.cd=polar_construction.coding(self.N,self.K)
        self.ec=polar_encode.encoding(self.cd)
        self.dc=polar_decode.decoding(self.cd,self.ec)
        #modulation
        self.modem=modulation.QAMModem(self.M)

        #channel
        self.ch=AWGN._AWGN()
        
        #filename
        self.filename="polar_code_{}_{}_{}".format(self.N,self.K,self.M)

    def main_func(self,EsNodB):
        #adaptive dicision of frozen bits
        const=polar_construction.Improved_GA()
        if self.cd.decoder_ver==2:
            CRC_len=len(self.cd.CRC_polynomial)-1    
            self.cd.frozen_bits,self.cd.info_bits=const.main_const(self.N,self.K+CRC_len,EsNodB,self.M)
        else:
            self.cd.frozen_bits,self.cd.info_bits=const.main_const(self.N,self.K,EsNodB,self.M)
        
        EsNo = 10 ** (EsNodB / 10)
        No=1/EsNo

        info,cwd=self.ec.polar_encode()
        if self.BICM==True:
            cwd=cwd[self.BICM_int]
        TX_conste=self.modem.modulate(cwd)
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,(No/2)**(1/2))
        if self.BICM==True:
            Lc=Lc[self.BICM_deint]
        EST_info=self.dc.polar_decode(Lc)
        
        return info,EST_info

if __name__=='__main__':
    K=256 #symbol数
    M=4
    EsNodB=-1
    system=Mysystem(M,K)
    print(system.N,system.K)
    info,EST_info=system.main_func(EsNodB)
    print(np.sum(info!=EST_info))