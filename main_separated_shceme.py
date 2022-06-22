import ray
import pickle
import sys
import numpy as np
import math
import os
#my module
from LDPC_code import LDPC_construction
from LDPC_code import LDPC_encode
from LDPC_code import LDPC_decode
from polar_code import polar_construction
from polar_code import polar_encode
from polar_code import polar_decode
from polar_code import RCA
from polar_code import iGA
from polar_code import monte_carlo_construction
from turbo_code import turbo_construction
from turbo_code import turbo_encode
from turbo_code import turbo_decode
from modulation import modulation
from modulation.BICM import make_BICM
from modulation.BICM import BICM_ID
from channel import AWGN
class Mysystem:
    def __init__(self,M,K):
        #make instance
        self.M=M
        self.K=K
        #self.N=self.K*int(np.log2(self.M))
        self.N=self.K*2
        self.BICM=False 
        const_var=2 #1:MC 2:iGA 3:RCA
        
        ##provisional const
        self.type=4 #1:No intlv 2:rand intlv 3:Block intlv 4:separated scheme
        
        #for construction
        if const_var==1:
            self.const=monte_carlo_construction.monte_carlo()
            const_name="_MC"
        elif const_var==2:
            self.const=iGA.Improved_GA()
            const_name="_iGA"
        elif const_var==3:
            self.const=RCA.RCA()
            const_name="_RCA"
        
        #ENCの数
        enc_num=int(np.log2(M**(1/2)))
        if self.N%enc_num!=0:
            print("encoder is not mod(enc_num)")
        self.N_sep=self.N//enc_num
        self.K_sep=self.K//enc_num #とりあえず適当なKで初期化する
       
        #coding
        self.cd=[]
        self.ec=[]
        self.dc=[]
        for i in range(enc_num):
            self.cd+=polar_construction.coding(self.N_sep,self.K_sep)
            self.ec+=polar_encode.encoding(self.cd)
            self.dc+=polar_decode.decoding(self.cd_sep,self.ec_sep)
       
        #modulation
        self.modem=modulation.QAMModem(self.M)
        #channel
        self.ch=AWGN._AWGN()
        
        #filename
        self.filename="polar_code_{}_{}_{}".format(self.N,self.K,self.M)
        self.filename=self.filename+const_name
        if self.cd.systematic_polar==True:
            self.filename="systematic_"+self.filename
        
        #BICM or not
        if self.BICM==True:
            #悪いチャネルと良いチャネルを別々にしてみる
            if self.type==3 or self.type==4:
                seq=np.arange(self.N,dtype=int)
                
                num_of_channels=int(np.log2(self.M**(1/2)))
                res=np.empty(0,dtype=int)
                for i in range(num_of_channels):
                    res=np.concatenate([res,seq[i::num_of_channels]])
                self.BICM_int=res
                self.BICM_deint=np.argsort(self.BICM_int)
            
        #add filename to the number of types 
        self.filename+="_{}".format(self.type)
        
        #output filename to confirm which program I run
        print(self.filename)
        
    def adaptive_BICM(self,EsNodB):
        from capacity_estimation.calc_capacity import make_BMI_list 
        tmp=make_BMI_list(EsNodB,self.M)
        seq_of_channels=np.argsort(tmp[:len(tmp)//2])
        num_of_channels=len(seq_of_channels)
        #print(seq_of_channels)
        seq=np.arange(self.N,dtype=int)
        res=np.empty(0,dtype=int)
        for i in seq_of_channels:
            res=np.concatenate([res,seq[i::num_of_channels]])
        self.BICM_int=res
        #print(self.BICM_int)
        self.BICM_deint=np.argsort(self.BICM_int)
           
    def main_func(self,EsNodB):
        #adaptive change of BICM interleaver
        if self.M!=4:
            if self.type==3:
                self.adaptive_BICM(EsNodB)
        
        #if BICM is True:
        elif self.BICM==True:#BICM purmutation
            if self.cd.decoder_ver==2:
                CRC_len=len(self.cd.CRC_polynomial)-1  
                frozen_bits,info_bits=self.const.main_const(self.N,self.K+CRC_len,EsNodB,self.M,BICM_int=self.BICM_int)
            else:
                frozen_bits,info_bits=self.const.main_const(self.N,self.K,EsNodB,self.M,BICM_int=self.BICM_int)
            
        #check
        for i in range(self.N):
            if (np.any(i==frozen_bits) or np.any(i==info_bits))==False:
                raise ValueError("The frozen set or info set is overlapped")
                
        self.cd.design_SNR==EsNodB    
        self.cd.frozen_bits=frozen_bits
        self.ec.frozen_bits=frozen_bits
        self.dc.frozen_bits=frozen_bits
        self.cd.info_bits=info_bits
        self.ec.info_bits=info_bits
        self.dc.info_bits=info_bits
        #for iGA and RCA and monte_carlo construction
                
        EsNo = 10 ** (EsNodB / 10)
        No=1/EsNo

        info,cwd=self.ec.polar_encode()
        if self.BICM==True:
            cwd=cwd[self.BICM_int]
        TX_conste=self.modem.modulate(cwd)
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,No)
        if self.BICM==True:
            Lc=Lc[self.BICM_deint]
        EST_info=self.dc.polar_decode(Lc)
        
        
        return info,EST_info