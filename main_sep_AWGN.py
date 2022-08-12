from grpc import Channel
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
import random

#それぞれのレベルにおける確率を考える
class Mysystem:
    def __init__(self,M,K):
        #make instance
        self.M=M
        self.K=K
        #self.N=self.K*int(np.log2(self.M))
        self.N=self.K*2
        const_var=3 #1:MC 2:iGA 3:RCA
        self.type=1 # fixed
        self.modem_num=0 #0:QAM 1:PSK
        
        self.channel_level=7#どのレベルのチャネルを使うのか特定する
        self.gray_mapping=False
        #for construction
        if const_var==1:
            self.const=monte_carlo_construction.monte_carlo()
            self.const_name="_MC"
        elif const_var==2:
            self.const=iGA.Improved_GA()
            self.const_name="_iGA"
        elif const_var==3:
            self.const=RCA.RCA()
            self.const_name="_RCA"
        
        #ENCの数
        if self.gray_mapping==True:
            self.enc_num=int(np.log2(M**(1/2)))
        elif self.gray_mapping==False:
            self.enc_num=int(np.log2(M))
        
       #coding
       #すべての符号化器で一応符号化する
        self.cd=[]
        self.ec=[]
        self.dc=[]
        for i in range(self.enc_num):
            self.cd+=[polar_construction.coding(self.N,self.K)]
            
            #CRCを調整する
            #if self.enc_num==2:
                #self.cd[i].CRC_polynomial =np.array([1,0,0,1,1])
            #elif self.enc_num==4:
                #self.cd[i].CRC_polynomial =np.array([1,0,1,0,0,1,1,0,1])
            #else:
                #print("unsupported encoder number")
            
            self.ec+=[polar_encode.encoding(self.cd[i])]
            self.dc+=[polar_decode.decoding(self.cd[i],self.ec[i])]
        
        print("R",self.cd[self.channel_level].R)
         
        #get decoder var
        self.decoder_ver=self.cd[0].decoder_ver
        #modulation
        if self.modem_num==0:
            self.modem=modulation.QAMModem(self.M)
        elif self.modem_num==1:
            self.modem=modulation.PSKModem(self.M)
        #channel
        self.ch=AWGN._AWGN()
        
        self.filename=self.make_filename()
    
    def make_filename(self):
        #filename
        filename="polar_{}_{}_{}".format(self.N,self.K,self.M)
        if self.modem_num==0:
            filename+="QAM"
        elif self.modem_num==1:
            filename+="PSK"
        
        filename=filename+self.const_name
        if self.cd[0].systematic_polar==True:
            filename="systematic_"+filename
            
        #decoder type
        if self.cd[0].decoder_ver==0:
            filename+="_SC"
        elif self.cd[0].decoder_ver==2:
            filename+="_CA_SCL"
                    
        #provisional
        filename+="_level{}".format(self.channel_level)
        
        #output filename to confirm which program I run
        print(filename)
        
        return filename
           
    def main_func(self,EsNodB):
        #すべてのデコーダ一律で凍結ビットを得る
        if self.decoder_ver==2:
            self.CRC_len=len(self.cd[0].CRC_polynomial)-1  
            frozen_bits,info_bits=self.const.main_const_unif(self.N,self.K+self.CRC_len,EsNodB,self.M,channel_level=self.channel_level)
        else:
            frozen_bits,info_bits=self.const.main_const_unif(self.N,self.K,EsNodB,self.M,channel_level=self.channel_level)
        
        #change frozen_bits and info_bits         
        self.cd[self.channel_level].design_SNR=EsNodB    
        self.cd[self.channel_level].frozen_bits=frozen_bits
        self.ec[self.channel_level].frozen_bits=frozen_bits
        self.dc[self.channel_level].frozen_bits=frozen_bits
        self.cd[self.channel_level].info_bits=info_bits
        self.ec[self.channel_level].info_bits=info_bits
        self.dc[self.channel_level].info_bits=info_bits
                
        EsNo = 10 ** (EsNodB / 10)
        No=1/EsNo

        #main procedure
        cwd=np.empty(0,dtype=int)
        for i in range(self.enc_num):
            if i==self.channel_level:
                info,cwd_sep=self.ec[self.channel_level].polar_encode()
            else:
                cwd_sep=np.random.randint(0,2,self.N)
                #cwd_sep=np.zeros(self.N,dtype=int)
            #print(len(info_sep))
            cwd=np.concatenate([cwd,cwd_sep])
        
        if self.gray_mapping==True:
            block_intlv_order=int(np.log2(self.M**(1/2)))
        elif self.gray_mapping==False:
            block_intlv_order=int(np.log2(self.M))
        
        cwd=np.reshape(cwd,[block_intlv_order,self.N],order='C')
        cwd=cwd.ravel(order="F")
        
        TX_conste=self.modem.modulate(cwd)
        #print(TX_conste)
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,No)
        
        Lc=np.reshape(Lc,[block_intlv_order,self.N],order='F')[self.channel_level]
        
        EST_info=self.dc[self.channel_level].polar_decode(Lc)
        
        return info,EST_info
    
if __name__=='__main__':
    K=512
    M=16 #symbol number
    
    EsNodB=10.0
    print("EsNodB",EsNodB)
    system=Mysystem(M,K)
    print("\n")
    print(system.N,system.K)
    
    MAXCNT=5
    count_err=0
    count_all=0
    while count_err<MAXCNT:
        count_all+=1
        info,EST_info=system.main_func(EsNodB)
        print("\r"+str(np.sum(info!=EST_info))+str(count_all)+str(count_err),end="")
        if np.any(info!=EST_info):
            count_err+=1
    print("result")
    print(count_err/count_all)