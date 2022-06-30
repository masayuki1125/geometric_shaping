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

FEC=1 #1:polar code 2:turbo code 3:LDPC code

class Mysystem_Polar:
    def __init__(self,M,K):
        #make instance
        self.M=M
        self.K=K
        #self.N=self.K*int(np.log2(self.M))
        self.N=self.K*2
        self.BICM=True 
        const_var=2 #1:MC 2:iGA 3:RCA
        
        ##provisional const
        self.type=3#1:No intlv 2:rand intlv 3:Block intlv 4:separated scheme
        if self.type==1:
            self.BICM=False
        elif self.type==2:
            self.BICM=True
        elif self.type==3:
            self.BICM=True
        elif self.type==4:
            self.BICM=True
        
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
        
        #coding
        self.cd=polar_construction.coding(self.N,self.K)
        self.ec=polar_encode.encoding(self.cd)
        self.dc=polar_decode.decoding(self.cd,self.ec)
        #modulation
        self.modem=modulation.QAMModem(self.M)
        #channel
        self.ch=AWGN._AWGN()
        
        #filename
        self.filename="polar_code_SC_{}_{}_{}".format(self.N,self.K,self.M)
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
                
            if self.type==2:
                self.BICM_int,self.BICM_deint=make_BICM(self.N,self.M)
                
            self.filename=self.filename+"_BICM"
            
        #provisional
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
        if self.type==3:
            self.adaptive_BICM(EsNodB)
        #print(self.BICM_int)
        #adaptive dicision of frozen bits
        if self.BICM==False:
            if self.cd.decoder_ver==2:
                CRC_len=len(self.cd.CRC_polynomial)-1  
                frozen_bits,info_bits=self.const.main_const(self.N,self.K+CRC_len,EsNodB,self.M)
            else:
                frozen_bits,info_bits=self.const.main_const(self.N,self.K,EsNodB,self.M)
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
  
class Mysystem_Turbo():
    def __init__(self,M,K):
        self.M=M
        self.K=K
        #self.N=self.K*int(np.log2(self.M))
        self.N=self.K*2
        self.BICM=False 
        
        #coding
        self.cd=turbo_construction.coding(self.N,self.K)
        self.ec=turbo_encode.encoding(self.cd)
        self.dc=turbo_decode.decoding(self.cd)
        #modulation
        self.modem=modulation.QAMModem(self.M)
        

        #channel
        self.ch=AWGN._AWGN()
        
        #filename
        self.filename="turbo_code_{}_{}_{}".format(self.N,self.K,self.M)
        if self.BICM==True:
            self.BICM_int,self.BICM_deint=make_BICM(self.N,self.M)
            self.filename=self.filename+"_BICM"
            
        #output filename to confirm which program I run
        print(self.filename)
    
    def main_func(self,EsNodB):
        EsNo = 10 ** (EsNodB / 10)
        No=1/EsNo

        info,cwd=self.ec.turbo_encode()
        if self.BICM==True:
            cwd=cwd[self.BICM_int]
        TX_conste=self.modem.modulate(cwd)
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,No)
        if self.BICM==True:
            Lc=Lc[self.BICM_deint]
        EST_info=self.dc.turbo_decode(Lc)
        
        return info,EST_info
        
class Mysystem_LDPC():
    def __init__(self,M,K):
        self.M=M
        self.K=K
        #self.N=self.K*int(np.log2(self.M))
        self.N=self.K*2
        self.BICM=True 
        self.BICM_ID=True
        
        if self.BICM_ID==True:
            self.BICM=True
            self.BICM_ID_itr=5
                
        #coding
        self.cd=LDPC_construction.coding(self.N,self.K)
        self.ec=LDPC_encode.encoding(self.cd)
        self.dc=LDPC_decode.decoding(self.cd)
        #modulation
        self.modem=modulation.QAMModem(self.M)
        
        #BICM_ID
        if self.BICM_ID==True:
            self.dmp=BICM_ID(self.modem)
        
        #channel
        self.ch=AWGN._AWGN()
        
        #filename
        self.filename="LDPC_code_{}_{}_{}".format(self.N,self.K,self.M)
        if self.BICM==True:
            self.BICM_int,self.BICM_deint=make_BICM(self.N,self.M)
            self.filename=self.filename+"_BICM"
        
        #output filename to confirm which program I run
        print(self.filename)
    
    def main_func(self,EsNodB):
        EsNo = 10 ** (EsNodB / 10)
        No=1/EsNo

        info,cwd=self.ec.LDPC_encode()
        info=cwd #BICMのとき、cwdがインターリーブされてしまい、比較できなくなる為、infoをcwdに変更する
        if self.BICM==True:
            cwd=cwd[self.BICM_int]
        TX_conste=self.modem.modulate(cwd)
        
        #channel
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        
        #at the reciever
        if self.BICM_ID==False:
            #demodulate
            Lc=self.modem.demodulate(RX_conste,No)
            if self.BICM==True:
                Lc=Lc[self.BICM_deint]
            EST_cwd,_=self.dc.LDPC_decode(Lc)
        
        elif self.BICM_ID==True:
            #demodulate      
            Lc,[zeros,ones]=self.modem.demodulate(RX_conste,No,self.BICM_ID)
            
            _,EX_info=self.dc.LDPC_decode(Lc[self.BICM_deint]) #first decoder
            
            self.dmp.zeros=zeros
            self.dmp.ones=ones
            
            #main loop
            count=0
            while count<self.BICM_ID_itr:
                count+=1
                Pre_info=EX_info[self.BICM_int]#順番の入れ替えをして、事前値にする
                new_Lc=self.dmp.demapper(Pre_info,Lc,No)
                EST_cwd,EX_info=self.dc.LDPC_decode(new_Lc[self.BICM_deint])

                #print(new_Lc)
                #print(Lc)
                #print(EST_cwd) 
            #print(info)
        return info,EST_cwd
    

if FEC==1:
    class Mysystem(Mysystem_Polar):
        def __init__(self,M,K):
            super().__init__(M,K)  
elif FEC==2:
    class Mysystem(Mysystem_Turbo):
        def __init__(self,M,K):
            super().__init__(M,K)      
elif FEC==3:
    class Mysystem(Mysystem_LDPC):
        def __init__(self,M,K):
            super().__init__(M,K)  

if __name__=='__main__':
    K=512 #symbol数
    M=16
    
    EsNodB=8.0
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
    
    '''
    K=512
    M_list=[16,256]
    EsNodB_list=np.arange(4,10,0.5)
    for M in M_list:
        for EsNodB in EsNodB_list:  
            if M==16:
                EsNodB+=0
            elif M==256:
                EsNodB+=10
            mysys=Mysystem(M,K)  
            mysys.main_func(EsNodB)
            const=monte_carlo_construction.monte_carlo()
            const.main_const(mysys.N,mysys.K,EsNodB,mysys.M,BICM_int=mysys.BICM_int)   
    '''
# %%
