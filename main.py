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
from channel import AWGN

FEC=1 #1:polar code 2:turbo code 3:LDPC code

class Mysystem_Polar:
    def __init__(self,M,K):
        #make instance
        self.M=M
        self.K=K
        #self.N=self.K*int(np.log2(self.M))
        self.N=self.K*2
        self.BICM=False 
        const_var=2
        
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
        self.filename="polar_code_{}_{}_{}".format(self.N,self.K,self.M)
        self.filename=self.filename+const_name
        
        #BICM or not
        if self.BICM==True:
            self.BICM_int,self.BICM_deint=make_BICM(self.N,self.M)
            self.filename=self.filename+"_BICM"
        
        #output filename to confirm which program I run
        print(self.filename)
            
    def main_func(self,EsNodB):
        #adaptive dicision of frozen bits
        if self.cd.decoder_ver==2:
            CRC_len=len(self.cd.CRC_polynomial)-1  
            #if self.BICM==True:  
                #self.cd.frozen_bits,self.cd.info_bits=self.const.main_const(self.N,self.K+CRC_len,EsNodB,self.M,BICM_int=self.BICM_int)
            #else:
            frozen_bits,info_bits=self.const.main_const(self.N,self.K+CRC_len,EsNodB,self.M)
        else:
            #if self.BICM==True:  
                #self.cd.frozen_bits,self.cd.info_bits=self.const.main_const(self.N,self.K,EsNodB,self.M,BICM_int=self.BICM_int)
            #else:
            frozen_bits,info_bits=self.const.main_const(self.N,self.K,EsNodB,self.M)
                
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
        
        #print(EST_info)
        #print(info)
        
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
        self.BICM=False 
                
        #coding
        self.cd=LDPC_construction.coding(self.N,self.K)
        self.ec=LDPC_encode.encoding(self.cd)
        self.dc=LDPC_decode.decoding(self.cd)
        #modulation
        self.modem=modulation.QAMModem(self.M)
        

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
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,No)
        if self.BICM==True:
            Lc=Lc[self.BICM_deint]
        EST_cwd,EX_info=self.dc.LDPC_decode(Lc)
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
    M=4
    
    EsNodB=3.0
    print("EsNodB",EsNodB)
    system=Mysystem(M,K)
    print("\n")
    print(system.N,system.K)
    info,EST_info=system.main_func(EsNodB)
    print(np.sum(info!=EST_info))
    
    
    '''
    K=4096
    M_list=[4,16,256]
    EsNodB_list=np.arange(0,10,0.5)
    for M in M_list:
        for EsNodB in EsNodB_list:  
            if M==16:
                EsNodB+=5
            elif M==256:
                EsNodB+=10
            mysys=Mysystem(M,K)  
            const=monte_carlo_construction.monte_carlo()
            const.main_const(mysys.N,mysys.K,EsNodB,mysys.M)    
    '''