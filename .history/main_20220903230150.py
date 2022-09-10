#import ray
#import pickle
#import sys
import numpy as np
#import math
#import os
#import random
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

from capacity_estimation.calc_capacity import make_BMI_list

FEC=1#1:polar code 2:turbo code 3:LDPC code

class Mysystem_Polar:
    def __init__(self,M,K):
        #make instance
        self.M=M
        self.K=K
        #self.N=self.K*int(np.log2(self.M))
        self.N=self.K*2
        const_var=3#1:MC 2:iGA 3:RCA 4:GA
        self.type=3#1:separated scheme 2:Block intlv(No intlv in arikan polar decoder) 3:No intlv(Block intlv in arikan polar decoder) 4:rand intlv
        self.adaptive_intlv=False #default:false
        
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
        
        #coding
        self.cd=polar_construction.coding(self.N,self.K)
        self.ec=polar_encode.encoding(self.cd)
        self.dc=polar_decode.decoding(self.cd,self.ec)
        #modulation
        self.modem=modulation.QAMModem(self.M)
        #channel
        self.ch=AWGN._AWGN()
        
        #interleaver design
        self.BICM_int,self.BICM_deint=self.make_BICM_int(self.N,self.M,self.type)
        
        self.filename=self.make_filename()
        
    def make_filename(self):
        filename="{}QAM".format(self.M)
        #filename="polar_{}_{}_{}QAM".format(self.N,self.K,self.M)
        filename=filename+self.const_name
        if self.cd.systematic_polar==True:
            filename="systematic_"+filename
            
        #decoder type
        if self.cd.decoder_ver==0:
            filename+="_SC"
        elif self.cd.decoder_ver==2:
            filename+="_CA_SCL"
                    
        #provisional
        filename+="_type{}".format(self.type)
        
        #output filename to confirm which program I run
        print(filename)
        return filename
        
    def make_BICM_int(self,N,M,type):
        
        BICM_int=np.arange(N,dtype=int)
        #modify BICM int from simplified to arikan decoder order
        
        if type==1:#1:separated scheme 
            print("err type1")
            pass #specific file is needed
        elif type==2:#2:No intlv in arikan polar decoder
            pass
            #BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            #BICM_int[0]=np.sort(BICM_int[0])
            #BICM_int[1]=np.sort(BICM_int[1])
            #BICM_int=np.ravel(BICM_int,order='C')
            #print(BICM_int)
            
        elif type==3:#3:Block intlv in arikan polar decoder
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            BICM_int=np.ravel(BICM_int,order='F')
            #print(BICM_int)
        elif type==4:#4:rand intlv
            #bit reversal order
            bit_reversal_sequence=self.cd.bit_reversal_sequence
            BICM_int=BICM_int[bit_reversal_sequence]
            
            tmp,_=make_BICM(N)
            BICM_int=BICM_int[tmp]
        elif type==5:#2:No intlv +rand intlv for each channel
            #bit reversal order
            bit_reversal_sequence=self.cd.bit_reversal_sequence
            BICM_int=BICM_int[bit_reversal_sequence]
            
            tmp,_=make_BICM(N//int(np.log2(M**(1/2))))
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            for i in range (int(np.log2(M**(1/2)))):
                BICM_int[i]=BICM_int[i][tmp]
            BICM_int=np.ravel(BICM_int,order='C')
            #print(BICM_int)
            
        elif type==6:#凍結ビットを低SNRに設定する
            self.adaptive_intlv=True
            pass#specific file is needed
        elif type==7:#compound polar codes
            #use block interleaver
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            BICM_int=np.ravel(BICM_int,order='F')
            print("err type7")
            pass #specific file is needed
            
        else:
            print("interleaver type error")
        BICM_deint=np.argsort(BICM_int)
        #np.savetxt("deint",BICM_deint,fmt='%.0f')
        #print(BICM_int)
        #print(BICM_deint) 
        return BICM_int,BICM_deint
    
    #以下の関数はType6用の関数
    def construction(self,BICM_int,EsNodB):
        frozen_bits,info_bits=self.const.main_const(self.N,self.K,EsNodB,self.M,BICM_int=BICM_int)
        
        #print(frozen_bits)
        BICM_int=np.concatenate([frozen_bits,info_bits])
        tmp=make_BMI_list(EsNodB,self.M)
        argtmp=np.argsort(tmp[:len(tmp)//2])
        #print(tmp)
        #print(argtmp)
        BICM_int=np.reshape(BICM_int,[int(np.log2(self.M**(1/2))),-1],order='C')
        BICM_int=BICM_int[argtmp,:]
        BICM_int=np.ravel(BICM_int,order='F')
        self.interleaver_check(EsNodB,BICM_int,frozen_bits)
        return BICM_int
    
    def interleaver_check(self,EsNodB,BICM_int,frozen_bits):
        BICM_deint=np.argsort(BICM_int)
        tmp=make_BMI_list(EsNodB,self.M)
        for a in range(len(tmp)):
            tmp[a]=self.const.calc_J_inv(tmp[a])
        #print(tmp)
        gamma=np.tile(tmp,self.N//int(np.log2(self.M)))
        xi=np.log(gamma)
        xi=xi[BICM_deint]
        if np.all(np.sort(np.argsort(xi)[:len(xi)//2])==frozen_bits)==False:
            print("interleaver error!!")
    
    def adaptive_BICM(self,EsNodB):
                
        count=0
        BICM_int=np.arange(self.N)
        #BICM_int_new=np.arange(cst.N)
        while True:
            count+=1
            #print("count:",count)
            BICM_int_new=self.construction(BICM_int,EsNodB)
            if np.all(BICM_int_new==BICM_int)==True:
                break
            else:
                BICM_int=BICM_int_new
        
        BICM_deint=np.argsort(BICM_int)
        #bit_reversal_sequence=self.cd.bit_reversal_sequence
        #BICM_int=BICM_int[bit_reversal_sequence]

        return BICM_int,BICM_deint
           
    def main_func(self,EsNodB):
        
        if self.adaptive_intlv==True and self.cd.design_SNR!=EsNodB:
            self.BICM_int,self.BICM_deint=self.adaptive_BICM(EsNodB)
            #BICM check
            if len(self.BICM_int)!=self.N:
                print("BICM_Error")
                
            for i in range(self.N):
                if np.any(i==self.BICM_int)==False:
                    print("BICM_error")
        
        if self.cd.decoder_ver==2:
            #print("pass")
            CRC_len=len(self.cd.CRC_polynomial)-1  
            frozen_bits,info_bits=self.const.main_const(self.N,self.K+CRC_len,EsNodB,self.M,BICM_int=self.BICM_int,type=self.type)
        else:
            #print('pass2')
            frozen_bits,info_bits=self.const.main_const(self.N,self.K,EsNodB,self.M,BICM_int=self.BICM_int,type=self.type)
            
        #check
        for i in range(self.N):
            if (np.any(i==frozen_bits) or np.any(i==info_bits))==False:
                raise ValueError("The frozen set or info set is overlapped")
                
        self.cd.design_SNR=EsNodB    
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
        cwd=cwd[self.BICM_int]
        TX_conste=self.modem.modulate(cwd)        
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,No)
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
    
    MAXCNT=10
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
            const.main_const(mysys.N,mysys.K,EsNodB,mysys.M,BICM_int=mysys.BICM_int,type=mysys.type)   
    '''
# %%
