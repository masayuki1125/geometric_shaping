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
        self.BICM=True
        const_var=3 #1:MC 2:iGA 3:RCA
        
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
        self.enc_num=int(np.log2(M**(1/2)))
    
        if self.N%self.enc_num!=0:
            print("encoder is not mod(enc_num)")
        self.N_sep=self.N//self.enc_num
        self.K_sep=self.K//self.enc_num #とりあえず適当なKで初期化する
       
        #coding
        self.cd=[]
        self.ec=[]
        self.dc=[]
        for i in range(self.enc_num):
            self.cd+=[polar_construction.coding(self.N_sep,self.K_sep)]
            
            #CRCを調整する
            if self.enc_num==2:
                self.cd[i].CRC_polynomial =np.array([1,0,0,1,1])
            elif self.enc_num==4:
                self.cd[i].CRC_polynomial =np.array([1,0,1,0,0,1,1,0,1])
            else:
                print("unsupported encoder number")
            
            self.ec+=[polar_encode.encoding(self.cd[i])]
            self.dc+=[polar_decode.decoding(self.cd[i],self.ec[i])]
            
        #get decoder var
        self.decoder_ver=self.cd[0].decoder_ver
       
        #modulation
        self.modem=modulation.QAMModem(self.M)
        #channel
        self.ch=AWGN._AWGN()
        
        #filename
        self.filename="polar_code_{}_{}_{}_4".format(self.N,self.K,self.M)
        self.filename=self.filename+const_name
        
        #add filename to the number of types 
        self.filename+="_{}".format(self.type)
        
        #output filename to confirm which program I run
        print(self.filename)
        
    def adaptive_BICM(self,EsNodB):
        #block interleaver
        from capacity_estimation.calc_capacity import make_BMI_list 
        tmp=make_BMI_list(EsNodB,self.M)
        print(tmp)
        seq_of_channels=np.argsort(tmp[:len(tmp)//2])
        #print(seq_of_channels)
        num_of_channels=len(seq_of_channels)
        
        seq=np.arange(self.N,dtype=int)
        seq=np.reshape(seq,[num_of_channels,-1],order='C')
        seq=seq[seq_of_channels,:]
        seq=np.ravel(seq,order='F')
        #print(seq_of_channels)
        print(seq)
        self.BICM_deint=seq
        self.BICM_int=np.argsort(self.BICM_deint)
           
    def main_func(self,EsNodB):
        #adaptive change of BICM interleaver
        if self.M!=4:
            if self.type==3 or self.type==4:
                self.adaptive_BICM(EsNodB)
        
        if self.decoder_ver==2:
            self.CRC_len=len(self.cd[0].CRC_polynomial)-1  
            frozen_bits,info_bits=self.const.main_const_sep(self.N,self.K+self.CRC_len,EsNodB,self.M,BICM_int=self.BICM_int)
        else:
            frozen_bits,info_bits=self.const.main_const_sep(self.N,self.K,EsNodB,self.M,BICM_int=self.BICM_int)
            
        #check
        for i in range(self.N):
            if (np.any(i==frozen_bits) or np.any(i==info_bits))==False:
                raise ValueError("The frozen set or info set is overlapped")
        
        #凍結ビットによって、情報ビットの長さを変える
        for i in range(self.enc_num):
            a=frozen_bits>=i*self.N_sep
            #print(a)
            b=frozen_bits<(i+1)*self.N_sep
            #print(b)
            c=a*b
            #print(c)
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            
            d=info_bits>=i*self.N_sep
            e=info_bits<(i+1)*self.N_sep
            f=d*e
            
            frozen_bits_sep=frozen_bits[c]%self.N_sep
            info_bits_sep=info_bits[f]%self.N_sep
            
            #frozen and info chech
            if (len(frozen_bits_sep)+len(info_bits_sep))!=self.N_sep:
                print("frozen_bit and info_bit len err")
            
            for j in range(self.N_sep):
                if np.any(frozen_bits_sep==i):
                    break
                elif np.any(info_bits_sep==i):
                    break
                else:
                    print(j)
                    print("frozen or info is missed")
                    print(frozen_bits_sep)
                    print(info_bits_sep)
                    from IPython.core.debugger import Pdb; Pdb().set_trace()
                    
            self.cd[i].design_SNR=EsNodB    
            self.cd[i].frozen_bits=frozen_bits_sep
            self.ec[i].frozen_bits=frozen_bits_sep
            self.dc[i].frozen_bits=frozen_bits_sep
            self.cd[i].info_bits=info_bits_sep
            self.ec[i].info_bits=info_bits_sep
            self.dc[i].info_bits=info_bits_sep
            
            if self.decoder_ver==2:
                res=len(info_bits_sep)-self.CRC_len
            else:
                res=len(info_bits_sep)
            
            self.cd[i].K=res
            self.ec[i].K=res
            self.dc[i].K=res
            
            #print(res)
            
            #print(len(info_bits_sep)/self.N_sep)
            
        #for iGA and RCA and monte_carlo construction
                
        EsNo = 10 ** (EsNodB / 10)
        No=1/EsNo

        info=np.empty(0,dtype=int)
        cwd=np.empty(0,dtype=int)
        
        info_use=np.empty(0,dtype=int)
        for i in range(self.enc_num):
            info_sep,cwd_sep=self.ec[i].polar_encode()
            #print(len(info_sep))
            info=np.concatenate([info,info_sep])
            cwd=np.concatenate([cwd,cwd_sep])
            
            if i==3:
                info_use=np.concatenate([info_use,info_sep])
                #print(len(info_use),"info_use")
                
        
        if self.BICM==True:
            cwd=cwd[self.BICM_int]
        
        TX_conste=self.modem.modulate(cwd)
        #print(TX_conste)
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,No)
        if self.BICM==True:
            Lc=Lc[self.BICM_deint]
            
        EST_info=np.empty(0)
        for i in range(self.enc_num):
            EST_info_sep=self.dc[i].polar_decode(Lc[i*self.N_sep:(i+1)*self.N_sep])
            EST_info=np.concatenate([EST_info,EST_info_sep])
            
        i=3
        EST_info=self.dc[i].polar_decode(Lc[i*self.N_sep:(i+1)*self.N_sep])
        
        #print(len(info_use))
        #print(len(EST_info))
        
        return info_use,EST_info
    
if __name__=='__main__':
    K=512 #symbol数
    M=256
    
    EsNodB=19.0
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