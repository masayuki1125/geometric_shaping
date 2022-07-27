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
class Mysystem:
    def __init__(self,M,K):
        #make instance
        self.M=M
        self.K=K
        #self.N=self.K*int(np.log2(self.M))
        self.N=self.K*2
        const_var=3 #1:MC 2:iGA 3:RCA
        
        self.type=1#1:separated scheme 2:Block intlv(No intlv in arikan polar decoder) 3:No intlv(Block intlv in arikan polar decoder) 4:rand intlv
        
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
        
        #interleaver design
        self.BICM_int,self.BICM_deint=self.make_BICM_int(self.N,self.M,self.type)
        #np.savetxt("BICM_int",self.BICM_int,fmt="%.0f")
        
        self.filename=self.make_filename()
    
    def make_filename(self):
        #filename
        filename="polar_{}_{}_{}QAM".format(self.N,self.K,self.M)
        filename=filename+self.const_name
        if self.cd[0].systematic_polar==True:
            filename="systematic_"+filename
            
        #decoder type
        if self.cd[0].decoder_ver==0:
            filename+="_SC"
        elif self.cd[0].decoder_ver==2:
            filename+="_CA_SCL"
                    
        #provisional
        filename+="_type{}".format(self.type)
        
        #output filename to confirm which program I run
        print(filename)
        
        return filename
    
    def reverse_bits(self,N):
        res=np.zeros(N,dtype=int)

        for i in range(N):
            tmp=format (i,'b')
            tmp=tmp.zfill(int(np.log2(N))+1)[:0:-1]
            #print(tmp) 
            res[i]=self.reverse(i,N)
        return res

    @staticmethod
    def reverse(n,N):
        tmp=format (n,'b')
        tmp=tmp.zfill(int(np.log2(N))+1)[:0:-1]
        res=int(tmp,2) 
        return res
        
    def make_BICM_int(self,N,M,type):
        
        BICM_int=np.arange(N,dtype=int)
        #modify BICM int from simplified to arikan decoder order
        bit_reversal_sequence=self.reverse_bits(N)
        BICM_int=BICM_int[bit_reversal_sequence]
        
        if type==1:#1:separated scheme 
            print(BICM_int%4)
            #np.savetxt("bicm_int",BICM_int%4,fmt="%.0f")
            #print("pass")
            pass
        elif type==2:#2:Block intlv(No intlv in arikan polar decoder) 
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            BICM_int[0]=np.sort(BICM_int[0])
            BICM_int[1]=np.sort(BICM_int[1])
            BICM_int=np.ravel(BICM_int,order='C')
            #print(BICM_int)
            
        elif type==3:#3:No intlv(Block intlv in arikan polar decoder) 
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            BICM_int=np.ravel(BICM_int,order='F')
        elif type==4:#4:rand intlv
            tmp,_=make_BICM(N,4)
            BICM_int=BICM_int[tmp]
        elif type==5:#2:Block intlv(No intlv in arikan polar decoder) 
            tmp=np.arange(N//int(np.log2(M**(1/2))),dtype=int)
            random.shuffle(tmp)
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            for i in range (int(np.log2(M**(1/2)))):
                BICM_int[i]=BICM_int[i][tmp]
            BICM_int=np.ravel(BICM_int,order='C')
        elif type==6:
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            for i in range (int(np.log2(M**(1/2)))):
                tmp=np.arange(N//int(np.log2(M**(1/2))),dtype=int)
                random.shuffle(tmp)
                BICM_int[i]=BICM_int[i][tmp]
            BICM_int=np.ravel(BICM_int,order='C')
            
        #elif type==7:
            #BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            #for i in range (int(np.log2(M**(1/2)))):
                #tmp=np.arange(N//int(np.log2(M**(1/2))),dtype=int)
                #random.shuffle(tmp)
                #BICM_int[i]=BICM_int[i][tmp]
            #BICM_int=np.ravel(BICM_int,order='C')
            
        else:
            print("interleaver type error")
            
        
            
        BICM_deint=np.argsort(BICM_int)
        
        #np.savetxt("deint",BICM_deint,fmt='%.0f')
        
        #print(BICM_int)
        #print(BICM_deint%4)
        
        return BICM_int,BICM_deint
           
    def main_func(self,EsNodB):
        #すべてのデコーダ一律で凍結ビットを得る
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
            
            #print("len",len(info_bits_sep))
            
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
            
            #if i==0:
                #info_use=np.concatenate([info_use,info_sep])
                #print(len(info_use),"info_use")
                
        cwd=cwd[self.BICM_int] #interleave
        TX_conste=self.modem.modulate(cwd)
        #print(TX_conste)
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,No)
        Lc=Lc[self.BICM_deint] #de interleave
            
        EST_info=np.empty(0)
        for i in range(self.enc_num):
            EST_info_sep=self.dc[i].polar_decode(Lc[i*self.N_sep:(i+1)*self.N_sep])
            EST_info=np.concatenate([EST_info,EST_info_sep])
            
            #if i==0:
                #EST_info=self.dc[i].polar_decode(Lc[i*self.N_sep:(i+1)*self.N_sep])
        
        #print(len(info_use))
        #print(len(EST_info))
        
        return info,EST_info
    
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