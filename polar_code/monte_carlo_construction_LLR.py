#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pickle
import sys
import multiprocessing
import math
import matplotlib.pyplot as plt
import os

sys.path.append(os.path.abspath(".."))
from modulation.BICM import make_BICM, make_BICM_multi
from modulation.modulation import QAMModem
from modulation.modulation import PSKModem

# In[75]:

#rate_0のPolar符号を作る
class Myconstruction:
    def __init__(self,N,M,design_SNR,**kwargs):
        
        if kwargs.get('BICM_int') is not None:
            #print("use BICM")
            self.BICM_int=kwargs.get("BICM_int")
            self.BICM=True
            self.BICM_deint=np.argsort(self.BICM_int)
            self.BICM=True
        else:
            print("no BICM_int input error")
            
        if kwargs.get("type") is not None: 
            self.type=kwargs.get("type")
            if self.type==6:
                print("specific program is needed")
            elif self.type==7:
                print("specific program is needed")
        else:
            print("no type input error")
            
        
        self.M=M
        self.N=N
        self.design_SNR=design_SNR
            
        #modulation
        self.modem=QAMModem(self.M)
        
    #Rate-0 Polar construction
    @staticmethod
    def generate_information(N):
        #generate information
        info=np.random.randint(0,2,N)
        return info

    def encode(self,u_message):
        """
        Implements the polar transform on the given message in a recursive way (defined in Arikan's paper).
        :param u_message: An integer array of N bits which are to be transformed;
        :return: codedword -- result of the polar transform.
        """
        u_message = np.array(u_message)

        if len(u_message) == 1:
            codeword = u_message
        else:
            u1u2 = np.logical_xor(u_message[::2] , u_message[1::2])
            u2 = u_message[1::2]

            codeword = np.concatenate([self.encode(u1u2), self.encode(u2)])
        return codeword

    def reverse_bits(self,N):
        res=np.zeros(N,dtype=int)

        for i in range(N):
            tmp=format (i,'b')
            tmp=tmp.zfill(int(math.log2(N))+1)[:0:-1]
            #print(tmp) 
            res[i]=self.reverse(i,N)
        return res

    @staticmethod
    def reverse(n,N):
        tmp=format (n,'b')
        tmp=tmp.zfill(int(math.log2(N))+1)[:0:-1]
        res=int(tmp,2) 
        return res

    def polar_encode(self,N):
    
        bit_reversal_sequence=self.reverse_bits(N)
        info=self.generate_information(N)
        cwd=self.encode(info[bit_reversal_sequence])
        
        return info,cwd

    #polar decode
    @staticmethod
    def chk(llr_1,llr_2):
        
        def max_str(a,b):
            res=max(a,b)
            res+=np.log(1+np.exp(-1*abs(a-b)))
            return res
        
        CHECK_NODE_TANH_THRES=37
        res=np.zeros(len(llr_1))
        for i in range(len(res)):

            if abs(llr_1[i]) > CHECK_NODE_TANH_THRES and abs(llr_2[i]) > CHECK_NODE_TANH_THRES:
                if llr_1[i] * llr_2[i] > 0:
                    # If both LLRs are of one sign, we return the minimum of their absolute values.
                    res[i]=min(abs(llr_1[i]), abs(llr_2[i]))
                else:
                    # Otherwise, we return an opposite to the minimum of their absolute values.
                    res[i]=-1 * min(abs(llr_1[i]), abs(llr_2[i]))
                #tmp=np.log(np.exp(-1*res)+np.exp(llr_1+llr_2-res))-np.log(np.exp(llr_1)+np.exp(llr_2))
                res[i]=res[i]+max_str(-1*res[i],llr_1[i]+llr_2[i]-res[i])-max_str(llr_1[i],llr_2[i])
                
            
            else:
                res[i]= 2 * np.arctanh(np.tanh(llr_1[i] / 2, ) * np.tanh(llr_2[i] / 2))
        return res

    def SC_decoding(self,N,Lc,info):
        #initialize constant
        itr_num=int(math.log2(N))    
        llr=np.zeros((itr_num+1,N))
        EST_codeword=np.zeros((itr_num+1,N))
        llr[0]=Lc

        #put decoding result into llr[logN]

        depth=0
        length=0
        before_process=0# 0:left 1:right 2:up 3:leaf

        while True:

            #left node operation
            if before_process!=2 and before_process!=3 and length%2**(itr_num-depth)==0:
                depth+=1
                before_process=0

                tmp1=llr[depth-1,length:length+2**(itr_num-depth)]
                tmp2=llr[depth-1,length+2**(itr_num-depth):length+2**(itr_num-depth+1)]

                llr[depth,length:length+N//(2**depth)]=self.chk(tmp1,tmp2)

            #right node operation 
            elif before_process!=1 and length%2**(itr_num-depth)==2**(itr_num-depth-1):
                
                #print(length%2**(self.itr_num-depth))
                #print(2**(self.itr_num-depth-1))
                
                depth+=1
                before_process=1
                
                tmp1=llr[depth-1,length-2**(itr_num-depth):length]
                tmp2=llr[depth-1,length:length+2**(itr_num-depth)]

                llr[depth,length:length+2**(itr_num-depth)]=tmp2+(1-2*EST_codeword[depth,length-2**(itr_num-depth):length])*tmp1

            #up node operation
            elif before_process!=0 and length!=0 and length%2**(itr_num-depth)==0:#今いるdepthより一個下のノードから、upすべきか判断する
            
                tmp1=EST_codeword[depth+1,length-2**(itr_num-depth):length-2**(itr_num-depth-1)]
                tmp2=EST_codeword[depth+1,length-2**(itr_num-depth-1):length]

                EST_codeword[depth,length-2**(itr_num-depth):length-2**(itr_num-depth-1)]=(tmp1+tmp2)%2
                EST_codeword[depth,length-2**(itr_num-depth-1):length]=tmp2

                depth-=1
                before_process=2
            
            else:
                print("error!")

            #leaf node operation
            if depth==itr_num:
            
                #for monte carlo construction
                EST_codeword[depth,length]=info[length]
                
                length+=1 #go to next length

                depth-=1 #back to depth
                before_process=3
                
                #print(llr[itr_num,length-1])
                #print(EST_codeword[itr_num,length-1])
                #from IPython.core.debugger import Pdb; Pdb().set_trace()
                
            
            if length==N:
                break
        
        #for monte calro construction  
        res=llr[itr_num]
        
        return res
    
    #AWGN channel
    @staticmethod
    def add_AWGN(const,No):
        noise = np.random.normal(0, math.sqrt(No / 2), (len(const))) + 1j * np.random.normal(0, math.sqrt(No / 2), (len(const)))
        return const+noise
    
    def main_func(self):
        #adaptive dicision of frozen bits
        EsNo = 10 ** (self.design_SNR / 10)
        No=1/EsNo
        
        info,cwd=self.polar_encode(self.N)
        if self.BICM==True:
            cwd=cwd[self.BICM_int]
        TX_conste=self.modem.modulate(cwd)
        RX_conste=self.add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,No)
        if self.BICM==True:
            Lc=Lc[self.BICM_deint]
        llr=self.SC_decoding(self.N,Lc,info)
        
        return info,llr
    
    def main_func_sep(self):
        '''
        function for type1 encoder 
        using type3 interleaver
        '''
        
        #adaptive dicision of frozen bits
        EsNo = 10 ** (self.design_SNR / 10)
        No=1/EsNo
        
        self.enc_num=int(np.log2(self.M**(1/2)))
        self.N_sep=self.N//self.enc_num
        
        #encode
        info=np.empty(0,dtype=int)
        cwd=np.empty(0,dtype=int)
        
        for _ in range(self.enc_num):
            info_sep,cwd_sep=self.polar_encode(self.N_sep)
            #print(len(info_sep))
            info=np.concatenate([info,info_sep])
            cwd=np.concatenate([cwd,cwd_sep])

        #channel
        cwd=cwd[self.BICM_int] #interleave
        TX_conste=self.modem.modulate(cwd)
        RX_conste=self.add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,No)
        Lc=Lc[self.BICM_deint] #de interleave
        
        #decode
        llr=np.empty(0)
        for i in range(self.enc_num):
            llr_sep=self.SC_decoding(self.N_sep,Lc[i*self.N_sep:(i+1)*self.N_sep],info[i*self.N_sep:(i+1)*self.N_sep])
            llr=np.concatenate([llr,llr_sep])
        
        return info,llr
        
class monte_carlo():
      
  def main_const(self,N,K,design_SNR,M,**kwargs):
    #check
    #design_SNR=100
    #get from kwargs
    
    const=Myconstruction(N,M,design_SNR,**kwargs)
    
    #check
    if N!=const.N:
        print("monte_carlo codelength error!!")
    
    #pickle class    
    dumped=pickle.dumps(const)
    
    c=self.output(dumped)
    
    tmp=np.argsort(c)#[::-1]
    frozen_bits=np.sort(tmp[:N-K])
    info_bits=np.sort(tmp[N-K:])
    
    return frozen_bits,info_bits

  @staticmethod
  def make_d(dumped):
    #initial constant
    const=pickle.loads(dumped)
    
    epoch=10**6//multiprocessing.cpu_count()
    #print(multiprocessing.cpu_count())
    #print(epoch)
    
    c=np.zeros(const.N)
    for _ in range(epoch):
        
      if const.type==1:
        info,llr=const.main_func_sep()
      else:
        info,llr=const.main_func()
      #print(llr)
      #from IPython.core.debugger import Pdb; Pdb().set_trace()
      
      #d=np.zeros(len(llr))
        #print(llr)
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
      #d[(2*info-1)*llr<0]=0 #no error occur
      #d[(2*info-1)*llr>=0]=1 # error occur
      #d+=llr
      
      res_llr=-1*(2*info-1)*llr
      
      c=c+res_llr
      
    #normarize
    c/=epoch
    
    return c

  def make_c(self,dumped):
    const=pickle.loads(dumped)
    
    c=np.zeros(const.N)
    multi_num=multiprocessing.cpu_count() #the number of multiprocessing 
    
    inputs=[]
    for _ in range(multi_num):
      inputs+=[dumped]
    
    #multiprocessing
    with multiprocessing.Pool(processes=multi_num) as pool:
      res = pool.map(self.make_d,inputs)
    
    for i in range(multi_num):
      c=c+res[i]
      
    #normarize
    c/=multi_num
      
    return c
  
  def output(self,dumped):
    
    # directory make
    home=os.environ['HOME']
    current_directory=home+"/Dropbox/programming/geometric_shaping/polar_code"
    #current_directory=os.getcwd()
    dir_name="monte_carlo_construction_LLR"
    dir_name=current_directory+"/"+dir_name
    
    try:
      os.makedirs(dir_name)
    except FileExistsError:
      #print("file exists!")
      pass
    
    const=pickle.loads(dumped)
    
      
    filename="{}QAM_{}_{}_type{}".format(const.M,const.N,const.design_SNR,const.type)
    
    #if file exists, then load txt file
    filename=dir_name+"/"+filename
    
    try:
      c=np.loadtxt(filename)
    except (OSError,FileNotFoundError):
      print("make frozen bits!")
      print(filename)
      c=self.make_c(dumped)
      #export file
      np.savetxt(filename,c)

    return c

#%%
if __name__=="__main__":
    
    def reverse_bits(N):
        res=np.zeros(N,dtype=int)

        for i in range(N):
            tmp=format (i,'b')
            tmp=tmp.zfill(int(math.log2(N))+1)[:0:-1]
            #print(tmp) 
            res[i]=reverse(i,N)
        return res
    
    def reverse(n,N):
        tmp=format (n,'b')
        tmp=tmp.zfill(int(math.log2(N))+1)[:0:-1]
        res=int(tmp,2) 
        return res
    
    #BICM interleaver
    def make_BICM_int(N,M,type):
        
        BICM_int=np.arange(N,dtype=int)
        
        if type==1:#1:separated scheme 
            #type3と同じで、Blockインターリーブする
            print("using block interleaver!")
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            BICM_int=np.ravel(BICM_int,order='F')
        elif type==2:#2:Block intlv in arikan polar decoder
            
            pass
            #BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            #BICM_int[0]=np.sort(BICM_int[0])
            #BICM_int[1]=np.sort(BICM_int[1])
            #BICM_int=np.ravel(BICM_int,order='C')
            #print(BICM_int)
            
        elif type==3:#3:Block intlv in arikan polar decoder
            bit_reversal_sequence=reverse_bits(N)
            BICM_int=BICM_int[bit_reversal_sequence]
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            BICM_int=np.ravel(BICM_int,order='F')
        elif type==4:#4:rand intlv
            #modify BICM int from simplified to arikan decoder order
            bit_reversal_sequence=reverse_bits(N)
            BICM_int=BICM_int[bit_reversal_sequence]
            tmp,_=make_BICM(N)
            BICM_int=BICM_int[tmp]
        elif type==5:#2:No intlv +rand intlv for each channel
            #modify BICM int from simplified to arikan decoder order
            bit_reversal_sequence=reverse_bits(N)
            BICM_int=BICM_int[bit_reversal_sequence]
            tmp,_=make_BICM_multi(N//int(np.log2(M**(1/2))),int(np.log2(M**(1/2))))
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='F')
            for i in range (int(np.log2(M**(1/2)))):
                BICM_int[i]=BICM_int[i][tmp[i]]
            BICM_int=np.ravel(BICM_int,order='F')
        elif type==6:#凍結ビットを低SNRに設定する
            BICM_int,_=adaptive_BICM(N,EsNodB,const)
            pass#specific file is needed
        elif type==7:#compound polar codes
            print("err type7")
            pass #specific file is needed
            
        else:
            print("interleaver type error")
        BICM_deint=np.argsort(BICM_int)
        #np.savetxt("deint",BICM_deint,fmt='%.0f')
        #print(BICM_int)
        #print(BICM_deint) 
        return BICM_int,BICM_deint
    
    def make_BMI_list(EsNodB,M):
        # directory make
        current_directory="/home/kaneko/Dropbox/programming/geometric_shaping/capacity_estimation"
        #current_directory=os.getcwd()
        dir_name="BMI"
        dir_name=current_directory+"/"+dir_name
        
        filename="{}QAM_{}".format(M,EsNodB)
        
        #if file exists, then load txt file
        filename=dir_name+"/"+filename
        
        #try:
        res=np.loadtxt(filename)
        #except FileNotFoundError:
        return res

    
    
    #以下の関数はType6用の関数
    def construction(N,K,M,BICM_int,EsNodB,const):
        frozen_bits,info_bits=const.main_const(N,K,EsNodB,M,BICM_int=BICM_int)
        
        #print(frozen_bits)
        BICM_int=np.concatenate([frozen_bits,info_bits])
        tmp=make_BMI_list(EsNodB,M)
        argtmp=np.argsort(tmp[:len(tmp)//2])
        #print(tmp)
        #print(argtmp)
        BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
        BICM_int=BICM_int[argtmp,:]
        BICM_int=np.ravel(BICM_int,order='F')
        interleaver_check(N,M,EsNodB,BICM_int,frozen_bits)
        return BICM_int
        
    def interleaver_check(N,M,EsNodB,BICM_int,frozen_bits):
        BICM_deint=np.argsort(BICM_int)
        tmp=make_BMI_list(EsNodB,M)
        for a in range(len(tmp)):
            tmp[a]=calc_C_inv(tmp[a])
        #print(tmp)
        gamma=np.tile(tmp,N//int(np.log2(M)))
        xi=np.log(gamma)
        xi=xi[BICM_deint]
        if np.all(np.sort(np.argsort(xi)[:len(xi)//2])==frozen_bits)==False:
            print("interleaver error!!")

    def adaptive_BICM(N,EsNodB,const):
                
        count=0
        BICM_int=np.arange(N)
        #BICM_int_new=np.arange(cst.N)
        while True:
            count+=1
            print("count:",count)
            BICM_int_new=construction(N,K,M,BICM_int,EsNodB,const)
            if np.all(BICM_int_new==BICM_int)==True:
                break
            else:
                BICM_int=BICM_int_new
        
        BICM_deint=np.argsort(BICM_int)
        #bit_reversal_sequence=self.cd.bit_reversal_sequence
        #BICM_int=BICM_int[bit_reversal_sequence]

        return BICM_int,BICM_deint
    
    def calc_C_inv(I):
        '''
        input:
        I:mutual information
        output:
        gamma:channel SNR Es/No
        ----
        referrence:
        POLAR CODES FOR ERROR CORRECTION:
        ANALYSIS AND DECODING ALGORITHMS
        p37
        (4.5)
        '''
        if I>1 or I<0:
            print("I is err")
        
        a1=1.09542
        b1=0.214217
        c1=2.33727
        a2=0.706692
        b2=0.386013
        c2=-1.75017
        I_thresh=0.3646
            
        if I<I_thresh:
            sigma=a1*I**2+b1*I+c1*I**(1/2)
        else:
            sigma=-a2*np.log(b2*(1-I))-c2*I
            
        gamma=sigma**2/8
        #gamma_dB=10*math.log10(gamma)
        return gamma

    
    

    K=512
    N=2*K
    
    type_list=[3,4,5]
    M_list=[16,256]
    EsNodB_list=np.arange(4,10,0.5)
    for type in type_list:
        for M in M_list:
            #インターリーバ設計
            BICM_int,_=make_BICM_int(N,M,type)
            
            for EsNodB in EsNodB_list:  
                if M==16:
                    EsNodB+=0
                elif M==256:
                    EsNodB+=10 
                const=monte_carlo()
                const.main_const(N,K,EsNodB,M,BICM_int=BICM_int,type=type)   
    