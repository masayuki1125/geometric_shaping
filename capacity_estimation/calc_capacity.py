#!/usr/bin/env python
# coding: utf-8

# In[100]:


# In[8]:
import math
import numpy as np
import cupy as cp
#pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
#cp.cuda.set_allocator(pool.malloc)
import sys
import os
from scipy.stats import norm
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))

from modulation.modulation import QAMModem
from modulation.modulation import PSKModem


# In[101]:


def add_AWGN_GPU(constellation,No):
  # AWGN雑音の生成
  noise = cp.random.normal(0, math.sqrt(No / 2), (len(constellation)))           + 1j * cp.random.normal(0, math.sqrt(No / 2), (len(constellation)))

  # AWGN通信路 = 送信シンボル間干渉が生じないような通信路で送信
  RX_constellation = constellation + noise

  # 以下のprint関数の出力を表示すると、Noとほぼ一致するはず
  #print(cp.dot(noise[0, :], cp.conj(noise[0, :]))/bit_num)

  return RX_constellation


# In[102]:


def make_AMI(EsNodB,M):
    modem_ver=1 #0:QAMModem 1:PSKModem
    
  
    EsNo = 10 ** (EsNodB / 10)
    No=1/EsNo
    count_num=1000000

    #make info matrices
    info=cp.random.randint(0,M,count_num)

    #make constellation
    if M!=2:
        if modem_ver==0:
            modem=QAMModem(M)
        elif modem_ver==1:
            modem=PSKModem(M)
        else:
            print("modem version error")
    
    
        tmp=modem.code_book
        symbol=cp.zeros(M,dtype=complex)
        for i in tmp:
            symbol[modem.bin2de(i)]=tmp[i]
        
    elif M==2:
        symbol=cp.array([-1,1])
        
    mat_symbol=cp.tile(symbol,(count_num,1))
    const=cp.take_along_axis(mat_symbol,info[:,None],axis=1)[:,0]
    

    #if cp.any(symbol==const)!=True:
        #print("error")
        #print(symbol)
        #print(const)

    RX_const=add_AWGN_GPU(const,No)
        
    num=cp.sum(cp.exp(-1*cp.abs(np.tile(RX_const,(len(symbol),1))-symbol.reshape(-1,1))**2/No),axis=0)
    
    den=cp.exp(-1*cp.abs(RX_const-const)**2/No)
    H=cp.sum(cp.log2(num/den))
    H/=count_num
    res=math.log2(M)-H
    return res

def make_BMI(EsNodB,M):
    modem_ver=0 #0:QAMModem 1:PSKModem
    each_res=False #default:false
    
    EsNo = 10 ** (EsNodB / 10)
    No=1/EsNo
    count_num=100000
        
    result=0
    
    if each_res==True:
        result=np.zeros(int(math.log2(M)))
    else:    
        result=0
    
    for i in range(1):
    
        #make info matrices
        info=cp.random.randint(0,M,count_num)
        #print(info)
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        #info=cp.zeros(count_num,dtype=int)
        #make constellation
        if modem_ver==0:
            modem=QAMModem(M)
        elif modem_ver==1:
            modem=PSKModem(M)
        else:
            print("modem version error")
            
        tmp=modem.code_book
        
        symbol=cp.zeros(M,dtype=complex)
        for i in tmp:
            symbol[modem.bin2de(i)]=tmp[i]
            #print(modem.bin2de(i))
            #print(tmp[i])
            #print("next")
        
        mat_symbol=cp.tile(symbol,(count_num,1))
        const=cp.take_along_axis(mat_symbol,info[:,None],axis=1)[:,0]
        #print(const)
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        RX_const=add_AWGN_GPU(const,No)

        #bitごとの0のシンボルと1のシンボルを出す
        ones=cp.array(modem.ones)
        zeros=cp.array(modem.zeros)
        ones_zeros=cp.stack([zeros,ones])
        #print(ones_zeros)
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        #print(ones)
        #print(zeros)
        
        #for a,b in zip(ones,zeros):
            #print("zip")
            #print(a,b)
        #print(info)
        #print(ones)
        H=0
        res=np.zeros(int(math.log2(M)))
        
        for i in range(0,int(math.log2(M)))[::-1]:
            ith_bits=((info)//(2**i))%2
            #print("bits",ith_bits[0:30])
            #print(const[0:10])
            
            ones_zeros_i=ones_zeros[:,int(math.log2(M))-i-1,:] #ここのインデックスが間違っていた
            #print(ones_zeros.shape)#(2,log2(M),M/2)
            #print(ones_zeros_i.shape)#(2,M/2)
            
            #print(ones_zeros_i) #check#ここが間違っていた
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            
            mat_ones_zeros_i=np.tile(ones_zeros_i,(len(ith_bits),1,1))
            #print(mat_ones_zeros_i.shape) #(count_num,2,M/2)
            #print(mat_ones_zeros_i[0])
            
            res_ones_zeros_i=cp.take_along_axis(mat_ones_zeros_i,ith_bits[:,None,None],axis=1)[:,0,:].T
            #print(res_ones_zeros_i.shape) #(count_num,1,M/2)
            #print(res_ones_zeros_i)
            #print(res_ones_zeros_i)
            #del mat_ones_zeros_i
            
            #res_ones_zeros_i=res_ones_zeros_i[:,0,:]
            #print(res_ones_zeros_i.shape) #(count_num,M/2)
            
            #res_ones_zeros_i=res_ones_zeros_i.T
            #print(RX_const[0])
            #print(cp.exp(-1*cp.abs(np.tile(RX_const,(len(symbol),1))-symbol.reshape(-1,1))**2/No))
            #print(cp.exp(-1*cp.abs(np.tile(RX_const,(len(ones[0]),1))-res_ones_zeros_i)**2/No))
            
            num=cp.sum(cp.exp(-1*cp.abs(np.tile(RX_const,(len(symbol),1))-symbol.reshape(-1,1))**2/No),axis=0)
            den=cp.sum(cp.exp(-1*cp.abs(np.tile(RX_const,(len(ones[0]),1))-res_ones_zeros_i)**2/No),axis=0)
            #print("num",num)
            #print("den",den)
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            H=cp.sum(cp.log2(num/den))
            tmp=1-H/count_num
            #print(tmp)
            res[i]+=tmp

        #H/=count_num
        #res=math.log2(M)-H
        #print(res)
        
        if each_res==True:
            result+=res[::-1]
        else:    
            result+=np.sum(res)

    return result

def make_BMI_list(EsNodB,M):
    # directory make
    current_directory="/home/kaneko/Dropbox/programming/geometric_shaping/capacity_estimation"
    #current_directory=os.getcwd()
    dir_name="BMI"
    dir_name=current_directory+"/"+dir_name
    
    try:
      os.makedirs(dir_name)
    except FileExistsError:
      pass
      
    filename="{}PSK_{}".format(M,EsNodB)
    
    #if file exists, then load txt file
    filename=dir_name+"/"+filename
    
    try:
      res=np.loadtxt(filename)
    except FileNotFoundError:
      print("make_BMI",EsNodB,M)
      res=make_BMI(EsNodB,M)
      #export file
      np.savetxt(filename,res)
      
    return res


# In[104]:
if __name__=='__main__':
    
    SNR_range=np.arange(0,20,0.5)
    M_list=[256]
    
    '''
    BMI_list=np.zeros(len(SNR_range))
    for M in M_list:
        for i,EsNodB in enumerate(SNR_range):
            make_BMI_list(EsNodB,M)
    '''
    
    print(SNR_range)

    BMI_list=np.zeros(len(SNR_range))

    for M in M_list:
        '''
        AMI_list=np.zeros(len(SNR_range))
        filename="AMI_{}PSK".format(M)
        for i,EsNodB in enumerate(SNR_range):
            AMI_list[i]=make_AMI(EsNodB,M)
            
            with open(filename,'w') as f:
                for i in range(len(SNR_range)):
                    print(str(SNR_range[i]),str(AMI_list[i]),file=f)       
        '''            
        BMI_list_a=np.zeros(len(SNR_range))
        #BMI_list_b=np.zeros(len(SNR_range))
        filename="BMI_{}QAM".format(M)
        for i,EsNodB in enumerate(SNR_range):
            #BMI_list[i]=make_BMI(EsNodB,M)
            BMI_list_a[i]=make_BMI(EsNodB,M)
            #print(make_BMI(EsNodB,M))
        
        #print(BMI_list_a)
        #print(BMI_list_b)
         
        with open(filename,'w') as f:
            for i in range(len(SNR_range)):
                #print(str(SNR_range[i]),str(BMI_list[i]),file=f)
                print(str(SNR_range[i]),str(BMI_list_a[i]),file=f)
        

