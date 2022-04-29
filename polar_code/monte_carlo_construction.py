#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pickle
import sys
import multiprocessing
import os
import math


# In[75]:


#polar code
def generate_information(N):
  #generate information
  info=np.random.randint(0,2,N)
  return info

def encode(u_message):
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

    codeword = np.concatenate([encode(u1u2), encode(u2)])
  return codeword

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

def polar_encode(N):
  
  bit_reversal_sequence=reverse_bits(N)
  info=generate_information(N)
  cwd=encode(info[bit_reversal_sequence])
  
  return info,cwd


# In[76]:


#polar code
def chk(llr_1,llr_2):
  CHECK_NODE_TANH_THRES=30
  res=np.zeros(len(llr_1))
  for i in range(len(res)):

    if abs(llr_1[i]) > CHECK_NODE_TANH_THRES and abs(llr_2[i]) > CHECK_NODE_TANH_THRES:
      if llr_1[i] * llr_2[i] > 0:
        # If both LLRs are of one sign, we return the minimum of their absolute values.
        res[i]=min(abs(llr_1[i]), abs(llr_2[i]))
      else:
        # Otherwise, we return an opposite to the minimum of their absolute values.
        res[i]=-1 * min(abs(llr_1[i]), abs(llr_2[i]))
    else:
      res[i]= 2 * np.arctanh(np.tanh(llr_1[i] / 2, ) * np.tanh(llr_2[i] / 2))
  return res

def SC_decoding(N,Lc,info):
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

      llr[depth,length:length+N//(2**depth)]=chk(tmp1,tmp2)

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
      
      #from IPython.core.debugger import Pdb; Pdb().set_trace()
      #print(llr)
      #print(EST_codeword)
      
      if length==N:
        break
  
  #for monte calro construction  
  res=llr[itr_num]
  
  return res             


# In[77]:


#モンテカルロ法を用いるときに使用する関数
def NOMA_mapping(beta,cwd1,cwd2):
  #cwd=np.array([0,0,0,1,1,1,1,0])
  const1=2*cwd1-1 #constellation for UE1
  const2=2*cwd2-1 #constellation for UE2

  #encode(p_all=1)
  const=(1-beta)**(1/2)*const2+(beta)**(1/2)*const1*(-1*const2)
  ## minus 00 -> 10 -> 11 -> 01 plus former_bit:UE1 latter_bit:UE2 
  return const


# In[78]:


#モンテカルロ法を用いるときに使用する関数
def add_AWGN(const,No):
  noise = np.random.normal(0, math.sqrt(No / 2), (len(const))) + 1j * np.random.normal(0, math.sqrt(No / 2), (len(const)))
  return const+noise


# In[79]:


#モンテカルロ法を用いるときに使用する関数
def calc_exp(x,A,No):
  #解が0にならないように計算する
  res=np.zeros(len(x))
  for i in range(len(x)):
    if (x[i]-A)**2/No<30:
      res[i]=np.exp(-1*(x[i]-A)**2/No)
    else:
      res[i]=10**(-15)
  return res


# In[80]:


#モンテカルロ法を用いるときに使用する関数
def calc_LLR(x,No,beta):
  #4PAMのLLRを導出する関数
  M=4 #１つのconstellationに対して、log2(4)=２bitの情報が入っている
  
  A1=calc_exp(x,-(1-beta)**(1/2)-(beta)**(1/2),No)
  A2=calc_exp(x,-(1-beta)**(1/2)+(beta)**(1/2),No)
  A3=calc_exp(x,(1-beta)**(1/2)-(beta)**(1/2),No)
  A4=calc_exp(x,(1-beta)**(1/2)+(beta)**(1/2),No)
  
  Lc2=np.log((A3+A4)/(A1+A2)) #latter bit
  Lc1=np.log((A2+A3)/(A1+A4)) #former bit
  
  #print(Lc)
  #print(y2)
  #print(y1)
  return Lc1,Lc2


# In[81]:


#モンテカルロ法を用いるときに使用する関数
def NOMA_decode(N,info1,info2,beta,const,design_SNR,User):
  
  #マッピングしたものを復号する
  #design_SNRはUE1かUE2の受信dB
  #for UE2
  if User==2:
    #p_all/No2=EsNodB2
    EsNo2 = 10 ** (design_SNR / 10)
    No2=1/EsNo2 #Es=1(fixed) #
    
    res_const=add_AWGN(const,No2)
    res_const=res_const.real
    _,Lc2=calc_LLR(res_const,No2,beta)
    Lc2=-1*Lc2
    llr2=SC_decoding(N,Lc2,info2)
  
    return info2,llr2

  #for UE1
  elif User==1:
    #p_all/No1=EsNodB1
    EsNo1 = 10 ** (design_SNR / 10)
    No1=1/EsNo1 #Es=1
    
    #UE1 constellation
    res_const=add_AWGN(const,No1)
    res_const=res_const.real
    #re_encode
    bit_reversal_sequence=reverse_bits(N)
    EST_cwd2=encode(info2[bit_reversal_sequence])
    EST_const2=2*EST_cwd2-1

    #subtract from res_const
    UE1_const=-1*EST_const2*(res_const-(1-beta)**(1/2)*EST_const2) #ただ引き算をすれば良いわけではないことに注意！！
    #print(const[::20])
    #print(UE1_const[::20])
    #make LLR
    Lc1=-4*UE1_const*beta/No1 #Esがbeta倍になっているので、Noもその分大きくなる #-がつくかどうかわからないので、後で確認
    #use decoder for UE1
    llr1=SC_decoding(N,Lc1,info1)
    
    return info1,llr1
  
  else:
    print("user error!")


# In[82]:


#モンテカルロ法によるLLRの推定
def NOMA(N,beta,design_SNR,User=0):
  #情報ビットと符号語を出力する
  info1,cwd1=polar_encode(N)
  info2,cwd2=polar_encode(N)
  
  #符号語をマッピングする
  const=NOMA_mapping(beta,cwd1,cwd2)
  
  #マッピングした符号語を復号する
  info,llr=NOMA_decode(N,info1,info2,beta,const,design_SNR,User)
  
  return info,llr


# In[86]:


class monte_carlo():
      
  def main_const(self,N,K,beta,design_SNR,User):
      
    #check
    #design_SNR=100
    
    c=self.output(N,beta,design_SNR,User)
    
    tmp=np.argsort(c)[::-1]
    frozen_bits=np.sort(tmp[:N-K])
    info_bits=np.sort(tmp[N-K:])
    
    return frozen_bits,info_bits

  @staticmethod
  def make_d(inputs):
    #unzip inputs
    N,beta,design_SNR,User=inputs
    
    #initial constant
    epoch=10**3
    c=np.zeros(N)
    for _ in range(epoch):
      info,llr=NOMA(N,beta,design_SNR,User)
      #print(llr)
      #from IPython.core.debugger import Pdb; Pdb().set_trace()
      
      d=np.zeros(len(llr))
        #print(llr)
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
      d[(2*info-1)*llr<0]=0
      d[(2*info-1)*llr>=0]=1
      
      c=c+d
    
    return c

  def make_c(self,N,beta,design_SNR,User):
    
    c=np.zeros(N)
    multi_num=100 #the number of multiprocessing 
    
    inputs=[]
    for _ in range(multi_num):
      inputs+=[(N,beta,design_SNR,User)]
    
    #multiprocessing
    with multiprocessing.Pool(processes=multi_num) as pool:
      res = pool.map(self.make_d,inputs)
    
    for i in range(multi_num):
      c=c+res[i]
      
    return c
  
  def output(self,N,beta,design_SNR,User):
    
    # directory make
    current_directory="/home/kaneko/Dropbox/programming/wireless_communication/polar_code"
    #current_directory=os.getcwd()
    dir_name="monte_carlo_construction"
    dir_name=current_directory+"/"+dir_name
    
    try:
      os.makedirs(dir_name)
    except FileExistsError:
      pass
    
    const="NOMA"
      
    filename=const+"_{}_{}_{}_User{}".format(beta,N,design_SNR,User)
    
    #if file exists, then load txt file
    filename=dir_name+"/"+filename
    
    try:
      c=np.loadtxt(filename)
    except FileNotFoundError:
      c=self.make_c(N,beta,design_SNR,User)
      #export file
      np.savetxt(filename,c)

    return c


# In[ ]:


if __name__=="__main__":
  NOMA(1024,0.01,100,1)
  NOMA(1024,0.2,100,2)


  #initial constant
  N=1024
  beta=0.1
  epoch=10**1
  c=np.zeros(N)
  for _ in range(epoch):
    info,llr=NOMA(N,beta,100,2)
    
    d=np.zeros(len(llr))
      #print(llr)
      #from IPython.core.debugger import Pdb; Pdb().set_trace()
    d[(2*info-1)*llr<0]=0
    d[(2*info-1)*llr>=0]=1
    #print(c)
    c=c+d
  
  print(c)

