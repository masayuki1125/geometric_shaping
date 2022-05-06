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

# In[78]:

#モンテカルロ法を用いるときに使用する関数
def add_AWGN(const,No):
  noise = np.random.normal(0, math.sqrt(No / 2), (len(const))) + 1j * np.random.normal(0, math.sqrt(No / 2), (len(const)))
  return const+noise

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

def main_func(N):
      

#%%


if __name__=="__main__":

  #initial constant
  N=1024
  beta=0.1
  epoch=10**1
  c=np.zeros(N)
  for _ in range(epoch):
    info,llr=main_func(N,beta,100,2)
    
    d=np.zeros(len(llr))
      #print(llr)
      #from IPython.core.debugger import Pdb; Pdb().set_trace()
    d[(2*info-1)*llr<0]=0
    d[(2*info-1)*llr>=0]=1
    #print(c)
    c=c+d
  
  print(c)

