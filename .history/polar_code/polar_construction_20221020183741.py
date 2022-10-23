import numpy as np
import math
import os
from decimal import *
import sys
import multiprocessing
import pickle
from polar_code import RCA
from polar_code import iGA

class coding():
  def __init__(self,N,K):
    '''
    polar_decode
    Lc: LLR fom channel
    decoder_var:int [0,1,2]
    0:simpified SC decoder
    1:simplified SCL decoder
    2:simplified CA SCL decoder
    '''
    self.N=N
    self.K=K
    self.R=K/N
    self.design_SNR=0
    
    #settings
    self.systematic_polar=False #default:false
    self.decoder_ver=2 #0:SC 1:SCL 2:CA_SCL
    self.bit_reversal_sequence=self.reverse_bits()

    #for encoder (CRC poly)
    #1+x+x^2+....
    self.CRC_polynomial =np.array([1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1])
      
    self.const=iGA.Improved_GA()#monte_carlo() #Improved_GA()
      
    #flozen_bit selection 
    if self.decoder_ver==2:
      CRC_len=len(self.CRC_polynomial)-1
      self.frozen_bits,self.info_bits=self.const.main_const(self.N,self.K+CRC_len,self.design_SNR)
    else:
      self.frozen_bits,self.info_bits=self.const.main_const(self.N,self.K,self.design_SNR)
    
  #reffered to https://dl.acm.org/doi/pdf/10.5555/1074100.1074303
  @staticmethod
  def cyclic(data,polynomial,memory):
    res=np.zeros(len(memory))
    pre_data=(memory[len(memory)-1]+data)%2
    res[0]=pre_data

    for i in range(1,len(polynomial)-1):
      if polynomial[i]==1:
        res[i]=(pre_data+memory[i-1])%2
      else:
        res[i]=memory[i-1]

    return res

  def CRC_gen(self,information,polynomial):
    parity=np.zeros(len(polynomial)-1)
    CRC_info=np.zeros(len(information)+len(parity),dtype='int')
    CRC_info[:len(information)]=information
    CRC_info[len(information):]=parity

    memory=np.zeros(len(polynomial)-1,dtype='int')
    CRC_info[:len(information)]=information
    for i in range(len(information)):
      memory=self.cyclic(information[i],polynomial,memory)
      #print(memory)
    #print(len(memory))
    CRC_info[len(information):]=memory[::-1]
    
    return CRC_info,np.all(memory==0)

  def reverse_bits(self):
    res=np.zeros(self.N,dtype=int)

    for i in range(self.N):
      tmp=format (i,'b')
      tmp=tmp.zfill(int(math.log2(self.N))+1)[:0:-1]
      #print(tmp) 
      res[i]=self.reverse(i)
    return res

  def reverse(self,n):
    tmp=format (n,'b')
    tmp=tmp.zfill(int(math.log2(self.N))+1)[:0:-1]
    res=int(tmp,2) 
    return res
  
if __name__=="__main__":
  N=1024
  for k in range(N):
    myPC=coding(N,k)
    #print(k)