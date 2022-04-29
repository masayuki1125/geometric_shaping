#!/usr/bin/env python
# coding: utf-8

# In[8]:


import math
import numpy as np
import cupy as cp
from scipy.stats import norm


# In[9]:


def add_AWGN_GPU(constellation,No):
  # AWGN雑音の生成
  noise = cp.random.normal(0, math.sqrt(No / 2), (len(constellation)))           + 1j * cp.random.normal(0, math.sqrt(No / 2), (len(constellation)))

  # AWGN通信路 = 送信シンボル間干渉が生じないような通信路で送信
  RX_constellation = constellation + noise

  # 以下のprint関数の出力を表示すると、Noとほぼ一致するはず
  #print(cp.dot(noise[0, :], cp.conj(noise[0, :]))/bit_num)

  return RX_constellation


# In[10]:


def add_Rayleigh_GPU(constellation,No,beta=1):
  noise = cp.random.normal(0, math.sqrt(No / 2), (len(constellation)))           + 1j * cp.random.normal(0, math.sqrt(No / 2), (len(constellation)))
  
  interference=cp.random.randint(0,2,len(constellation))
          
  # AWGN通信路 = 送信シンボル間干渉が生じないような通信路で送信
  RX_constellation = constellation + noise + (beta)**(1/2)*interference

  # 以下のprint関数の出力を表示すると、Noとほぼ一致するはず
  #print(cp.dot(noise[0, :], cp.conj(noise[0, :]))/bit_num)

  return RX_constellation


# In[11]:


def mutual_info(EsNodB,beta,User):
  
  p_all=1

  EsNo = 10 ** (EsNodB / 10)
  No=1/EsNo
  count_num=100000000
  M=2

  #make constellation
  info=cp.random.randint(0,2,count_num)
  if User==1:
    const=(beta*p_all)**(1/2)*(2*info-1)
    A=(beta*p_all)**(1/2) #small const
    symbol=cp.array([A,-A])
  elif User==2:
    const=((1-beta)*p_all)**(1/2)*(2*info-1)
    A=((1-beta)*p_all)**(1/2) #small const
    symbol=cp.array([A,-A])
  else:
    print("err")
    

  #if cp.any(symbol==const)!=True:
    #print("error")
    #print(symbol)
    #print(const)

  #make mutual entrophy
  if User==1:
    RX_const=add_AWGN_GPU(const,No)
  elif User==2:
    RX_const=add_Rayleigh_GPU(const,No,beta)
  else:
    print("err")
    
  num=cp.sum(cp.exp(-1*cp.abs(np.tile(RX_const,(len(symbol),1))-symbol.reshape(-1,1))**2/No),axis=0)
  
  den=cp.exp(-1*cp.abs(RX_const-const)**2/No)
  H=cp.sum(cp.log2(num/den))
  H/=count_num
  res=math.log2(M)-H
  return res


# In[14]:


def mutual_info_PAM(EsNodB,beta):
  p_all=1

  EsNo = 10 ** (EsNodB / 10)
  No=1/EsNo
  count_num=100000000
  M=4
  
  A=(beta*p_all)**(1/2) #small const
  B=((1-beta)*p_all)**(1/2) #big const
  symbol=cp.array([-A-B,A-B,-A+B,A+B])
  
  #make constellation
  info1=cp.random.randint(0,2,count_num)
  const1=(beta*p_all)**(1/2)*(2*info1-1)

  info2=cp.random.randint(0,2,count_num)
  const2=((1-beta)*p_all)**(1/2)*(2*info2-1)
  const=const1+const2

  #make mutual entrophy
  RX_const=add_AWGN_GPU(const,No)
  num=cp.sum(cp.exp(-1*cp.abs(np.tile(RX_const,(len(symbol),1))-symbol.reshape(-1,1))**2/No),axis=0)
  den=cp.exp(-1*cp.abs(RX_const-const)**2/No)
  H=cp.sum(cp.log2(num/den))
  H/=count_num
  res=math.log2(M)-H
  return res


# In[15]:


def finite_bound(n,epsilon,P,C_pre):
  C=C_pre/2
    
  V=P/2*(P+2)/(P+1)**2*(np.log(np.exp(1))**2)
  Q_inv=-1*norm.ppf(epsilon)
  logM=n*C-(n*V)**(1/2)*Q_inv+1/2*np.log(n)
  R=2*logM/n
  
  if -(n*V)**(1/2)*Q_inv+1/2*np.log(n)>0:
    print("penalty error")
    print(P)
    print(V)
    print(Q_inv)
  
  #print(R)
  if R>C_pre:
    print("R big error!")
  
  return R


# In[16]:


def calc_capacity(EsNodB1,N):
  
  #set constant
  itr_num=1000

  #N=512
  #print("N=",N)
  #EsNodB1
  EsNodB2=0
  EsNo1=10 ** (EsNodB1 / 10)
  EsNo2=10 ** (EsNodB2 / 10)
  target_BLER=10**-3

  #NOMA
  C1_infinite_NOMA=cp.zeros(itr_num)
  C2_infinite_NOMA=cp.zeros(itr_num)

  C1_NOMA=cp.zeros(itr_num)
  C2_NOMA=cp.zeros(itr_num)

  #OMA
  C1_infinite_OMA=cp.zeros(itr_num)
  C2_infinite_OMA=cp.zeros(itr_num)

  C1_OMA=cp.zeros(itr_num)
  C2_OMA=cp.zeros(itr_num)

  #PAM
  C1_infinite_PAM=cp.zeros(itr_num)
  C2_infinite_PAM=cp.zeros(itr_num)

  C1_PAM=cp.zeros(itr_num)
  C2_PAM=cp.zeros(itr_num)


  for i in range(itr_num):
    beta=i/itr_num

    #NOMA
    C1_infinite_NOMA[i]=mutual_info(EsNodB1,beta)
    C2_infinite_NOMA[i]=mutual_info(EsNodB2,beta,True)
    C1_NOMA[i]=finite_bound(N,target_BLER,EsNo1,C1_infinite_NOMA[i])
    C2_NOMA[i]=finite_bound(N,target_BLER,EsNo2,C2_infinite_NOMA[i])
    
    #OMA
    C1_infinite_OMA[i]=beta*mutual_info(EsNodB1,1)
    C2_infinite_OMA[i]=(1-beta)*mutual_info(EsNodB2,1)
    C1_OMA[i]=finite_bound(N,target_BLER,EsNo1,C1_infinite_OMA[i])
    C2_OMA[i]=finite_bound(N,target_BLER,EsNo2,C2_infinite_OMA[i])
    
    #PAM
    C1_infinite_PAM=C1_infinite_NOMA
    C1_PAM=C1_NOMA
    C2_infinite_PAM[i]=mutual_info_PAM(EsNodB2,beta)
    C2_PAM[i]=finite_bound(N*2,target_BLER,EsNo2,C2_infinite_PAM[i])
    C2_infinite_PAM[i]-=C1_infinite_PAM[i]
    C2_PAM[i]-=C1_PAM[i]
    
    
    print("\r"+"C1_inf="+str(C1_infinite_NOMA[i])+"C2_inf="+str(C2_infinite_NOMA[i])+"C1="+str(C1_NOMA[i])+"C2="+str(C2_NOMA[i]),end="")

  #NOMA
  filename="Capacity_inf_NOMA_{}_{}_{}".format(EsNodB1,EsNodB2,N)
  with open(filename,'w') as f:

    print("#C1_inf,C2_inf",file=f)  
    for i in range(itr_num):
      print(str(C1_infinite_NOMA[i]),str(C2_infinite_NOMA[i]),file=f)

  filename="Capacity_NOMA_{}_{}_{}".format(EsNodB1,EsNodB2,N)
  with open(filename,'w') as f:

    print("#C1,C2",file=f)  
    for i in range(itr_num):
      print(str(C1_NOMA[i]),str(C2_NOMA[i]),file=f)
  
  #OMA   
  filename="Capacity_inf_OMA_{}_{}_{}".format(EsNodB1,EsNodB2,N)
  with open(filename,'w') as f:

    print("#C1_inf,C2_inf",file=f)  
    for i in range(itr_num):
      print(str(C1_infinite_OMA[i]),str(C2_infinite_OMA[i]),file=f)

  filename="Capacity_OMA_{}_{}_{}".format(EsNodB1,EsNodB2,N)
  with open(filename,'w') as f:

    print("#C1,C2",file=f)  
    for i in range(itr_num):
      print(str(C1_OMA[i]),str(C2_OMA[i]),file=f)

  #PAM  
  filename="Capacity_inf_PAM_{}_{}_{}".format(EsNodB1,EsNodB2,N*2)
  with open(filename,'w') as f:

    print("#C1_inf,C2_inf",file=f)  
    for i in range(itr_num):
      print(str(C1_infinite_PAM[i]),str(C2_infinite_PAM[i]),file=f)

  filename="Capacity_PAM_{}_{}_{}".format(EsNodB1,EsNodB2,N*2)
  with open(filename,'w') as f:

    print("#C1,C2",file=f)  
    for i in range(itr_num):
      print(str(C1_PAM[i]),str(C2_PAM[i]),file=f)

if __name__=="__main__":   
  for i in [256,512]:
    for j in [3,5,10]:
      calc_capacity(j,i)

