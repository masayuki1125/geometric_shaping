#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import math
from decimal import *
import sys
import os
sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
from channel.AWGN import _AWGN
from modulation.modulation import QAMModem
from polar_construction import Improved_GA
from polar_construction import GA
from polar_construction import monte_carlo

# In[2]:



class coding():
  def __init__(self,N,K,beta=1,M=2,Rayleigh=False):
    super().__init__()
    '''
    polar_decode
    Lc: LLR fom channel
    decoder_var:int [0,1,2]
    0:simpified SC decoder
    1:simplified SCL decoder
    2:simplified CA SCL decoder
    '''
    self.N=N
    #self.K=math.floor(self.R*self.N)
    self.K=K #K=0 :monte_carlo_construction
    self.R=K/N
    self.design_SNR=0
    self.Rayleigh=Rayleigh
    
    #decide channel coding variance
    self.ch=_AWGN(beta,M)
    
    if self.K!=0:
      self.const=Improved_GA() #Improved_GA,GA
      self.decoder_var=2 #0:SC 1:SCL 2:SCL_CRC
    if self.K==0: #for monte_carlo_construction
      self.decoder_var=0
      self.const=GA()
    
    self.adaptive_design_SNR=False #default:False
    self.systematic_polar=False #default:false

    #for SCL decoder
    self.list_size=4

    #prepere constants
    self.itr_num=int(math.log2(self.N))
    self.bit_reversal_sequence=self.reverse_bits()

    #for encoder (CRC poly)
    #1+x+x^2+....
    self.CRC_polynomial =np.array([1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1])
    self.CRC_len=len(self.CRC_polynomial)
    
    #flozen_bit selection 
    self.frozen_bits,self.info_bits=self.choose_frozen_bits(self.design_SNR)  
    
    if self.systematic_polar==True:
      self.filename="systematic_"+self.filename
      
      


# In[3]:


class coding(coding):
  def choose_frozen_bits(self,design_SNR,beta=1000):
    if self.decoder_var==0:
      self.filename="polar_SC_{}_{}".format(self.N,self.K)
      #construction
      frozen_bits,info_bits=self.const.main_const(self.N,self.K,design_SNR,beta)
    
    elif self.decoder_var==1:
      self.filename="polar_SCL_{}_{}_{}".format(self.N,self.K,self.list_size)
      #construction
      frozen_bits,info_bits=self.const.main_const(self.N,self.K,design_SNR,beta)
        
    elif self.decoder_var==2:
      self.filename="polar_SCL_CRC_{}_{}_{}".format(self.N,self.K,self.list_size)
      #construction
      frozen_bits,info_bits=self.const.main_const(self.N,self.K+self.CRC_len-1,design_SNR,beta)
      
    return frozen_bits,info_bits
    


# In[4]:


class coding(coding):
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

# In[5]:

class coding(coding):
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


# In[6]:


class coding(coding):
  def reverse_bits(self):
    res=np.zeros(self.N,dtype=int)

    for i in range(self.N):
      tmp=format (i,'b')
      tmp=tmp.zfill(self.itr_num+1)[:0:-1]
      #print(tmp) 
      res[i]=self.reverse(i)
    return res

  def reverse(self,n):
    tmp=format (n,'b')
    tmp=tmp.zfill(self.itr_num+1)[:0:-1]
    res=int(tmp,2) 
    return res


# In[7]:


class encoding(coding):
  def __init__(self,N,K,beta=1,M=2,Rayleigh=False):
    super().__init__(N,K,beta,M,Rayleigh)
  
  def generate_information(self):
    #generate information
    
    if self.K!=0:
      information=np.random.randint(0,2,self.K)
    
    else:
      information=np.zeros(self.K)

    if self.decoder_var==0 or self.decoder_var==1:
      return information
    
    elif self.decoder_var==2:
      parity=np.zeros(len(self.CRC_polynomial)-1)
      CRC_info,_=self.CRC_gen(information,self.CRC_polynomial)

      ##check CRC_info
      _,check=self.CRC_gen(CRC_info,self.CRC_polynomial)
      if check!=True:
        print("CRC_info error")
      
      return CRC_info
      

  def generate_U(self,information):
    u_message=np.zeros(self.N)
    u_message[self.info_bits]=information
    return u_message

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


# In[8]:


#print(ec.frozen_bits)
#print(ec.info_bits)
class encoding(encoding):
  def systematic_encode(self,information):
    X=np.zeros((self.N,self.itr_num+1))
    X[self.frozen_bits,0]=0
    X[self.info_bits,self.itr_num]=information

    for i in reversed(range(1,self.N+1)):
      if np.any(i-1==self.info_bits):
        s=self.itr_num+1
        delta=-1
      else:
        s=1
        delta=1

      #binary representation
      tmp=format (i-1,'b')
      b=tmp.zfill(self.itr_num+1)
        
      for j in range(1,self.itr_num+1):
        t=s+delta*j
        l=min(t,t-delta)
        kai=2**(self.itr_num-l)
        #print(l)
        if int(b[l])==0:
          #print("kai")
          #print(i-kai-1)
          #print(i-1)
          X[i-1,t-1]=(X[i-1,t-delta-1]+X[i+kai-1,t-delta-1])%2
        
        else:
          #print("b")
          #print("kai")
          #print(i-kai)
          X[i-1,t-1]=X[i-1,t-delta-1]
        
    #print(X)

    #check
    x=X[:,self.itr_num]
    y=X[:,0]
    codeword=self.encode(y[self.bit_reversal_sequence])
    if np.any(codeword!=x):
      print(codeword)
      print("err")
    
    return x
      


# In[9]:


class encoding(encoding):
  def polar_encode(self):
    information=self.generate_information()
    
    if self.systematic_polar==False:
      u_message=self.generate_U(information)
      codeword=self.encode(u_message[self.bit_reversal_sequence])
    
    elif self.systematic_polar==True:
      codeword=self.systematic_encode(information)
    
    tmp=np.zeros(len(codeword))
    tmp=codeword
    #codeword=u_message@self.Gres%2
    return information,tmp


# In[10]:


class decoding(coding):

  def __init__(self,N,K,beta=1,M=2,Rayleigh=False):
    super().__init__(N,K,beta,M,Rayleigh)
    
  @staticmethod
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


# In[11]:


class decoding(decoding):
  def SC_decoding(self,Lc):
    #initialize constant    
    llr=np.zeros((self.itr_num+1,self.N))
    EST_codeword=np.zeros((self.itr_num+1,self.N))
    llr[0]=Lc

    #put decoding result into llr[logN]

    depth=0
    length=0
    before_process=0# 0:left 1:right 2:up 3:leaf

    while True:
  
      #left node operation
      if before_process!=2 and before_process!=3 and length%2**(self.itr_num-depth)==0:
        depth+=1
        before_process=0

        tmp1=llr[depth-1,length:length+2**(self.itr_num-depth)]
        tmp2=llr[depth-1,length+2**(self.itr_num-depth):length+2**(self.itr_num-depth+1)]

        llr[depth,length:length+self.N//(2**depth)]=self.chk(tmp1,tmp2)

      #right node operation 
      elif before_process!=1 and length%2**(self.itr_num-depth)==2**(self.itr_num-depth-1):
        
        #print(length%2**(self.itr_num-depth))
        #print(2**(self.itr_num-depth-1))
        
        depth+=1
        before_process=1

        tmp1=llr[depth-1,length-2**(self.itr_num-depth):length]
        tmp2=llr[depth-1,length:length+2**(self.itr_num-depth)]

        llr[depth,length:length+2**(self.itr_num-depth)]=tmp2+(1-2*EST_codeword[depth,length-2**(self.itr_num-depth):length])*tmp1

      #up node operation
      elif before_process!=0 and length!=0 and length%2**(self.itr_num-depth)==0:#今いるdepthより一個下のノードから、upすべきか判断する
      
        tmp1=EST_codeword[depth+1,length-2**(self.itr_num-depth):length-2**(self.itr_num-depth-1)]
        tmp2=EST_codeword[depth+1,length-2**(self.itr_num-depth-1):length]

        EST_codeword[depth,length-2**(self.itr_num-depth):length-2**(self.itr_num-depth-1)]=(tmp1+tmp2)%2
        EST_codeword[depth,length-2**(self.itr_num-depth-1):length]=tmp2

        depth-=1
        before_process=2
      
      else:
        print("error!")

      #leaf node operation
      if depth==self.itr_num:
        
        #frozen_bit or not
        if np.any(self.frozen_bits==length):
          EST_codeword[depth,length]=0
        
        #info_bit operation
        else :
          EST_codeword[depth,length]=(-1*np.sign(llr[depth,length])+1)//2
        
        length+=1 #go to next length

        depth-=1 #back to depth
        before_process=3
        
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        #print(llr)
        #print(EST_codeword)
        
        if length==self.N:
          break
    
    if self.K!=0: 
      res=EST_codeword[self.itr_num]
    else:
      res=llr[self.itr_num]
      #np.savetxt("llr",res)
      #from IPython.core.debugger import Pdb; Pdb().set_trace()
      
    if self.systematic_polar==True:
      #re encode polar
      u_message=self.generate_U(res[self.info_bits])
      res=self.encode(u_message[self.bit_reversal_sequence])
    
    return res


# In[12]:


class decoding(decoding):
  
  @staticmethod
  def calc_BM(u_tilde,llr):
    if u_tilde*llr>30:
      return u_tilde*llr
    else:
      return math.log(1+math.exp(u_tilde*llr))

  def SCL_decoding(self,Lc):

    #initialize constant    
    llr=np.zeros((self.list_size,self.itr_num+1,self.N))
    EST_codeword=np.zeros((self.list_size,self.itr_num+1,self.N))
    llr[0,0]=Lc
    PML=np.full(self.list_size,10.0**10) #path metric of each list 
    PML[0]=0 

    #put decoding result into llr[L,logN]

    #prepere constant
    depth=0
    length=0
    before_process=0# 0:left 1:right 2:up 3:leaf
    branch=1#the number of branchs. 1 firstly, and increase up to list size 
    BM=np.full((self.list_size,2),10.0**10)#branch metrics
    # low BM is better

    while True:
      
      #interior node operation
  
      #left node operation
      if before_process!=2 and before_process!=3 and length%2**(self.itr_num-depth)==0:
        depth+=1
        before_process=0

        tmp1=llr[:,depth-1,length:length+2**(self.itr_num-depth)]
        tmp2=llr[:,depth-1,length+2**(self.itr_num-depth):length+2**(self.itr_num-depth+1)]

        #carculate each list index
        for i in range(branch):
          llr[i,depth,length:length+self.N//(2**depth)]=self.chk(tmp1[i],tmp2[i])

      #right node operation 
      elif before_process!=1 and length%2**(self.itr_num-depth)==2**(self.itr_num-depth-1):
        
        depth+=1
        before_process=1

        tmp1=llr[:,depth-1,length-2**(self.itr_num-depth):length]
        tmp2=llr[:,depth-1,length:length+2**(self.itr_num-depth)]

        #carculate each list index
        for i in range(branch):
          llr[i,depth,length:length+2**(self.itr_num-depth)]=tmp2[i]+(1-2*EST_codeword[i,depth,length-2**(self.itr_num-depth):length])*tmp1[i]

      #up node operation
      elif before_process!=0 and length!=0 and length%2**(self.itr_num-depth)==0:#今いるdepthより一個下のノードから、upすべきか判断する
      
        tmp1=EST_codeword[:,depth+1,length-2**(self.itr_num-depth):length-2**(self.itr_num-depth-1)]
        tmp2=EST_codeword[:,depth+1,length-2**(self.itr_num-depth-1):length]

        #carculate each list index
        for i in range(branch):
          EST_codeword[i,depth,length-2**(self.itr_num-depth):length-2**(self.itr_num-depth-1)]=(tmp1[i]+tmp2[i])%2
          EST_codeword[i,depth,length-2**(self.itr_num-depth-1):length]=tmp2[i]

        depth-=1
        before_process=2

      #leaf node operation
      if depth==self.itr_num:
        
        #frozen_bit or not
        if np.any(self.frozen_bits==length):

          #decide each list index
          for i in range(branch):
            EST_codeword[i,depth,length]=0
          
          #update path metric
          u_tilde=-1#because frozen_bit is 0
          for i in range(branch):
            PML[i]=PML[i]+self.calc_BM(u_tilde,llr[i,depth,length])
        
        #info_bit operation
        else :

          #decide each list index
          for i in range(branch):

            u_tilde=-1*np.sign(llr[i,depth,length])#[-1,1]
            #llr<0 -> u_tilde=1 u_hat=1 // u_hat(sub)=0
            #llr>0 -> u_tilde=-1 u_hat=0 // u_hat(sub)=1
            
            #decide main u_hat 
            tmp0=self.calc_BM(u_tilde,llr[i,depth,length])
            tmp1=self.calc_BM(-1*u_tilde,llr[i,depth,length])
            BM[i,int((u_tilde+1)//2)]=tmp0

            #decide sub u_hat
            BM[i,int((-1*u_tilde+1)//2)]=tmp1

          #branch*2 path number
          #update PM

          #update BM to PML 2d array
          BM[0:branch]+=np.tile(PML[0:branch,None],(1,2))

          #update branch
          branch=branch*2
          if branch>self.list_size:
            branch = self.list_size 

          #trim PML 2d array and update PML
          PML[0:branch]=np.sort(np.ravel(BM))[0:branch]
          list_num=np.argsort((np.ravel(BM)))[0:branch]//2#i番目のPMを持つノードが何番目のリストから来たのか特定する
          u_hat=np.argsort((np.ravel(BM)))[0:branch]%2##i番目のPMを持つノードがu_hatが0か1か特定する
          
          #copy before data
          #選ばれたパスの中で、何番目のリストが何番目のリストからの派生なのかを計算する
          #その後、llr，EST_codewordの値をコピーし、今計算しているリーフノードの値も代入する
          llr[0:branch]=llr[list_num,:,:]#listを並び替えru
          EST_codeword[0:branch]=EST_codeword[list_num,:,:]
          EST_codeword[0:branch,depth,length]=u_hat
                  
        length+=1 #go to next length

        depth-=1 #back to depth
        before_process=3
        
        if length==self.N:
          break
    
    res_list_num=0
    res=np.zeros((self.list_size,self.N))
    #print(res.shape)
    #print(res[i].shape)
         
      #set candidates
    for i in range(self.list_size):
      res[i,:]=EST_codeword[i,self.itr_num]
        
    #for systematic_polar
    if self.systematic_polar==True:
      #re encode polar
      for i in range(self.list_size):
        #print(i)
        #print(res[i][self.info_bits])
        u_message=self.generate_U(res[i,:][self.info_bits])
        res[i,:]=self.encode(u_message[self.bit_reversal_sequence])
    
 
    
    #CRC_check
    if self.decoder_var==2:
      for i in range(self.list_size):
        EST_CRC_info=res[i][self.info_bits]
        _,check=self.CRC_gen(EST_CRC_info,self.CRC_polynomial)
        #print(check)
        if check==True:
          res_list_num=i
          break
      
      #else:
        #print("no codeword")
    #print("CRC_err")

    #print("\r",PML,end="")
    
    return res[res_list_num]


# In[13]:


class decoding(decoding):
  def polar_decode(self,Lc):
    '''
    polar_decode
    Lc: LLR fom channel
    decoder_var:int [0,1] (default:0)
    0:simpified SC decoder
    1:simplified SCL decoder
    '''
    #initialize 

    if self.decoder_var==0:
      EST_codeword=self.SC_decoding(Lc)

    elif self.decoder_var==1 or self.decoder_var==2:
      EST_codeword=self.SCL_decoding(Lc)
      
    if self.K==0:
      res=EST_codeword
    else:
      res=EST_codeword[self.info_bits]
    return res


# In[14]:


class polar_code(encoding,decoding):
  def __init__(self,N,K,beta=1,M=2,Rayleigh=False):
    super().__init__(N,K,beta,M,Rayleigh)

  def main_func(self,EbNodB): 
    EbNo = 10 ** (EbNodB / 10)
    No=1/EbNo
    if self.adaptive_design_SNR==True:
      if self.design_SNR!=EbNodB:
        self.design_SNR=EbNodB
        self.frozen_bits,self.info_bits=self.choose_frozen_bits(self.design_SNR)
    modem=QAMModem(4)
    ch=_AWGN()
    information,codeword=self.polar_encode()
    codeword=codeword.astype(np.int)
    #codeword=np.random.randint(0,2,len(codeword))
    TX_const=modem.modulate(codeword)
    RX_const=ch.add_AWGN(TX_const,No)
    #print(RX_const)
    Lc=modem.demodulate(RX_const,No)
    EST_information=self.polar_decode(Lc) 
      
    if len(EST_information)!=len(information):
      print("len_err")

    return information,EST_information

# In[15]:

if __name__=="__main__":

    N=1024
    K=512
    M=4
    Rayleigh=False
    pc=polar_code(N,K,0.2,M,Rayleigh)
    #a,b=pc.main_func(1)
    #print(len(a))
    #print(len(b))
    def output(EbNodB):
      count_err=0
      count_all=0
      count_berr=0
      count_ball=0
      MAX_ERR=20

      while count_err<MAX_ERR:
        
        #pc=polar_code(N,K)
        information,EST_information=pc.main_func(EbNodB)
      
        if np.any(information!=EST_information):#BLOCK error check
          count_err+=1
          #from IPython.core.debugger import Pdb; Pdb().set_trace()
          
        
        count_all+=1

        #calculate bit error rate 
        count_berr+=np.sum(information!=EST_information)
        count_ball+=N

        print("\r","count_all=",count_all,",count_err=",count_err,"count_ball="              ,count_ball,"count_berr=",count_berr,end="")

      print("BER=",count_berr/count_ball)
      
      return  count_err,count_all,count_berr,count_all
    
    output(3)
    

