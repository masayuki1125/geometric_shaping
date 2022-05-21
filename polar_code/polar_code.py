import numpy as np
import math
from decimal import *
from AWGN import _AWGN

ch=_AWGN()


# In[362]:


class coding():
  def __init__(self,N):
    super().__init__()
    '''
    polar_decode
    Lc: LLR fom channel
    decoder_var:int [0,1,2]
    0:simpified SC decoder
    1:simplified SCL decoder
    2:simplified CA SCL decoder
    '''
    self.decoder_var=1 #0:SC 1:SCL_CRC
    self.N=N
    self.R=0.5
    self.K=math.floor(self.R*self.N)
    self.design_SNR=1

    #for SCL decoder
    self.list_size=4

    #prepere constants
    self.itr_num=np.log2(self.N).astype(int)
    self.bit_reversal_sequence=self.reverse_bits()

    #for construction(GA)
    self.G_0=0.2
    self.G_1=0.7
    self.G_2=10
    self.a0=-0.002706
    self.a1=-0.476711
    self.a2=0.0512
    self.a=-0.4527
    self.b=0.0218
    self.c=0.86
    self.K_0=8.554

    self.Z_0=self.xi(self.G_0)
    self.Z_1=self.xi(self.G_1)
    self.Z_2=self.xi(self.G_2)

    #for encoder (CRC poly)
    #x^15+x^14+...+x+1
    self.CRC_polynomial =np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    self.CRC_len=len(self.CRC_polynomial)

    if self.decoder_var==0:
      self.filename="polar_SC_{}_{}".format(self.N,self.K)
      #construction
      self.frozen_bits,self.info_bits=self.Improved_GA(self.K)
    
    elif self.decoder_var==1:
      self.filename="polar_SCL_{}_{}".format(self.N,self.K)
      #construction
      self.frozen_bits,self.info_bits=self.Improved_GA(self.K)
      
    
    elif self.decoder_var==2:
      self.filename="polar_SCL_CRC_{}_{}".format(self.N,self.K)
      #construction
      self.frozen_bits,self.info_bits=self.Improved_GA(self.K+self.CRC_len-1)
      


# In[363]:


class coding(coding):
  @staticmethod
  def cyclic(data,polynomial,memory):
    tmp=-1*np.ones(len(memory)+1,dtype=int)
    tmp[len(tmp)-1]=data
    pre_data=memory[0]
    tmp[:len(tmp)-1]=memory

    for i in range(len(polynomial)-1):
      if polynomial[i]==1:
        tmp[i]=(pre_data+tmp[i+1])%2
      else:
        tmp[i]=tmp[i+1]

    #from IPython.core.debugger import Pdb; Pdb().set_trace()
    memory=tmp[:len(memory)]

    return memory


# In[364]:


class coding(coding):
  def CRC_gen(self,information,parity,polynomial):
    CRC_info=np.zeros(len(information)+len(parity),dtype='int')
    CRC_info[:len(information)]=information
    CRC_info[len(information):]=parity

    memory=np.zeros(len(polynomial)-1,dtype='int')
    CRC_info[:len(information)]=information
    for i in range(len(CRC_info)):
      memory=self.cyclic(CRC_info[i],polynomial,memory)
    #print(len(memory))
    CRC_info[len(information):]=memory
    
    return CRC_info,np.all(memory==0)


# In[365]:


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


# In[366]:


class coding(coding):
   
  def xi(self,gamma):

    if gamma<=self.G_0:
      zeta=-1*gamma/2+(gamma**2)/8-(gamma**3)/8
    
    elif self.G_0<gamma<=self.G_1:
      zeta=self.a0+self.a1*gamma+self.a2*(gamma**2)

    elif self.G_1<gamma<self.G_2:
      zeta=self.a*(gamma**self.c)+self.b

    elif self.G_2<=gamma:
      zeta=-1*gamma/4+math.log(math.pi)/2-math.log(gamma)/2+math.log(1-(math.pi**2)/(4*gamma)+self.K_0/(gamma**2))
    
    if zeta>0:
      print("zeta is + err")

    return zeta

  def xi_inv(self,zeta):

    if self.Z_0<=zeta:
      gamma=-2*zeta+zeta**2+zeta**3

    elif self.Z_1<=zeta<self.Z_0:
      gamma=(-1*self.a1-(self.a1**2-4*self.a2*(self.a0-zeta))**(1/2))/(2*self.a2)

    elif self.Z_2<zeta<self.Z_1:
      gamma=((zeta-self.b)/self.a)**(1/self.c)

    elif zeta<=self.Z_2:
      gamma=self.bisection_method(zeta)
      #gamma=-4*zeta

    if gamma<0:
      print("gamma is - err")

    return gamma


# In[367]:


class coding(coding):
  def bisection_method(self,zeta):

    #set constant
    min_num=self.G_2
    max_num=10**7
    error_accept=1

    def f(x):
      zeta=-1*x/4+math.log(math.pi)/2-math.log(x)/2+math.log(1-(math.pi**2)/(4*x)+self.K_0/(x**2))
      return zeta

    #initial value
    a=min_num
    b=max_num
    error=b-a

    #very small zeta situation
    if f(max_num)>zeta:
      print(f(max_num))
      print(zeta)
      gamma=max_num

    else:

      while error>error_accept:
        c=(b+a)/2 #center value

        if f(c)>=zeta:
          a=c
          error=b-a
        
        elif f(c)<zeta:
          b=c
          error=b-a
        
        if error<0:
          print("something is wrong")
        #print("\r",error,end="")
      
      gamma=(b+a)/2

      if gamma<0:
        print("gamma is - err")

    return gamma


# In[368]:


class coding(coding):
    
  def Improved_GA(self,K,bit_reverse=True):
    gamma=np.zeros(self.N)
    
    gamma[0]=4*self.design_SNR
    for i in range(1,self.itr_num+1):
      J=2**(i-1)
      for j in range(0,J):
        u=gamma[j]
        if u<=self.G_0:
          gamma[j]=(u**2)/2-(u**3)/2+2*(u**4)/3
        else:
          z=self.xi(u)
          gamma[j]=self.xi_inv(z+math.log(2-math.e**z))
        
        gamma[j+J]=2*u
  
    tmp=self.indices_of_elements(gamma,self.N)
    frozen_bits=np.sort(tmp[:self.N-K])
    info_bits=np.sort(tmp[self.N-K:])

    if bit_reverse==True:
      for i in range(len(frozen_bits)):
        frozen_bits[i]=self.reverse(frozen_bits[i])
      frozen_bits=np.sort(frozen_bits)

      for i in range(len(info_bits)):
        info_bits[i]=self.reverse(info_bits[i])
      info_bits=np.sort(info_bits)

    return frozen_bits,info_bits

  @staticmethod
  def indices_of_elements(v,l):
    tmp=np.argsort(v)
    res=tmp[0:l]
    return res


# In[369]:


class encoding(coding):
  def __init__(self,N):
    super().__init__(N)
  
  def generate_information(self):
    #generate information
    information=np.random.randint(0,2,self.K)
    #information=np.zeros(self.K)

    if self.decoder_var==0 or self.decoder_var==1:
      return information
    
    elif self.decoder_var==2:
      parity=np.zeros(len(self.CRC_polynomial)-1)
      CRC_info,_=self.CRC_gen(information,parity,self.CRC_polynomial)

      ##check CRC_info
      parity=CRC_info[len(information):]
      information=CRC_info[:len(information)]
      _,check=self.CRC_gen(information,parity,self.CRC_polynomial)
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

  def polar_encode(self):
    information=self.generate_information()
    u_message=self.generate_U(information)
    codeword=self.encode(u_message[self.bit_reversal_sequence])
    #codeword=u_message@self.Gres%2
    return information,codeword


# In[370]:


class decoding(coding):

  def __init__(self,N):
    super().__init__(N)
   
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


# In[371]:


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
      
      #interior node operation
      #up node operation
  
      #left node operation
      if before_process!=2 and before_process!=3 and length%2**(self.itr_num-depth)==0:
        depth+=1
        before_process=0

        tmp1=llr[depth-1,length:length+2**(self.itr_num-depth)]
        tmp2=llr[depth-1,length+2**(self.itr_num-depth):length+2**(self.itr_num-depth+1)]

        llr[depth,length:length+self.N//(2**depth)]=self.chk(tmp1,tmp2)

      #right node operation 
      elif before_process!=1 and length%2**(self.itr_num-depth)==2**(self.itr_num-depth-1):
        
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
        
        if length==self.N:
          break
    
    return EST_codeword[self.itr_num]


# In[372]:


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
      
      else:
        print("error!")

      #leaf node operation
      if depth==self.itr_num:
        
        #frozen_bit or not
        if np.any(self.frozen_bits==length):

          #decide each list index
          for i in range(branch):
            EST_codeword[i,depth,length]=0
        
        #info_bit operation
        else :

          #decide each list index
          for i in range(branch):

            u_tilde=-1*np.sign(llr[i,depth,length])#[-1,1]

            #decide main u_hat. PM
            BM[i,int((u_tilde+1)//2)]=self.calc_BM(u_tilde,llr[i,depth,length])

            #decide sub u_hat. PM
            BM[i,int((-1*u_tilde+1)//2)]=self.calc_BM(-1*u_tilde,llr[i,depth,length])

          #branch*2 path number
          #update PM
          
          #print("before PML")
          #print(PML)
          #print(np.tile(PML[0:branch,None],(1,2)))
          #print("before BM")
          #print(BM)

          #update BM to PML 2d array
          BM[0:branch]+=np.tile(PML[0:branch,None],(1,2))

          #update branch
          branch=branch*2
          if branch>self.list_size:
            branch = self.list_size 

          #trim PML 2d array and update PML
          PML[0:branch]=np.sort(np.ravel(BM))[0:branch]


          #print("updated BM")
          #print(BM)
          #print("updated PML")
          #print(PML)
          #from IPython.core.debugger import Pdb; Pdb().set_trace()
          
          #copy before data
          #選ばれたパスの中で、何番目のリストが何番目のリストからの派生なのかを計算する
          #その後、llr，EST_codewordの値をコピーし、今計算しているリーフノードの値も代入する
          list_num=np.argsort((np.ravel(BM)))[0:branch]//2#i番目のPMを持つノードが何番目のリストから来たのか特定する
          u_hat=np.argsort((np.ravel(BM)))[0:branch]%2##i番目のPMを持つノードがu_tildeが0か1か特定する
          llr[0:branch]=llr[list_num,:,:]#listを並び替えru
          EST_codeword[0:branch]=EST_codeword[list_num,:,:]
          EST_codeword[0:branch,depth,length]=u_hat
                  
        length+=1 #go to next length

        depth-=1 #back to depth
        before_process=3
        
        if length==self.N:
          break
    
    
    res_list_num=0
    
    #CRC_check
    if self.decoder_var==2:
      for i in range(self.list_size):
        EST_CRC_info=EST_codeword[i,self.itr_num][self.info_bits]
        _,check=self.CRC_gen(EST_CRC_info[:self.K],EST_CRC_info[self.K:],self.CRC_polynomial)
        #print(check)
        if check==True:
          res_list_num=i
          break
    #print("CRC_err")
    
    return EST_codeword[res_list_num,self.itr_num]


# In[373]:


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

    EST_information=EST_codeword[self.info_bits]
    return EST_information


# In[374]:


class polar_code(encoding,decoding):
  def __init__(self,N):
    super().__init__(N)

  def main_func(self,EbNodB): 
    information,codeword=self.polar_encode()
    Lc=-1*ch.generate_LLR(codeword,EbNodB)#デコーダが＋、ー逆になってしまうので-１をかける
    EST_information=self.polar_decode(Lc) 
      
    if len(EST_information)!=len(information):
      print("len_err")

    return information,EST_information


# In[375]:


if __name__=="__main__":

    N=1024
    pc=polar_code(N)
    #a,b=pc.main_func(1)
    #print(len(a))
    #print(len(b))
    def output(EbNodB):
      count_err=0
      count_all=0
      count_berr=0
      count_ball=0
      MAX_ERR=8

      while count_err<MAX_ERR:
        
        pc=polar_code(N)
        information,EST_information=pc.main_func(EbNodB)
      
        if np.any(information!=EST_information):#BLOCK error check
          count_err+=1
        
        count_all+=1

        #calculate bit error rate 
        count_berr+=np.sum(information!=EST_information)
        count_ball+=N

        print("\r","count_all=",count_all,",count_err=",count_err,"count_ball="              ,count_ball,"count_berr=",count_berr,end="")

      print("BER=",count_berr/count_ball)
      return  count_err,count_all,count_berr,count_all
    
    output(0)