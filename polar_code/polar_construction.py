import numpy as np
import math
import os
from decimal import *
import sys
import multiprocessing
import pickle

class coding():
  def __init__(self,N,K,design_SNR=0):
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
    self.design_SNR=design_SNR
    
    #settings
    self.adaptive_design_SNR=False #default:False
    self.systematic_polar=False #default:false
    self.decoder_ver=0 #0:SC 1:SCL 2:CA_SCL

    self.bit_reversal_sequence=self.reverse_bits()

    #for encoder (CRC poly)
    #1+x+x^2+....
    self.CRC_polynomial =np.array([1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1])
      
    self.const=Improved_GA()#monte_carlo() #Improved_GA()
      
      #flozen_bit selection 
    if self.decoder_ver==2:
      CRC_len=len(self.CRC_polynomial)-1
      self.frozen_bits,self.info_bits=self.const.main_const(self.N,self.K+CRC_len,self.design_SNR)
    else:
      self.frozen_bits,self.info_bits=self.const.main_const(self.N,self.K,self.design_SNR)
    
    if self.systematic_polar==True:
      self.filename="systematic_"+self.filename
    
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

class RCA():
    @staticmethod
    def reverse(index,n):
        '''
        make n into bit reversal order
        '''
        tmp=format (index,'b')
        tmp=tmp.zfill(n+1)[:0:-1]
        res=int(tmp,2) 
        return res

    @staticmethod
    def indices_of_elements(v,l):
        tmp=np.argsort(v)
        res=tmp[0:l]
        return res

    def main_const(self,N,K,design_SNR):
        #design SNRが複数あるかどうか判定する
        if type(design_SNR).__module__ != np.__name__:
            design_SNR=design_SNR*np.ones(N)
        elif len(design_SNR)!=N:
            print("design_SNR_len_error!")
            
        xi=np.log(design_SNR)
        
        n=int(math.log2(N))
        for i in range(1,n+1):
            J=2**i
            for k in range(0,int(N/J)-1):
                for j in range(0,int(J/2)-1):
                    xi0=xi[k*J+j]
                    xi1=xi[k*J+j+int(J/2)] 
                    lambda0=self.calc_lambda(xi0)
                    lambda1=self.calc_lambda(xi1)
                    xi[k*J+j]=self.calc_lambda(max(lambda0,lambda1)+math.log(1+math.exp(-1*abs(lambda0-lambda1))))
                    xi[k*J+j+int(J/2)]=max(xi0,xi1)+math.log(1+math.exp(-1*abs(xi0-xi1)))
        #return xi
    
        tmp=self.indices_of_elements(xi,N)
        frozen_bits=np.sort(tmp[:N-K])
        info_bits=np.sort(tmp[N-K:])
        
        #bit reversal order
        for i in range(len(frozen_bits)):
            frozen_bits[i]=self.reverse(frozen_bits[i],n)
        frozen_bits=np.sort(frozen_bits)
            
        for i in range(len(info_bits)):
            info_bits[i]=self.reverse(info_bits[i],n)
        info_bits=np.sort(info_bits)
        
        return frozen_bits,info_bits

    @staticmethod
    def calc_lambda(xi):
        Alpha=1.16125
        Gamma1=0.04
        Gamma2=1
        Gamma3=10
        Xi0=-11.3143
        C1=0.55523
        C2=0.721452
        H21=1.396634
        H22=0.872764
        H23=1.148562
        H31=1.266967
        H32=0.938175
        H33=0.986830

        if xi<Xi0:
            B=math.log(2)+2*math.log(math.log(2))+2*math.log(Alpha)-2*xi
            return math.log(B+(1/B-1)*math.log(B))-math.log(2)

        gamma=math.exp(xi)
        if gamma>Gamma3:
            return math.log(math.log(2))+math.log(Alpha)-gamma-xi/2

        elif gamma<Gamma1:
            U=1-(gamma-gamma**2+4/3*gamma**3)/math.log(2)

        elif gamma<Gamma2:
            U=1-(1-math.exp(-1*H21*(gamma**H22)))**H23
            
        else:
            U=1-(1-math.exp(-1*H31*(gamma**H32)))*H33
            
        if U<C1:
            A=(-5+24*math.log(2)*U+2*math.sqrt(13+12*math.log(2)*U*(12*math.log(2)*U-5)))**(1/3)
            return math.log(1-3/A+A)-2*math.log(2)
            
        elif U<C2:
            return (math.log(-1*math.log(1-U**(1/H23)))-math.log(H21))/H22

        else:
            return (math.log(-1*math.log(1-U**(1/H33)))-math.log(H31))/H32
 
class Improved_GA():
    
  def __init__(self):
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

  def reverse(self,index,n):
    '''
    make n into bit reversal order
    '''
    tmp=format (index,'b')
    tmp=tmp.zfill(n+1)[:0:-1]
    res=int(tmp,2) 
    return res

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
      print(gamma)
      print(zeta)
      print("gamma is - err")

    return gamma

  def bisection_method(self,zeta):
  
    #set constant
    min_num=self.G_2
    max_num=-4*(zeta-1/2*math.log(math.pi))
    error_accept=1/100

    def f(x):
      zeta=-1*x/4+math.log(math.pi)/2-math.log(x)/2+math.log(1-(math.pi**2)/(4*x)+self.K_0/(x**2))
      return zeta

    #initial value
    a=min_num
    b=max_num
    error=b-a

    #very small zeta situation
    if f(max_num)>zeta:
      print("error")
      #gamma=max_num

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
      print(a)
      print(b)
      print(gamma)
      print("gamma is - err")    
      
    if gamma==0.0:
      print("gamma is underflow")
      print(gamma)
      print(zeta) 

    return gamma

  def main_const(self,N,K,design_SNR,M=2):
    # if bit_reverse or not
    bit_reverse=True,
    #make n where 2**n=N
    n=np.log2(N).astype(int)
    
    gamma=np.zeros(N)
    
    if M==2:
      gamma[0]=4*(10 ** (design_SNR / 10)) #mean of LLR when transmit all 0
    else:
      dmin=(6/(M-1))**(1/2)
      gamma[0]=4*(10 ** (design_SNR / 10))*dmin
    
    for i in range(1,n+1):
      J=2**(i-1)
      for j in range(0,J):
        u=gamma[j]
        if u<=self.G_0:
          gamma[j]=(u**2)/2-(u**3)/2+2*(u**4)/3
        else:
          z=self.xi(u)
          gamma[j]=self.xi_inv(z+math.log(2-math.e**z))
        
        gamma[j+J]=2*u

    tmp=self.indices_of_elements(gamma,N)
    frozen_bits=np.sort(tmp[:N-K])
    info_bits=np.sort(tmp[N-K:])

    if bit_reverse==True:
      for i in range(len(frozen_bits)):
        frozen_bits[i]=self.reverse(frozen_bits[i],n)
      frozen_bits=np.sort(frozen_bits)

      for i in range(len(info_bits)):
        info_bits[i]=self.reverse(info_bits[i],n)
      info_bits=np.sort(info_bits)

    return frozen_bits,info_bits

  @staticmethod
  def indices_of_elements(v,l):
    tmp=np.argsort(v)
    res=tmp[0:l]
    return res

'''
# In[8]:


class Improved_GA(Improved_GA):
  def maxstr(self,a,b):
    def f(c):
      return np.log(1+np.exp(-1*c))
    return max(a,b)+f(abs(a-b))


# In[9]:


class Improved_GA(Improved_GA):
  def left_operation(self,gamma1,gamma2):
    
    #calc zeta
    zeta1=self.xi(gamma1)
    zeta2=self.xi(gamma2)
    
    if gamma1<=self.G_0 and gamma2<=self.G_0:
        
      sq=1/2*gamma1*gamma2
      cu=-1/4*gamma1*(gamma2**2)-1/4*(gamma1**2)*gamma2
      fo=5/24*gamma1*(gamma2**3)+1/4*(gamma1**2)*(gamma2**2)+5/24*(gamma1**3)*gamma2
      
      gamma=sq+cu+fo
      
            
    else:
      
      gamma=self.xi_inv(zeta1+math.log(2-math.e**zeta1)) 
      #tmp=self.maxstr(zeta1,zeta2)
      #zeta=self.maxstr(tmp,zeta1-zeta2)

      #gamma=self.xi_inv(zeta)
      
      #gamma=inv_phi(1-(1-phi(gamma1))*(1-phi(gamma2)))
      #print("1")
    return gamma
          
  def right_operation(self,gamma1,gamma2):
    #print("0")
    return gamma1+gamma2
  
  
  def main_const(self,N,K,high_des,beta=1000,ind_high_des=False,ind_low_des=False):
    
    n=np.log2(N).astype(int)
    gamma=np.zeros((n+1,N)) #matrix
    
    #初期値の代入
    if beta==1000:
      gamma[0,:]=4*(10 ** (high_des / 10))
      
    else:
      gamma[0,ind_high_des]=4*(10 ** (high_des / 10))
      gamma[0,ind_low_des]=(beta**2)*4*(10 ** (high_des / 10))
    
    for i in range(1,gamma.shape[0]):
      for j in range(gamma.shape[1]):
        if (j//2**(n-i))%2==0:
          gamma[i,j]=self.left_operation(gamma[i-1,j],gamma[i-1,j+2**(n-i)])
        
        else :
          gamma[i,j]=self.right_operation(gamma[i-1,j],gamma[i-1,j-2**(n-i)])
    
    tmp=np.argsort(gamma[n,:])
    
    frozen_bits=np.sort(tmp[:N-K])
    info_bits=np.sort(tmp[N-K:])
    
    return frozen_bits, info_bits
'''

# In[10]:

class GA():
    
  def main_const(self,N,K,high_des,beta=1000,ind_high_des=False,ind_low_des=False):
    
    #make n where 2**n=N
    n=np.log2(N).astype(int)
    
    #O(N**2) complexity
    #constant for GA operation
    a=-0.4527
    b=0.0218
    c=0.86
    G=10
    
    def phi(gamma):   
      if gamma<=G:
        zeta=math.exp(a*gamma**c+b)
        
      else:
        zeta=(math.pi/gamma)**(1/2)*math.exp(-gamma/4)*(1-10/(7*gamma))
      
      return zeta
    
    Z=phi(G)
    
    def inv_phi(zeta):
      if zeta>=Z:
        gamma=((math.log(zeta)-b)/a)**(1/c)
      else:
        gamma=self.bisection_method(zeta)
    
      return gamma
    
    
    def left_operation(gamma1,gamma2):
      
      #calc zeta
      zeta1=phi(gamma1)
      zeta2=phi(gamma2)
           
      d1=Decimal("1")
      d2=Decimal(zeta1)
      d3=Decimal(zeta2)
      
      zeta=d1-(d1-d2)*(d1-d3)
      #print(zeta)
      
      #for underflow
      if zeta==0:
        zeta=10**(-50)

      gamma=inv_phi(zeta)
      
      #gamma=inv_phi(1-(1-phi(gamma1))*(1-phi(gamma2)))
      #print("1")
      return gamma
            
    def right_operation(gamma1,gamma2):
      #print("0")
      return gamma1+gamma2
    
    #main operation
    
    gamma=np.zeros((n+1,N)) #matrix
    
    if beta==1000:
      gamma[0,:]=4*(10 ** (high_des / 10))
      
    else:
      gamma[0,ind_high_des]=4*(10 ** (high_des / 10))
      gamma[0,ind_low_des]=(beta**2)*4*(10 ** (high_des / 10))
    
    for i in range(1,gamma.shape[0]):
      for j in range(gamma.shape[1]):
        if (j//2**(n-i))%2==0:
          gamma[i,j]=left_operation(gamma[i-1,j],gamma[i-1,j+2**(n-i)])
        
        else :
          gamma[i,j]=right_operation(gamma[i-1,j],gamma[i-1,j-2**(n-i)])
    
    tmp=np.argsort(gamma[n,:])
    
    frozen_bits=np.sort(tmp[:N-K])
    info_bits=np.sort(tmp[N-K:])
    
    '''
    削除予定
    if bit_reverse==True:
      for i in range(len(frozen_bits)):
        frozen_bits[i]=self.reverse(frozen_bits[i])
      frozen_bits=np.sort(frozen_bits)

      for i in range(len(info_bits)):
        info_bits[i]=self.reverse(info_bits[i])
      info_bits=np.sort(info_bits)
    '''

    return frozen_bits,info_bits

  
  def bisection_method(self,zeta):
      
    #set constant
    
    min_num=10
    max_num=-4*math.log(zeta)
    error_accept=10**(-10)

    def f(x):
      zeta=(math.pi/x)**(1/2)*math.exp(-x/4)*(1-10/(7*x))
      return zeta

    #initial value
    a=min_num
    b=max_num
    error=b-a

    #very small zeta situation
    if f(max_num)>zeta:
      print("error")
      #gamma=max_num

    count=0
    while error>error_accept:
      count+=1
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
      
    if gamma==0.0:
      print("gamma is underflow")
      print(gamma)
      print(zeta) 

    return gamma

'''
# In[12]:


class inv_GA():
  
  def __init__(self):
    self.a=-0.4527
    self.b=0.0218
    self.c=0.86
    self.G=10
    self.Z=self.phi(self.G)
  
  def phi(self,gamma):   
    if gamma<=self.G:
      zeta=math.exp(self.a*gamma**self.c+self.b)
    else:
      zeta=(math.pi/gamma)**(1/2)*math.exp(-gamma/4)*(1-10/(7*gamma))
    
    return zeta
  
  def inv_phi(self,zeta):
    if zeta>=self.Z:
      gamma=((math.log(zeta)-self.b)/self.a)**(1/self.c)
    else:
      gamma=self.bisection_method(zeta)
    return gamma 
  
  def main_const(self,N,frozen_bits,info_bits):
    
    #make n where 2**n=N
    n=np.log2(N).astype(int)
    
    zero=1
    inf=10000
    
    gamma=np.zeros((n+1,N)) #matrix
    
    #一番下の行に代入
    gamma[n,frozen_bits]=zero
    gamma[n,info_bits]=inf
    
    def left_operation(gamma1,gamma2):
      #C=phi(gamma1)
      #gamma2-x=x'(x+x'=gamma2)
      #x=gamma
      #res1=x
      #res2=A-x
      #A=gamma2
      
      #calc zeta
      #print(gamma1)
      
      zeta=self.phi(gamma1)
      #if gamma2<1: #しきい値1は適当に設定した。
        #逆関数から計算する
        #if f(gamma2/2,gamma2)<zeta:
          #取りうる値のペアではなかったとき、取りうる値の中で最小の値を出力
          #res1=gamma2/2
        
        #else:
          #res1=self.res.subs([(self.x, gamma1),(self.A, gamma2)])
      
      #else:
      res1=self.bisection_method_for_inv_GA(zeta,gamma2)
      
      if np.random.randint(2,size=1)==0:
        res2=gamma2-res1
       
      res2=gamma2-res1
      #res1<res2 と仮定した
      
      if res1<0 or res2<0:
        print("res minus error")
      
      return res1,res2
    
    #削除予定        
    #def right_operation(gamma1,gamma2):
      
    #  res=gamma1-gamma2
      
    #  if res<0:
    #    print("right_operation error")
    
    #  #print("0")
    #  return res
    
    
    #inv_GA process
    for i in reversed(range(0,gamma.shape[0]-1)):
      for j in range(gamma.shape[1]):
        if (j//2**(n-i-1))%2==0:
          gamma[i,j],gamma[i,j+2**(n-i-1)]=left_operation(gamma[i+1,j],gamma[i+1,j+2**(n-i-1)])
      
        #print(i,j) 
        #print(gamma[i,j])
    
    print(gamma[0,:])
    #gamma[0,:]が大きいほど、信号点配置が大きくなければならない
    tmp=np.argsort(gamma[0,:])
    low_power_bits=np.sort(tmp[:N//2])
    high_power_bits=np.sort(tmp[N//2:])
    
    return low_power_bits,high_power_bits
    


# In[13]:


class inv_GA(inv_GA):
  def bisection_method(self,zeta):
      
    #set constant
    min_num=10
    max_num=-4*math.log(zeta)
    error_accept=10**(-10)

    def f(x):
      zeta=(math.pi/x)**(1/2)*math.exp(-x/4)*(1-10/(7*x))
      return zeta

    #initial value
    a=min_num
    b=max_num
    error=b-a

    #very small zeta situation
    if f(max_num)>zeta:
      print("error1")
      #gamma=max_num

    count=0
    while error>error_accept:
      count+=1
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
      
    if gamma==0.0:
      print("gamma is underflow")
      print(gamma)
      print(zeta) 

    return gamma


# In[14]:


class inv_GA(inv_GA):
  def bisection_method_for_inv_GA(self,zeta,A):
    #増大関数について考える
      
    #set constant
    min_num=0
    max_num=A/2
    error_accept=10**(-10)

    def f(x):
      zeta=self.phi(x)+self.phi(A-x)-self.phi(x)*self.phi(A-x)
      #if zeta>1:
        #print("zeta error")
        #print(zeta)
        
      return zeta

    #initial value
    a=min_num
    b=max_num
    error=b-a

    #very small zeta situation
    if f(max_num)>zeta:
      #取りうる値のペアではなかったとき、取りうる値の中で最小の値を出力
      return A/2
      #print("error2")
      #print(zeta)
      #print(f(max_num))
      #gamma=max_num

    count=0
    while error>error_accept:
      count+=1
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
      
    if gamma==0.0:
      print("gamma is underflow")
      print(gamma)
      print(zeta) 

    return gamma
'''

# In[15]:


class monte_carlo():
      
  def main_const(self,N,K,design_SNR,channel):
      
    #check
    #design_SNR=100
    
    c=self.output(N,design_SNR,channel)
    
    tmp=np.argsort(c)[::-1]
    frozen_bits=np.sort(tmp[:N-K])
    info_bits=np.sort(tmp[N-K:])
    
    return frozen_bits,info_bits

  @staticmethod
  def make_d(inputs):
    #unzip inputs
    ec,dc,ch,design_SNR=inputs
    ec=pickle.loads(ec)
    dc=pickle.loads(dc)
    ch=pickle.loads(ch)
    
    #initial constant
    epoch=10**3
    c=np.zeros(ec.N)
    for _ in range(epoch):
          #main
      info,cwd=ec.polar_encode()
      Lc=-1*ch.generate_LLR(cwd,design_SNR)#デコーダが＋、ー逆になってしまうので-１をかける
      #rint(Lc)
      llr=dc.polar_decode(Lc,info) 
      #print(llr)
      #from IPython.core.debugger import Pdb; Pdb().set_trace()
      
      d=np.zeros(len(llr))
        #print(llr)
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
      d[(2*info-1)*llr<0]=0
      d[(2*info-1)*llr>=0]=1
      
      c=c+d
    
    return c

  def make_c(self,N,design_SNR,channel):
    #append path
    sys.path.append("polar_code")
    sys.path.append("channel")
    from polar_encode import encoding
    from polar_decode import decoding
    from AWGN import _AWGN
    
    
    #initialize polarcode
    monte_carlo_const=True
    cd=coding(N,256,design_SNR,channel,monte_carlo_const)
    ec=encoding(cd)
    dc=decoding(cd,ec)
    #unzip channel
    if channel!=False:
      ch=channel
    else:
      ch=_AWGN()
    cd=pickle.dumps(cd)
    ec=pickle.dumps(ec)
    dc=pickle.dumps(dc)
    ch=pickle.dumps(ch)
    
    c=np.zeros(N)
    multi_num=100 #the number of multiprocessing 
    
    inputs=[]
    for _ in range(multi_num):
      inputs+=[(ec,dc,ch,design_SNR)]
    
    #multiprocessing
    with multiprocessing.Pool(processes=multi_num) as pool:
      res = pool.map(self.make_d,inputs)
    
    for i in range(multi_num):
      c=c+res[i]
      
    return c
  
  def output(self,N,design_SNR,channel):

    #print(channel)
    #unzip channel
    if channel!=False:
      beta=channel.beta
      M=channel.M
      
      #from IPython.core.debugger import Pdb; Pdb().set_trace()
    else:
      #this is the same as initial value of _AWGN class
      beta=1
      M=2
      
    
    # directory make
    current_directory="/home/kaneko/Dropbox/programming/wireless_communication/polar_code"
    #current_directory=os.getcwd()
    dir_name="monte_carlo_construction"
    dir_name=current_directory+"/"+dir_name
    
    try:
      os.makedirs(dir_name)
    except FileExistsError:
      pass
    
    #make filename 
    if M==2:
      const="BPSK"
    elif M==4:
      const="PAM"
      
    filename=const+"_{}_{}_{}".format(beta,N,design_SNR)
    
    #if file exists, then load txt file
    filename=dir_name+"/"+filename
    
    try:
      c=np.loadtxt(filename)
    except FileNotFoundError:
      c=self.make_c(N,design_SNR,channel)
      #export file
      np.savetxt(filename,c)

    return c
  
if __name__=="__main__":
  N=1024
  for k in range(N):
    myPC=coding(N,k)
    #print(k)