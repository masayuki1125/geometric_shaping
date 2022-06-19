import numpy as np
import math
from decimal import *
import sympy as sp


# In[3]:


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

  def main_const(self,N,K,design_SNR,M=2,**kwargs): 
    if kwargs.get('soft_output') is not None:
        soft_output=kwargs.get("soft_output")
    else:
        soft_output=False
      
        #make n where 2**n=N
    n=np.log2(N).astype(int)
    
    gamma=np.ones(N)
     
     
    if M==2:
      gamma=gamma*4*(10 ** (design_SNR / 10)) #mean of LLR when transmit all 0
    else:
      #print(M)
      dmin=(6/(M-1))**(1/2)
      #dmin=1
      #print(dmin/2)
      gamma=gamma*4*(10 ** (design_SNR / 10))*(dmin/2)
        
    for i in range(0,n):
        J = 2**(n-i)
        for k in range(0,int(N/J)):
            #import pdb; pdb.set_trace()
            for j in range(0,int(J/2)):
                u1 = gamma[k * J + j ]
                u2 = gamma[k * J + j + int(J/2) ]:

                if u1<=self.G_0 and u1<=self.G_1:
                    res=1/2*u1*u2-1/4*u1*u2**2-1/4*u1**2*u2+5/24*u1*u2**3+1/4*u1**2*u2**2+5/24*u1**3*u2
                else:
                    z1=self.xi(u1)
                    z2=self.xi(u2)
                    res=self.xi_inv(z+math.log(2-math.e**z))
                    
                gamma[k * J + j] = res
                gamma[k * J + j + int(J/2)] = u1+u2
    
    if soft_output==True:
        return gamma
    
    tmp=self.indices_of_elements(gamma,N)
    frozen_bits=np.sort(tmp[:N-K])
    info_bits=np.sort(tmp[N-K:])
    
    return frozen_bits,info_bits

  @staticmethod
  def indices_of_elements(v,l):
    tmp=np.argsort(v)
    res=tmp[0:l]
    return res

  @staticmethod
  def maxstr(a,b):
    def f(c):
      return np.log(1+np.exp(-1*c))
    return max(a,b)+f(abs(a-b))

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