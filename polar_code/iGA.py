import numpy as np
import math
from decimal import *
import sympy as sp
import sys
import os
sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
from capacity_estimation.calc_capacity import make_BMI_list 
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
    #print(self.Z_0)
    self.Z_1=self.xi(self.G_1)
    #print(self.Z_1)
    self.Z_2=self.xi(self.G_2)
    #print(self.Z_2)
    #from IPython.core.debugger import Pdb; Pdb().set_trace()

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

    if zeta>=self.Z_0:
      gamma=-2*zeta+zeta**2+zeta**3
      count=1

    elif self.Z_1<=zeta<self.Z_0:
      gamma=(-1*self.a1-(self.a1**2-4*self.a2*(self.a0-zeta))**(1/2))/(2*self.a2)
      count=2
      
    elif self.Z_2<zeta<self.Z_1:
      gamma=((zeta-self.b)/self.a)**(1/self.c)
      count=3

    elif zeta<=self.Z_2:
      gamma=self.bisection_method(zeta)
      count=4
      #gamma=-4*zeta

    if gamma<0:
      print(gamma)
      print(zeta)
      print("gamma is - err")
      print(count)

    return gamma

  def bisection_method(self,zeta):

    #set constant
    min_num=self.G_2
    max_num=-4*(zeta-1/2*math.log(math.pi))
    error_accept=1/10000

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
      print("bisection method")

    if gamma==0.0:
      print("gamma is underflow")
      print(gamma)
      print(zeta) 

    return gamma

  def main_const(self,N,K,design_SNR,M=2,**kwargs): 
    
    '''
    input:
    N:codeword length
    K:info length
    design_SNR target EsNo dB
    M:modulation order
    '''
    #extract from dictionary 
    if kwargs.get('BICM_int') is not None:
        BICM_int=kwargs.get("BICM_int")
        BICM_deint=np.argsort(BICM_int)
        BICM=True
    else:
        BICM=False
    
    if kwargs.get('soft_output') is not None:
        soft_output=kwargs.get("soft_output")
    else:
        soft_output=False
    
    #check if mapping is the divisor of N
    if N%int(np.log2(M))!=0:
        print("mapping error")
     
    if M==2:
      gamma=np.ones(N)
      gamma=gamma*4*(10 ** (design_SNR / 10)) #mean of LLR when transmit all 0
    elif M==4:
      #print(M)
      dmin=(6/(M-1))**(1/2)
      #dmin=1
      #print(dmin/2)
      gamma=np.ones(N)
      gamma=gamma*4*(10 ** (design_SNR / 10))*(dmin/2)
    else:
      #print("multi QAM")
      tmp=make_BMI_list(design_SNR,M)
      #print(tmp)
      for a in range(len(tmp)):
        tmp[a]=4*self.calc_J_inv(tmp[a])
      #print(tmp)
      gamma=np.tile(tmp,N//int(np.log2(M)))
    
    if BICM==True:
        gamma=gamma[BICM_deint]
            
    #tmp=np.zeros(len(gamma))
    #n=int(log2(N))
    #bit reversal order
    #for i in range(len(tmp)):
        #tmp[i]=gamma[self.reverse(i,n)]  
    
    n=int(np.log2(N))
    for i in range(0,n):
        J = 2**(n-i)
        for k in range(0,int(N/J)):
            #import pdb; pdb.set_trace()
            for j in range(0,int(J/2)):
                u1 = gamma[k * J + j ]
                u2 = gamma[k * J + j + int(J/2) ]
                #if u1!=u2:
                    #print("u1 not equal u2")

                if u1<=self.G_0 and u1<=self.G_1:
                    #es=(u1**2)/2-(u1**3)/2+2*(u1**4)/3
                    #res=1/2*u1**2-1/2*u1**3+2/3*u1**4
                    res=1/2*u1*u2-1/4*u1*u2**2-1/4*u1**2*u2+5/24*u1*u2**3+1/4*u1**2*u2**2+5/24*u1**3*u2
                else:
                    '''
                    calc
                    a=max_str(z1,z2)
                    b=z1+z2
                    and calc
                    ln(exp(a)-exp(b))
                    (a must be greater than b)
                    '''
                    z1=self.xi(u1)
                    z2=self.xi(u2)
                    
                    #a=max(z1,z2)+np.log(1+np.exp(-1*abs(z1-z2)))
                    #b=z1+z2
                    #if a<b:
                        #print("false const")
                    #c=a+np.log(1-np.exp(b-a))
                    c=np.log(np.exp(z1)+np.exp(z2)-np.exp(z1+z2))
                    #print(c)
                    if c>0:
                        print("c is plus err")
                    if np.isnan(c):
                        print("nan err")
                        
                    res=self.xi_inv(c)
                    
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
  def calc_J_inv(I):
    var=2
    if var==1:
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
    
    elif var==2:
        def A(c):
            return (-5+24*np.log(2)*c+2*(13+12*np.log(2)*c*(12*np.log(2)*c-5))**(1/2))**(1/3)
    
        def W0(x):
            '''
            Lambert W function
            reference:
            On the lambert W function
            (3.1)
            '''
            def a(n):
                ((-n)**(n-1)/np.math.factorial(n))*x**n
                
                
            res=0
            for i in range(100):
                res+=a(i)
            return res
        
        if I>1 or I<0:
            print("I is err")
                
        #print("use new funcß")   
        C1=0.055523
        C2=0.721452
        C3=0.999983
        H21=1.396634
        H22=0.872764
        H23=1.148562
        H31=1.266967
        H32=0.938175
        H33=0.986830
        alpha=1.16125
        
        if I<C1:
            gamma=1/4*(1-3/A(I)+A(I))
            
        elif I<C2:
            gamma=(-1/H21*np.log(1-I**(1/H23)))**(1/H22)
            
        elif I<C3:
            gamma=(-1/H31*np.log(1-I**(1/H33)))**(1/H32)
        
        else:
            gamma=1/2*W0(2*(alpha/(1-I))**2)
        
        return gamma
        