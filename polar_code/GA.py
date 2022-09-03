import numpy as np
import math
from decimal import *
import sympy as sp
import sys
import os
sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
from capacity_estimation.calc_capacity import make_BMI_list 

class GA():

    def __init__(self):
        self.a=-0.4527
        self.b=0.0218
        self.c=0.86
        self.G=10
        self.Z=self.phi(self.G)

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
            dmin=(6/(M-1))**(1/2)
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
        
        #print(gamma)
        if BICM==True:
            gamma=gamma[BICM_deint]
        
        #print(gamma)
        n=int(np.log2(N))
        
        for i in range(0,n):
            J = 2**(n-i)
            #print(int(J/2))
            for k in range(0,int(N/J)):
                
                for j in range(0,int(J/2)):
                    u1 = gamma[k * J + j ]
                    u2 = gamma[k * J + j + int(J/2) ]

                    #左側ノードの計算をする
                    
                    res=self.left_operation(u1,u2)
                    gamma[k * J + j] = res
                    gamma[k * J + j + int(J/2)] = u1+u2
        
            #print(gamma)
            #from IPython.core.debugger import Pdb; Pdb().set_trace()  
        if soft_output==True:
            return gamma
        
        tmp=self.indices_of_elements(gamma,N)
        frozen_bits=np.sort(tmp[:N-K])
        info_bits=np.sort(tmp[N-K:])
        
        return frozen_bits,info_bits
    
    def phi(self,gamma):   
        if gamma<=self.G:
            zeta=np.exp(self.a*gamma**self.c+self.b)
            
        else:
            zeta=(np.pi/gamma)**(1/2)*np.exp(-gamma/4)*(1-10/(7*gamma))
        
        return zeta
    
    def inv_phi(self,zeta):
        if zeta>=self.Z:
            gamma=((np.log(zeta)-self.b)/self.a)**(1/self.c)
        else:
            gamma=self.bisection_method(zeta)
        
        return gamma
    
    def left_operation(self,gamma1,gamma2):
      
      #calc zeta
      zeta1=self.phi(gamma1)
      zeta2=self.phi(gamma2)
           
      zeta=1-(1-zeta1)*(1-zeta2)
      #print(zeta)
      
      #for underflow
      if zeta==0:
        zeta=10**(-50)

      gamma=self.inv_phi(zeta)
      
      return gamma
  
    def bisection_method(self,zeta):
          
        #set constant
        min_num=10
        max_num=-4*np.log(zeta)
        error_accept=10**(-10)

        def f(x):
            zeta=(np.pi/x)**(1/2)*np.exp(-x/4)*(1-10/(7*x))
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
                    ((-n)**(n-1)/np.factorial(n))*x**n
                    
                    
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
        
if __name__=="__main__":
    N=1024
    K=512
    design_SNR=14.0
    const=GA()
    f,i=const.main_const(N,K,design_SNR,256)
    #print(f,i)
    
    from iGA import Improved_GA
    from RCA import RCA
    
    
    const1=Improved_GA()
    f1,i1=const.main_const(N,K,design_SNR,256)
    
    const2=RCA()
    
    print(np.sum(f!=f1))
    
    
    N=1024*8
    K=N//2
    
    a=np.arange(10,20,0.5)
    for design_SNR in a:
        print(design_SNR)
        f1,i1=const.main_const(N,K,design_SNR,256)
        f2,i2=const2.main_const(N,K,design_SNR,256) 
        print(np.sum(f1!=f2))