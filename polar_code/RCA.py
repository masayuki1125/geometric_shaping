#!/usr/bin/env python
# coding: utf-8

# In[59]:

from numpy import log, log10,log2, exp, sqrt,e
from numpy import power as pow
from numpy import log,exp,fabs

import numpy as np
import math
import math
import sys
import os
sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
from capacity_estimation.calc_capacity import make_BMI_list 

# In[60]:


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
    
    def main_const_unif(self,N,K,design_SNR,M=2):
        
        xi=np.zeros(N)
        
        if M==2:
            #print("BPSK")
            xi[0] = 0.1 * design_SNR / log10(e)#BPSK
            #gamma=design_SNR
            
        else:
            #print("QPSK")
            dmin=(6/(M-1))**(1/2)
            xi[0] = 0.1 * design_SNR / log10(e) +log(dmin/2) #QPSK(sqrt(2))
        
        n=int(log2(N))
        #print("loop st")
        for i in range(1,n+1):
            J=2**(i-1)
            
            for j in range(0,J):
                #print(j)
                xi0=xi[j]
                lambda0=self.calc_lambda(xi0)
                xi[j]=self.calc_lambda(lambda0+log(2))
                xi[j+J]=xi0+log(2)
                #from IPython.core.debugger import Pdb; Pdb().set_trace()
    
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
        
        #check if mapping is the divisor of N
        if N%int(log2(M))!=0:
            print("mapping error")
        
        if M==2:
            #print("BPSK")
            gamma=10**(design_SNR/10)#BPSK
            gamma=gamma*np.ones(N,dtype="float128")
            
        elif M==4:
            #print("QPSK")
            dmin=(6/(M-1))**(1/2)
            gamma=10**(design_SNR/10)*dmin/2 #QPSK(sqrt(2))
            gamma=gamma*np.ones(N,dtype="float128")
        
        else:
            #print("multi QAM")
            tmp=make_BMI_list(design_SNR,M)
            #print(tmp)
            for a in range(len(tmp)):
                tmp[a]=self.calc_J_inv(tmp[a])
            gamma=np.tile(tmp,N//int(log2(M)))
            #print(gamma)
 
        xi=log(gamma).astype("float128")
        
        if BICM==True:
            xi=xi[BICM_deint]
            
        
        #check if xi array is length N
        if len(xi)!=N:
            print("xi length error!")
            print(len(xi))
                       
        n=int(log2(N))
        
        for i in range(0,n):
            J = 2**(i+1)
            for k in range(0,int(N/J)):
                for j in range(0,int(J/2)):
                    xi0 = xi[k * J + j ]
                    xi1 = xi[k * J + j + int(J/2) ]
                    L0 = self.calc_lambda( xi0 )
                    L1 = self.calc_lambda( xi1 )
                    
                    xi[k * J + j] = self.calc_lambda( max( L0, L1 )  + log( 1.0 + exp( - fabs( L0 - L1 ) ) ) )
                    xi[k * J + j + int(J/2)] = max( xi0, xi1 ) + log( 1.0 + exp( - fabs( xi0 - xi1 ) ) )
                    
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
    '''
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
        #first if st
        if xi<Xi0:
            B=math.log(2)+2*math.log(math.log(2))+2*math.log(Alpha)-2*xi
            return math.log(B+(1/B-1)*math.log(B))-math.log(2)
        
        #first if fin
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
    '''
    @staticmethod
    def calc_lambda(xi):
        alpha =  1.16125
        h21 = 1.396634
        h22 = 0.872764
        h23 = 1.148562
        h31 = 1.266967
        h32 = 0.938175
        h33 = 0.986830
        
        if( xi < -11.3143 ):
            B = log( 2.0 ) + 2.0 * log( log( 2.0 ) )+ 2.0 * log( alpha )- 2.0 * xi
            rt = log( B + ( 1.0 / B - 1.0 ) * log( B ) ) - log( 2.0 )
            return rt
        g = exp( xi )
        if( g > 10.0 ):
            return log( log( 2.0 ) ) + log( alpha ) - g - 0.5 * xi
        elif ( g < 0.04 ):
            L = 1.0 - ( g - pow( g, 2.0 ) + 4.0 / 3.0 * pow( g, 3.0 ) )/ log( 2.0 )
        elif( g < 1.0 ):
            L = 1.0 - pow( 1.0 - exp( - h21 * pow( g, h22 ) ), h23)
        else: 
            L = 1.0 - pow( 1.0 - exp( - h31 * pow( g, h32 ) ), h33)
        if( L < 0.055523 ):
            A = pow( -5.0 + 24.0 * log( 2.0 ) * L + 2.0 * sqrt( 13.0 + 12.0 * log(2.0) * L * ( 12.0 * log(2.0) * L - 5.0 ) ), 1.0 / 3.0 )
            rt = log( 1.0 - 3.0 / A + A ) - 2.0 * log( 2.0 )
            return rt
        elif( L < 0.721452 ):
            rt = ( log( - log( 1.0 - pow( L, 1.0 / h23 ) ) ) - log( h21 ) ) / h22
            return rt
        else:
            rt = ( log( - log( 1.0 - pow( L, 1.0 / h33 ) ) ) - log( h31 ) ) / h32
            return rt
            
    
    @staticmethod
    def calc_J_inv(I):
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


# In[64]:


#import matplotlib.pyplot as plt

#x=np.arange(-1000,800)
#y=np.zeros((len(x)))
#for i,a in enumerate(x):
#    y[i]=calc_lambda(a)
    
#plt.plot(x, y)


# In[65]:

if __name__=="__main__":
    from iGA import Improved_GA
    const=RCA()
    const2=Improved_GA()

    a,b=const.main_const(1024,512,1,2)
    c,d=const.main_const_unif(1024,512,1,2)
    
    e,f=const2.main_const(1024,512,1,2)
    
    print(np.sum(a!=c))
    #print(a)
    #print(c)
    print(np.sum(b!=d))
    
    print(np.sum(c!=e))
    print(np.sum(d!=f))
    
    print(np.sum(a!=e))
    print(np.sum(b!=f))
    
    for i in range(len(e)):
        count=0
        if np.any(e[i]!=a):
            count+=1
    
    print("count",count)
    


# In[ ]:




