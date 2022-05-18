#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import math
import math
import sys
import os
sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
from capacity_estimation.calc_capacity import make_BMI 

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

    def main_const(self,N,K,design_SNR,M=2):
        
        '''
        input:
        N:codeword length
        K:info length
        design_SNR target EsNo dB
        M:modulation order
        '''
        
        #check if mapping is the divisor of N
        if N%int(math.log2(M))!=0:
            print("mapping error")
        
        if M==2:
            #print("BPSK")
            gamma=10**(design_SNR/10)#BPSK
            gamma=gamma*np.ones(N)
            
        elif M==4:
            #print("QPSK")
            dmin=(6/(M-1))**(1/2)
            gamma=10**(design_SNR/10)*dmin/2 #QPSK(sqrt(2))
            gamma=gamma*np.ones(N)
        
        else:
            #print("multi QAM")
            tmp=make_BMI(design_SNR,M)
            for a in range(len(tmp)):
                tmp[a]=self.calc_J_inv(tmp[a])
            gamma=np.tile(tmp,N//int(math.log2(M)))
            
        xi=np.log(gamma)
            
        
        #check if xi array is length N
        if len(xi)!=N:
            print("xi length error!")
            print(len(xi))
                       
        n=int(math.log2(N))
        for i in range(1,n+1):
            J=2**i
            for k in range(0,int(N/J)):
                for j in range(0,int(J/2)):
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
    const=RCA()

    print(const.main_const(1024,512,1,256))


# In[ ]:




