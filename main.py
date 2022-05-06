import ray
import pickle
import sys
import numpy as np
import math
#my module
from LDPC_code import LDPC_construction
from LDPC_code import LDPC_encode
from polar_code import polar_construction
from polar_code import polar_encode
from polar_code import polar_decode
from polar_code import monte_carlo_construction
from turbo_code import turbo_code
from modulation import modulation
from channel import AWGN

class Mysystem:
    def __init__(self,M,K):
        self.M=M
        self.K=K
        self.N=self.K*int(np.log2(self.M))
        
        self.BICM=True 
        
        if self.BICM==True:
            self.BICM_int=self.srandom_interleave()
            self.BICM_deint=np.argsort(self.BICM_int)
            
            #check
            a=np.arange(self.N)
            b=a[self.BICM_int]
            c=b[self.BICM_deint]
            if np.any(a!=c):
                print("BICM interleaver error!")
                
        #coding
        self.cd=polar_construction.coding(self.N,self.K)
        self.ec=polar_encode.encoding(self.cd)
        self.dc=polar_decode.decoding(self.cd,self.ec)
        #modulation
        self.modem=modulation.QAMModem(self.M)

        #channel
        self.ch=AWGN._AWGN()
        
        #filename
        self.filename="polar_code_{}_{}_{}_exactLLR".format(self.N,self.K,self.M)
        if self.BICM==True:
            self.filename=self.filename+"_BICM"
            
    def srandom_interleave(self):
        
        mod=(self.M)**(1/2)//2
        s=math.floor(math.sqrt(self.N))-5
        print(s)
        #step 1 generate random sequence
        vector=np.arange(self.N,dtype='int')
        np.random.shuffle(vector)

        itr=True
        count=0
        while itr:
            #intialize for each iteration
            heap=np.zeros(self.N,dtype='int')
            position=np.arange(self.N,dtype='int')

            #step2 set first vector to heap
            heap[0]=vector[0]
            position=np.delete(position,0)

            #step3 bubble sort 
            #set to ith heap
            for i in range(1,self.N):
                #serch jth valid position
                for pos,j in enumerate(position):
                    # confirm valid or not
                    for k in range(1,s+1):
                        if i-k>=0 and (abs(heap[i-k]-vector[j])+abs(i-k-j))<=s or (vector[j]%mod)!=(i%mod):
                            '''
                            i-k>=0 : for the part i<s 
                            (abs(heap[i-k]-vector[j]))<=s : srandom interleaver
                            vector[j]//mod!=i//mod : mod M interleaver(such as odd-even)
                            '''
                            #vector[j] is invalid and next vector[j+1]
                            break

                    #vector[j] is valid and set to heap[i]
                    else:
                        heap[i]=vector[j]
                        position=np.delete(position,pos)
                        break
                #if dont exit num at heap[i]
                else:
                    #set invalid sequence to the top and next iteration
                    tmp=vector[position]
                    np.random.shuffle(tmp)
                    vector[0:self.N-i]=tmp
                    vector[self.N-i:self.N]=heap[0:i]
                    break

            #if all the heap num is valid, end iteration
            else:
                itr=False
            
            #print(heap)
            #print(vector)
            print("\r","itr",count,end="")
            count+=1
        
        return heap

    def main_func(self,EsNodB):
        #adaptive dicision of frozen bits
        const=monte_carlo_construction.monte_carlo()
        if self.cd.decoder_ver==2:
            CRC_len=len(self.cd.CRC_polynomial)-1    
            self.cd.frozen_bits,self.cd.info_bits=const.main_const(self.N,self.K+CRC_len,EsNodB,self.M)
        else:
            self.cd.frozen_bits,self.cd.info_bits=const.main_const(self.N,self.K,EsNodB,self.M)
        
        EsNo = 10 ** (EsNodB / 10)
        No=1/EsNo

        info,cwd=self.ec.polar_encode()
        if self.BICM==True:
            cwd=cwd[self.BICM_int]
        TX_conste=self.modem.modulate(cwd)
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,No)
        if self.BICM==True:
            Lc=Lc[self.BICM_deint]
        EST_info=self.dc.polar_decode(Lc)
        
        return info,EST_info

if __name__=='__main__':
    K=256 #symbol数
    M=16
    EsNodB=3
    system=Mysystem(M,K)
    print("\n")
    print(system.N,system.K)
    info,EST_info=system.main_func(EsNodB)
    print(np.sum(info!=EST_info))