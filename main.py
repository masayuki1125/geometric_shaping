import ray
import pickle
import sys
import numpy as np
import math
import os
#my module
from LDPC_code import LDPC_construction
from LDPC_code import LDPC_encode
from LDPC_code import LDPC_decode
from polar_code import polar_construction
from polar_code import polar_encode
from polar_code import polar_decode
from polar_code import RCA
from polar_code import monte_carlo_construction
from turbo_code import turbo_construction
from turbo_code import turbo_encode
from turbo_code import turbo_decode
from modulation import modulation
from channel import AWGN

#FEC=1 #1:polar code 2:turbo code 3:LDPC code
class Mysystem:
    def __init__(self,M,K):
        self.M=M
        self.K=K
        #self.N=self.K*int(np.log2(self.M))
        self.N=self.K*2
        #self.const=monte_carlo_construction.monte_carlo()
        #self.const=polar_construction.Improved_GA()
        #self.const=RCA.RCA()
        self.BICM=False 
        
        if self.BICM==True:
            #make BICM directory
            # directory make
            current_directory="/home/kaneko/Dropbox/programming/geometric_shaping"
            #current_directory=os.getcwd()
            dir_name="BICM_interleaver"
            dir_name=current_directory+"/"+dir_name
            
            try:
                os.makedirs(dir_name)
            except FileExistsError:
                pass
            
            filename="length{}_mod{}".format(self.N,int((self.M)**(1/2)//2))
    
            #if file exists, then load txt file
            filename=dir_name+"/"+filename
            
            try:
                self.BICM_int=np.loadtxt(filename,dtype='int')
            except FileNotFoundError:
                self.BICM_int=self.srandom_interleave()
                #export file
                np.savetxt(filename,self.BICM_int,fmt='%d')
            
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
        
        if self.cd.design_SNR!=EsNodB:
            if self.cd.decoder_ver==2:
                CRC_len=len(self.cd.CRC_polynomial)-1    
                self.cd.frozen_bits,self.cd.info_bits=self.const.main_const(self.N,self.K+CRC_len,EsNodB,self.M)
            else:
                self.cd.frozen_bits,self.cd.info_bits=self.const.main_const(self.N,self.K,EsNodB,self.M)
                
            self.cd.design_SNR==EsNodB
            #for iGA and RCA and monte_carlo construction
        
        '''
        if self.cd.decoder_ver==2:
            CRC_len=len(self.cd.CRC_polynomial)-1  
            if self.BICM==True:  
                self.cd.frozen_bits,self.cd.info_bits=self.const.main_const(self.N,self.K+CRC_len,EsNodB,self.M,BICM_int=self.BICM_int)
            else:
                self.cd.frozen_bits,self.cd.info_bits=self.const.main_const(self.N,self.K+CRC_len,EsNodB,self.M)
        else:
            if self.BICM==True:  
                self.cd.frozen_bits,self.cd.info_bits=self.const.main_const(self.N,self.K,EsNodB,self.M,BICM_int=self.BICM_int)
            else:
                self.cd.frozen_bits,self.cd.info_bits=self.const.main_const(self.N,self.K,EsNodB,self.M)
        '''#for monte carlo construction
                
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
'''    
class Mysystem():
    def __init__(self,M,K):
        self.M=M
        self.K=K
        #self.N=self.K*int(np.log2(self.M))
        self.N=self.K*2
        self.BICM=True 
        
        if self.BICM==True:
            #make BICM directory
            # directory make
            current_directory="/home/kaneko/Dropbox/programming/geometric_shaping"
            #current_directory=os.getcwd()
            dir_name="BICM_interleaver"
            dir_name=current_directory+"/"+dir_name
            
            try:
                os.makedirs(dir_name)
            except FileExistsError:
                pass
            
            filename="length{}_mod{}".format(self.N,int((self.M)**(1/2)//2))
    
            #if file exists, then load txt file
            filename=dir_name+"/"+filename
            
            try:
                self.BICM_int=np.loadtxt(filename,dtype='int')
            except FileNotFoundError:
                self.BICM_int=self.srandom_interleave()
                #export file
                np.savetxt(filename,self.BICM_int,fmt='%d')
            
            self.BICM_deint=np.argsort(self.BICM_int)
            
            #check
            a=np.arange(self.N)
            b=a[self.BICM_int]
            c=b[self.BICM_deint]
            if np.any(a!=c):
                print("BICM interleaver error!")
                
        #coding
        self.cd=turbo_construction.coding(self.N,self.K)
        self.ec=turbo_encode.encoding(self.cd)
        self.dc=turbo_decode.decoding(self.cd)
        #modulation
        self.modem=modulation.QAMModem(self.M)
        

        #channel
        self.ch=AWGN._AWGN()
        
        #filename
        self.filename="turbo_code_{}_{}_{}".format(self.N,self.K,self.M)
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
    
                            #i-k>=0 : for the part i<s 
                            #(abs(heap[i-k]-vector[j]))<=s : srandom interleaver
                            #vector[j]//mod!=i//mod : mod M interleaver(such as odd-even)
    
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
        EsNo = 10 ** (EsNodB / 10)
        No=1/EsNo

        info,cwd=self.ec.turbo_encode()
        if self.BICM==True:
            cwd=cwd[self.BICM_int]
        TX_conste=self.modem.modulate(cwd)
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,No)
        if self.BICM==True:
            Lc=Lc[self.BICM_deint]
        EST_info=self.dc.turbo_decode(Lc)
        
        return info,EST_info
'''
class Mysystem():
    def __init__(self,M,K):
        self.M=M
        self.K=K
        #self.N=self.K*int(np.log2(self.M))
        self.N=self.K*2
        self.BICM=False 
        
        if self.BICM==True:
            #make BICM directory
            # directory make
            current_directory="/home/kaneko/Dropbox/programming/geometric_shaping"
            #current_directory=os.getcwd()
            dir_name="BICM_interleaver"
            dir_name=current_directory+"/"+dir_name
            
            try:
                os.makedirs(dir_name)
            except FileExistsError:
                pass
            
            filename="length{}_mod{}".format(self.N,int((self.M)**(1/2)//2))
    
            #if file exists, then load txt file
            filename=dir_name+"/"+filename
            
            try:
                self.BICM_int=np.loadtxt(filename,dtype='int')
            except FileNotFoundError:
                self.BICM_int=self.srandom_interleave()
                #export file
                np.savetxt(filename,self.BICM_int,fmt='%d')
            
            self.BICM_deint=np.argsort(self.BICM_int)
            
            #check
            a=np.arange(self.N)
            b=a[self.BICM_int]
            c=b[self.BICM_deint]
            if np.any(a!=c):
                print("BICM interleaver error!")
                
        #coding
        self.cd=LDPC_construction.coding(self.N,self.K)
        self.ec=LDPC_encode.encoding(self.cd)
        self.dc=LDPC_decode.decoding(self.cd)
        #modulation
        self.modem=modulation.QAMModem(self.M)
        

        #channel
        self.ch=AWGN._AWGN()
        
        #filename
        self.filename="LDPC_code_{}_{}_{}".format(self.N,self.K,self.M)
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
    
                            #i-k>=0 : for the part i<s 
                            #(abs(heap[i-k]-vector[j]))<=s : srandom interleaver
                            #vector[j]//mod!=i//mod : mod M interleaver(such as odd-even)
    
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
        EsNo = 10 ** (EsNodB / 10)
        No=1/EsNo

        info,cwd=self.ec.LDPC_encode()
        if self.BICM==True:
            cwd=cwd[self.BICM_int]
        TX_conste=self.modem.modulate(cwd)
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        Lc=-1*self.modem.demodulate(RX_conste,No)
        if self.BICM==True:
            Lc=Lc[self.BICM_deint]
        EST_cwd=self.dc.LDPC_decode(Lc)
        return cwd,EST_cwd

if __name__=='__main__':
    K=512 #symbol数
    M=4
    EsNodB=-3.0
    system=Mysystem(M,K)
    print("\n")
    print(system.N,system.K)
    info,EST_info=system.main_func(EsNodB)
    print(np.sum(info!=EST_info))
    '''
    M_list=[4,16,256]
    EsNodB_list=np.arange(0,10,0.5)
    for M in M_list:
        for EsNodB in EsNodB_list:  
            if M==16:
                EsNodB+=5
            elif M==256:
                EsNodB+=10
            mysys=Mysystem(M,K)  
            const=monte_carlo_construction.monte_carlo()
            const.main_const(mysys.N,mysys.K,EsNodB,mysys.M)    
    '''