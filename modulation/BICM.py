import numpy as np
import math
import os

def srandom_interleave(N,M):
        
    mod=(M)**(1/2)//2
    s=math.floor(math.sqrt(N))-5
    print(s)
    #step 1 generate random sequence
    vector=np.arange(N,dtype='int')
    np.random.shuffle(vector)

    itr=True
    count=0
    while itr:
        #intialize for each iteration
        heap=np.zeros(N,dtype='int')
        position=np.arange(N,dtype='int')

        #step2 set first vector to heap
        heap[0]=vector[0]
        position=np.delete(position,0)

        #step3 bubble sort 
        #set to ith heap
        for i in range(1,N):
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
                vector[0:N-i]=tmp
                vector[N-i:N]=heap[0:i]
                break

        #if all the heap num is valid, end iteration
        else:
            itr=False
        
        #print(heap)
        #print(vector)
        print("\r","itr",count,end="")
        count+=1
    
    return heap

def make_BICM(N,M):
    
    #make BICM directory
    # directory make
    current_directory="/home/kaneko/Dropbox/programming/geometric_shaping/modulation"
    #current_directory=os.getcwd()
    dir_name="BICM_interleaver"
    dir_name=current_directory+"/"+dir_name
    
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        pass
    
    filename="length{}_mod{}".format(N,int((M)**(1/2)//2))

    #if file exists, then load txt file
    filename=dir_name+"/"+filename
    
    try:
        BICM_int=np.loadtxt(filename,dtype='int')
    except FileNotFoundError:
        BICM_int=srandom_interleave(N,M)
        #export file
        np.savetxt(filename,BICM_int,fmt='%d')
    
    BICM_deint=np.argsort(BICM_int)
    
    #check
    a=np.arange(N)
    b=a[BICM_int]
    c=b[BICM_deint]
    if np.any(a!=c):
        print("BICM interleaver error!")
    
    
    return BICM_int,BICM_deint

class BICM_ID:
    def __init__(self,modem,zeros=0,ones=0):
        self.modem=modem
        self.zeros=zeros
        self.ones=ones
        self.zeros_key,self.ones_key=self.key_preparation(self.modem)
        self.mat=self.create_mat(self.modem)
        print("mat shape")
        print(self.mat)
    
    @staticmethod
    def key_preparation(modem):
        """ Creates the coordinates
        where either zeros or ones can be placed in the signal constellation..
        Returns
        -------
        zeros : list of lists of complex values
            The coordinates where zeros can be placed in the signal constellation.
        ones : list of lists of complex values
            The coordinates where ones can be placed in the signal constellation.
        """

        zeros = [[] for i in range(modem.N)]
        ones = [[] for i in range(modem.N)]

        bin_seq = modem.de2bin(modem.m)

        for bin_idx, bin_symb in enumerate(bin_seq):
            if modem.bin_input == True:
                key = bin_symb
            else:
                key = bin_idx
            for possition, digit in enumerate(bin_symb):
                if digit == '0':
                    zeros[possition].append(key)
                else:
                    ones[possition].append(key)
        
        #from str list to int array 
        for i in range(len(zeros)):
            zeros[i]=np.array([int(s, 2) for s in zeros[i]])
            ones[i]=np.array([int(s, 2) for s in ones[i]])
                    
        return zeros, ones
    
    @staticmethod
    def create_mat(modem):
        '''
        return 2D numpy array
        in which the first column is 0(bin)
        second column is1(bin),,,, 
        '''
        #print(str(0)+str(modem.N)+'b')
        #print(format(4,str(0)+str(modem.N)+'b'))
        mat=np.zeros((modem.M,modem.N),dtype=int)
        for i in range(modem.M):
            digit_num=str(0)+str(modem.N)
            tmp=format(i,digit_num+'b')
            for j in range(modem.N):
                mat[i,j]=int(tmp[j])
        return mat

    @staticmethod
    def max_str(a,b):
        THRESHOLD=30
        if abs(a-b)>THRESHOLD:
            #print("max")
            res=max(a,b)
        else:
            res=max(a,b)+np.log(1+np.exp(-1*abs(a-b)))
        return res
    
    def max_str_array(self,x):
        for i in range(1,len(x)):
            x[0]=self.max_str(x[0],x[i])
        tmp=np.max(x)
        if x[0]!=tmp:
            print("err")
        return x[0]
    
    def demapper(self,Pre_info,Lc,No):
        '''
        inputs:
        ----
        return:
        updated LLR from demapper
        '''

        symbol_num=int(len(Lc)/self.modem.N) #シンボルの長さ

        #print(Pre_info[:30])
        #Pre_info=Pre_info.reshape([modem.N,symbol_num],order='F') #各シンボルで受信したビットごとに並べ替える　(symbol_num*bits_in_symbol)

        #print(Pre_info.reshape([self.modem.N,symbol_num],order='F').shape)
        #Pre_info-=Lc
        Pre_info_mat=self.mat@(Pre_info.reshape([self.modem.N,symbol_num],order='F'))#-Lc.reshape([self.modem.N,symbol_num],order='F'))
        
        #print(self.modem.N)
        #print(Pre_info_mat.shape)
        ex_mat_z=np.zeros(self.zeros.shape) #the matrix of symbol generate probability of the bit zero
        ex_mat_o=np.zeros(self.zeros.shape) #bit ones
        
        #print(self.zeros.shape)

        for i in range(symbol_num): #1シンボル毎
            for j in range(self.modem.N): #シンボル内の1ビットごと
                ex_mat_z[:,i,j]-=Pre_info_mat[self.zeros_key[j],i]
                ex_mat_o[:,i,j]-=Pre_info_mat[self.ones_key[j],i]
                

        num=-1*np.array(self.zeros)/No+ex_mat_z#-ex_mat_o
        #num=np.clip(num,-30,30)
        #num=np.exp(num)
        for i in range(num.shape[1]):#symbol
            for j in range(num.shape[2]):#bit in the same symbol
                num[0,i,j]=self.max_str_array(num[:,i,j])
        #num=np.sum(num,axis=0,keepdims=True)
        #num=np.clip(num,10**(-15),10**15)
        #num=np.log(num)

        den=-1*np.array(self.ones)/No+ex_mat_o
        #den=np.clip(den,-30,30)
        #den=np.exp(den)
        for i in range(den.shape[1]):#symbol
            for j in range(den.shape[2]):#bit in the same symbol
                den[0,i,j]=self.max_str_array(den[:,i,j])
        
        #den=np.sum(den,axis=0,keepdims=True)
        #den=np.clip(den,10**(-15),10**15)
        #den=np.log(den)

        res_Lc=(np.transpose(num[0]) - np.transpose(den[0])).ravel(order='F')  
        
        return res_Lc-Pre_info

if __name__=='__main__':
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
    #from main import Mysystem_LDPC
    from LDPC_code import LDPC_construction
    from LDPC_code import LDPC_encode
    from LDPC_code import LDPC_decode
    import modulation
    from channel import AWGN
    
    class Mysystem_LDPC():
        def __init__(self,M,K):
            self.M=M
            self.K=K
            #self.N=self.K*int(np.log2(self.M))
            self.N=self.K*2
            self.BICM=True 
            self.BICM_ID=True
            
            if self.BICM_ID==True:
                self.BICM=True
                self.BICM_ID_itr=10
                    
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
                self.BICM_int,self.BICM_deint=make_BICM(self.N,self.M)
                self.filename=self.filename+"_BICM"
            
            #output filename to confirm which program I run
            print(self.filename)
    
        def main_func(self,EsNodB):
            EsNo = 10 ** (EsNodB / 10)
            No=1/EsNo

            info,cwd=self.ec.LDPC_encode()
            info=cwd #BICMのとき、cwdがインターリーブされてしまい、比較できなくなる為、infoをcwdに変更する
            if self.BICM==True:
                cwd=cwd[self.BICM_int]
            TX_conste=self.modem.modulate(cwd)
            
            #channel
            RX_conste=self.ch.add_AWGN(TX_conste,No)
            
            #at the reciever
            if self.BICM_ID==False:
                Lc=self.modem.demodulate(RX_conste,No)
                if self.BICM==True:
                    Lc=Lc[self.BICM_deint]
                EST_cwd,_=self.dc.LDPC_decode(Lc)
                return info,EST_cwd
            
            
            elif self.BICM_ID==True:
                
                #demodulate      
                Lc,[zeros,ones]=self.modem.demodulate(RX_conste,No,self.BICM_ID)
                
                ###check
                num=self.modem.calc_exp(zeros,No)
                denum=self.modem.calc_exp(ones,No)
                Lc_check=(np.transpose(num[0]) - np.transpose(denum[0])).ravel(order='F')
                #print(Lc3)
                if np.any(Lc!=Lc_check):
                    print("Lc is different")
                ###check end
                
                return Lc,[zeros,ones]
    
    K=512
    M=256
    EsNodB=20
    EsNo = 10 ** (EsNodB / 10)
    No=1/EsNo
    mysys=Mysystem_LDPC(M,K)
    
    modem=mysys.modem
    Lc,[zeros,ones]=mysys.main_func(EsNodB) #Lcはデインターリーブされていない
    #decode 
    EST_cwd,EX_info=mysys.dc.LDPC_decode(Lc[mysys.BICM_deint]) #MAPデコーダで出てきた外部値を取得

    Pre_info=EX_info[mysys.BICM_int]+Lc#順番の入れ替えをして、事前値にする
    #print(Pre_info.shape)
    #print(Lc.shape)

    dmp=BICM_ID(modem,zeros,ones)
    print(dmp.demapper(Pre_info,Lc,No)[0:30],"\n")
    print(Lc[0:30])