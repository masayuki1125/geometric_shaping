#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pickle
import sys
import multiprocessing
import os
import math
import matplotlib.pyplot as plt

# In[75]:


#polar code
def generate_information(N):
    #generate information
    info=np.random.randint(0,2,N)
    return info

def encode(u_message):
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

        codeword = np.concatenate([encode(u1u2), encode(u2)])
    return codeword

def reverse_bits(N):
    res=np.zeros(N,dtype=int)

    for i in range(N):
        tmp=format (i,'b')
        tmp=tmp.zfill(int(math.log2(N))+1)[:0:-1]
        #print(tmp) 
        res[i]=reverse(i,N)
    return res

def reverse(n,N):
    tmp=format (n,'b')
    tmp=tmp.zfill(int(math.log2(N))+1)[:0:-1]
    res=int(tmp,2) 
    return res

def polar_encode(N):
  
    bit_reversal_sequence=reverse_bits(N)
    info=generate_information(N)
    cwd=encode(info[bit_reversal_sequence])
    
    return info,cwd


# In[76]:


#polar code
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

def SC_decoding(N,Lc,info):
    #initialize constant
    itr_num=int(math.log2(N))    
    llr=np.zeros((itr_num+1,N))
    EST_codeword=np.zeros((itr_num+1,N))
    llr[0]=Lc

    #put decoding result into llr[logN]

    depth=0
    length=0
    before_process=0# 0:left 1:right 2:up 3:leaf

    while True:

        #left node operation
        if before_process!=2 and before_process!=3 and length%2**(itr_num-depth)==0:
            depth+=1
            before_process=0

            tmp1=llr[depth-1,length:length+2**(itr_num-depth)]
            tmp2=llr[depth-1,length+2**(itr_num-depth):length+2**(itr_num-depth+1)]

            llr[depth,length:length+N//(2**depth)]=chk(tmp1,tmp2)

        #right node operation 
        elif before_process!=1 and length%2**(itr_num-depth)==2**(itr_num-depth-1):
            
            #print(length%2**(self.itr_num-depth))
            #print(2**(self.itr_num-depth-1))
            
            depth+=1
            before_process=1
            
            tmp1=llr[depth-1,length-2**(itr_num-depth):length]
            tmp2=llr[depth-1,length:length+2**(itr_num-depth)]

            llr[depth,length:length+2**(itr_num-depth)]=tmp2+(1-2*EST_codeword[depth,length-2**(itr_num-depth):length])*tmp1

        #up node operation
        elif before_process!=0 and length!=0 and length%2**(itr_num-depth)==0:#今いるdepthより一個下のノードから、upすべきか判断する
        
            tmp1=EST_codeword[depth+1,length-2**(itr_num-depth):length-2**(itr_num-depth-1)]
            tmp2=EST_codeword[depth+1,length-2**(itr_num-depth-1):length]

            EST_codeword[depth,length-2**(itr_num-depth):length-2**(itr_num-depth-1)]=(tmp1+tmp2)%2
            EST_codeword[depth,length-2**(itr_num-depth-1):length]=tmp2

            depth-=1
            before_process=2
        
        else:
            print("error!")

        #leaf node operation
        if depth==itr_num:
        
            #for monte carlo construction
            EST_codeword[depth,length]=info[length]
            
            length+=1 #go to next length

            depth-=1 #back to depth
            before_process=3
            
            #print(llr[itr_num,length-1])
            #print(EST_codeword[itr_num,length-1])
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            
        
        if length==N:
            break
    
    #for monte calro construction  
    res=llr[itr_num]
    
    return res             

# In[78]:

#モンテカルロ法を用いるときに使用する関数
def add_AWGN(const,No):
    noise = np.random.normal(0, math.sqrt(No / 2), (len(const))) + 1j * np.random.normal(0, math.sqrt(No / 2), (len(const)))
    return const+noise

# In[86]:
class monte_carlo():
      
  def main_const(self,N,K,design_SNR,M,**kwargs):
    #check
    #design_SNR=100
    #get from kwargs
    
    const=Myconstruction(N,M,design_SNR,**kwargs)
    "comment if BICM or not"
    #if const.BICM==True:
    #    print("use BICM")
    #else:
    #    print("don't use BICM")
    
    if N!=const.N:
        print("monte_carlo codelength error!!")
    
    dumped=pickle.dumps(const)
    
    c=self.output(dumped)
    
    tmp=np.argsort(c)[::-1]
    frozen_bits=np.sort(tmp[:N-K])
    info_bits=np.sort(tmp[N-K:])
    
    return frozen_bits,info_bits

  @staticmethod
  def make_d(dumped):
    #initial constant
    const=pickle.loads(dumped)
    
    if const.N<=2048:
        epoch=10**4
    else:
        epoch=10**3
    
    c=np.zeros(const.N)
    for _ in range(epoch):
      info,llr=const.main_func()
      #print(llr)
      #from IPython.core.debugger import Pdb; Pdb().set_trace()
      
      d=np.zeros(len(llr))
        #print(llr)
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
      d[(2*info-1)*llr<0]=0
      d[(2*info-1)*llr>=0]=1
      
      c=c+d
    
    return c

  def make_c(self,dumped):
    const=pickle.loads(dumped)
    
    c=np.zeros(const.N)
    multi_num=100 #the number of multiprocessing 
    
    inputs=[]
    for _ in range(multi_num):
      inputs+=[dumped]
    
    #multiprocessing
    with multiprocessing.Pool(processes=multi_num) as pool:
      res = pool.map(self.make_d,inputs)
    
    for i in range(multi_num):
      c=c+res[i]
      
    return c
  
  def output(self,dumped):
    
    # directory make
    current_directory="/home/kaneko/Dropbox/programming/geometric_shaping/polar_code"
    #current_directory=os.getcwd()
    dir_name="monte_carlo_construction"
    dir_name=current_directory+"/"+dir_name
    
    try:
      os.makedirs(dir_name)
    except FileExistsError:
      pass
    
    const=pickle.loads(dumped)
      
    filename="{}QAM_{}_{}".format(const.M,const.N,const.design_SNR)
    
    if const.BICM==True:
        filename+="_BICM"
    
    #if file exists, then load txt file
    filename=dir_name+"/"+filename
    
    try:
      c=np.loadtxt(filename)
    except FileNotFoundError:
      print("make frozen bits!")
      print(filename)
      c=self.make_c(dumped)
      #export file
      np.savetxt(filename,c)

    return c

#make modulation
class Modem:
    def __init__(self, M, gray_map=True, bin_input=True, soft_decision=True, bin_output=True):

        N = np.log2(M)  # bits per symbol
        if N != np.round(N):
            raise ValueError("M should be 2**n, with n=1, 2, 3...")
        if soft_decision == True and bin_output == False:
            raise ValueError("Non-binary output is available only for hard decision")

        self.M = M  # modulation order
        self.N = int(N)  # bits per symbol
        self.m = [i for i in range(self.M)]
        self.gray_map = gray_map
        self.bin_input = bin_input
        self.soft_decision = soft_decision
        self.bin_output = bin_output

    ''' SERVING METHODS '''

    def __gray_encoding(self, dec_in):
        """ Encodes values by Gray encoding rule.
        Parameters
        ----------
        dec_in : list of ints
            Input sequence of decimals to be encoded by Gray.
        Returns
        -------
        gray_out: list of ints
            Output encoded by Gray sequence.
        """

        bin_seq = [np.binary_repr(d, width=self.N) for d in dec_in]
        gray_out = []
        for bin_i in bin_seq:
            gray_vals = [str(int(bin_i[idx]) ^ int(bin_i[idx - 1]))
                         if idx != 0 else bin_i[0]
                         for idx in range(0, len(bin_i))]
            gray_i = "".join(gray_vals)
            gray_out.append(int(gray_i, 2))
        return gray_out

    def create_constellation(self, m, s):
        """ Creates signal constellation.
        Parameters
        ----------
        m : list of ints
            Possible decimal values of the signal constellation (0 ... M-1).
        s : list of complex values
            Possible coordinates of the signal constellation.
        Returns
        -------
        dict_out: dict
            Output dictionary where
            key is the bit sequence or decimal value and
            value is the complex coordinate.
        """

        if self.bin_input == False and self.gray_map == False:
            dict_out = {k: v for k, v in zip(m, s)}
        elif self.bin_input == False and self.gray_map == True:
            mg = self.__gray_encoding(m)
            dict_out = {k: v for k, v in zip(mg, s)}
        elif self.bin_input == True and self.gray_map == False:
            mb = self.de2bin(m)
            dict_out = {k: v for k, v in zip(mb, s)}
        elif self.bin_input == True and self.gray_map == True:
            mg = self.__gray_encoding(m)
            mgb = self.de2bin(mg)
            dict_out = {k: v for k, v in zip(mgb, s)}
        return dict_out

    def llr_preparation(self):
        """ Creates the coordinates
        where either zeros or ones can be placed in the signal constellation..
        Returns
        -------
        zeros : list of lists of complex values
            The coordinates where zeros can be placed in the signal constellation.
        ones : list of lists of complex values
            The coordinates where ones can be placed in the signal constellation.
        """
        code_book = self.code_book

        zeros = [[] for i in range(self.N)]
        ones = [[] for i in range(self.N)]

        bin_seq = self.de2bin(self.m)

        for bin_idx, bin_symb in enumerate(bin_seq):
            if self.bin_input == True:
                key = bin_symb
            else:
                key = bin_idx
            for possition, digit in enumerate(bin_symb):
                if digit == '0':
                    zeros[possition].append(code_book[key])
                else:
                    ones[possition].append(code_book[key])
        return zeros, ones

    ''' DEMODULATION ALGORITHMS '''
    @staticmethod
    def calc_exp(x,No):
        #クリップする
        res=np.exp(-1*np.array(x)/No)
        res=np.sum(res,axis=0,keepdims=True)
        res=np.clip(res,10**(-15),10**15)
        res=np.log(res)
        return res
    

    def __ApproxLLR(self, x, No):
        """ Calculates approximate Log-likelihood Ratios (LLRs) [1].
        Parameters
        ----------
        x : 1-D ndarray of complex values
            Received complex-valued symbols to be demodulated.
        No: float
            Additive noise variance.
        (additional)
        exact=False/True
            using max/min approximation or not
        Returns
        -------
        
        result: 1-D ndarray of floats
            Output LLRs.
        Reference:
            [1] Viterbi, A. J., "An Intuitive Justification and a
                Simplified Implementation of the MAP Decoder for Convolutional Codes,"
                IEEE Journal on Selected Areas in Communications,
                vol. 16, No. 2, pp 260–264, Feb. 1998
        """
        exact=True

        zeros = self.zeros
        ones = self.ones
        
        LLR = []
        for (zero_i, one_i) in zip(zeros, ones): #iビット目のビットが0のときの信号点と1のときの信号点のリスト
            num = [((np.real(x) - np.real(z)) ** 2)
                   + ((np.imag(x) - np.imag(z)) ** 2)
                   for z in zero_i]
            denum = [((np.real(x) - np.real(o)) ** 2)
                     + ((np.imag(x) - np.imag(o)) ** 2)
                     for o in one_i]
            
            #print(len(zero_i))

            if exact==False:
                num_post = np.amin(num, axis=0, keepdims=True)
                denum_post = np.amin(denum, axis=0, keepdims=True)
                llr = np.transpose(num_post[0]) - np.transpose(denum_post[0]) #二次元配列になってしまっているので、1次元に直す
                LLR.append(-llr / No)
            elif exact==True:
                num_post = self.calc_exp(num,No)
                denum_post = self.calc_exp(denum,No)
                llr = np.transpose(num_post[0]) - np.transpose(denum_post[0]) #二次元配列になってしまっているので、1次元に直す
                LLR.append(llr)

            

        result = np.zeros((len(x) * len(zeros)))
        for i, llr in enumerate(LLR):
            result[i::len(zeros)] = llr
        return result

    ''' METHODS TO EXECUTE '''

    def modulate(self, msg):
        """ Modulates binary or decimal stream.
        Parameters
        ----------
        x : 1-D ndarray of ints
            Decimal or binary stream to be modulated.
        Returns
        -------
        modulated : 1-D array of complex values
            Modulated symbols (signal envelope).
        """

        if (self.bin_input == True) and ((len(msg) % self.N) != 0):
            raise ValueError("The length of the binary input should be a multiple of log2(M)")

        if (self.bin_input == True) and ((max(msg) > 1.) or (min(msg) < 0.)):
            raise ValueError("The input values should be 0s or 1s only!")
        if (self.bin_input == False) and ((max(msg) > (self.M - 1)) or (min(msg) < 0.)):
            raise ValueError("The input values should be in following range: [0, ... M-1]!")

        if self.bin_input:
            msg = [str(bit) for bit in msg]
            splited = ["".join(msg[i:i + self.N])
                       for i in range(0, len(msg), self.N)]  # subsequences of bits
            modulated = [self.code_book[s] for s in splited]
        else:
            modulated = [self.code_book[dec] for dec in msg]
        return np.array(modulated)

    def demodulate(self, x, No=1.):
        """ Demodulates complex symbols.
         Yes, MathWorks company provides several algorithms to demodulate
         BPSK, QPSK, 8-PSK and other M-PSK modulations in hard output manner:
         https://www.mathworks.com/help/comm/ref/mpskdemodulatorbaseband.html
         However, to reduce the number of implemented schemes the following way is used in our project:
            - calculate LLRs (soft decision)
            - map LLR to bits according to the sign of LLR (inverse of NRZ)
         We guess the complexity issues are not the critical part due to hard output demodulators are not so popular.
         This phenomenon depends on channel decoders properties:
         e.g., Convolutional codes, Turbo convolutional codes and LDPC codes work better with LLR.
        Parameters
        ----------
        x : 1-D ndarray of complex symbols
            Decimal or binary stream to be demodulated.
        No: float
            Additive noise variance.
        Returns
        -------
        result : 1-D array floats
            Demodulated message (LLRs or binary sequence).
        """

        if self.soft_decision:
            result = self.__ApproxLLR(x, No)
        else:
            if self.bin_output:
                llr = self.__ApproxLLR(x, No)
                result = (np.sign(-llr) + 1) / 2  # NRZ-to-bin
            else:
                llr = self.__ApproxLLR(x, No)
                result = self.bin2de((np.sign(-llr) + 1) / 2)
        return result

class QAMModem(Modem):
    def __init__(self, M, gray_map=True, bin_input=True, soft_decision=True, bin_output=True):
        super().__init__(M, gray_map, bin_input, soft_decision, bin_output)

        if np.sqrt(M) != np.fix(np.sqrt(M)) or np.log2(np.sqrt(M)) != np.fix(np.log2(np.sqrt(M))):
            raise ValueError('M must be a square of a power of 2')

        self.m = [i for i in range(self.M)]
        self.s = self.__qam_symbols()
        self.code_book = self.create_constellation(self.m, self.s)

        if self.gray_map:
            self.__gray_qam_arange()

        self.zeros, self.ones = self.llr_preparation()

    def __qam_symbols(self):
        """ Creates M-QAM complex symbols."""

        c = np.sqrt(self.M)
        b = -2 * (np.array(self.m) % c) + c - 1
        a = 2 * np.floor(np.array(self.m) / c) - c + 1
        s = list((a + 1j * b))
        
        ave=np.average(np.abs(s)**2)
        s/=ave**(1/2)
        
        return s

    def __gray_qam_arange(self):
        """ This method re-arranges complex coordinates according to Gray coding requirements.
        To implement correct Gray mapping the additional heuristic is used:
        the even "columns" in the signal constellation is complex conjugated.
        """

        for idx, (key, item) in enumerate(self.code_book.items()):
            if (np.floor(idx / np.sqrt(self.M)) % 2) != 0:
                self.code_book[key] = np.conj(item)

    def de2bin(self, decs):
        """ Converts values from decimal to binary representation.
        Parameters
        ----------
        decs : list of ints
            Input decimal values.
        Returns
        -------
        bin_out : list of ints
            Output binary sequences.
        """
        bin_out = [np.binary_repr(d, width=self.N) for d in decs]
        return bin_out

    def bin2de(self, bin_in):
        """ Converts values from binary to decimal representation.
        Parameters
        ----------
        bin_in : list of ints
            Input binary values.
        Returns
        -------
        dec_out : list of ints
            Output decimal values.
        """

        dec_out = []
        N = self.N  # bits per modulation symbol (local variables are tiny bit faster)
        Ndecs = int(len(bin_in) / N)  # length of the decimal output
        for i in range(Ndecs):
            bin_seq = bin_in[i * N:i * N + N]  # binary equivalent of the one decimal value
            str_o = "".join([str(int(b)) for b in bin_seq])  # binary sequence to string
            dec_out.append(int(str_o, 2))
        return dec_out

    def plot_const(self):
        """ Plots signal constellation """

        if self.M <= 16:
            limits = np.log2(self.M)
            size = 'small'
        elif self.M == 64:
            limits = 1.5 * np.log2(self.M)
            size = 'x-small'
        else:
            limits = 2.25 * np.log2(self.M)
            size = 'xx-small'

        const = self.code_book
        fig = plt.figure(figsize=(6, 4), dpi=150)
        for i in list(const):
            x = np.real(const[i])
            y = np.imag(const[i])
            plt.plot(x, y, 'o', color='red')
            if x < 0:
                h = 'right'
                xadd = -.05
            else:
                h = 'left'
                xadd = .05
            if y < 0:
                v = 'top'
                yadd = -.05
            else:
                v = 'bottom'
                yadd = .05
            if abs(x) < 1e-9 and abs(y) > 1e-9:
                h = 'center'
            elif abs(x) > 1e-9 and abs(y) < 1e-9:
                v = 'center'
            #plt.annotate(i, (x + xadd, y + yadd), ha=h, va=v, size=size)
        M = str(self.M)
        if self.gray_map:
            mapping = 'Gray'
        else:
            mapping = 'Binary'

        if self.bin_input:
            inputs = 'Binary'
        else:
            inputs = 'Decimal'

        plt.grid()
        plt.axvline(linewidth=1.0, color='black')
        plt.axhline(linewidth=1.0, color='black')
        #plt.axis([-limits, limits, -limits, limits])
        plt.title(M + '-QAM, Mapping: ' + mapping + ', Input: ' + inputs)
        plt.show()

class Myconstruction:
    def __init__(self,N,M,design_SNR,**kwargs):
        
        if kwargs.get('BICM_int') is not None:
            self.BICM_int=kwargs.get("BICM_int")
            self.BICM=True
            self.BICM_deint=np.argsort(self.BICM_int)
            self.BICM=True
        else:
            self.BICM=False
        
        self.M=M
        self.N=N
        self.design_SNR=design_SNR
            
        #modulation
        self.modem=QAMModem(self.M)

    def main_func(self):
        #adaptive dicision of frozen bits
        EsNo = 10 ** (self.design_SNR / 10)
        No=1/EsNo
        
        info,cwd=polar_encode(self.N)
        if self.BICM==True:
            cwd=cwd[self.BICM_int]
        TX_conste=self.modem.modulate(cwd)
        RX_conste=add_AWGN(TX_conste,No)
        Lc=self.modem.demodulate(RX_conste,No)
        if self.BICM==True:
            Lc=Lc[self.BICM_deint]
        llr=SC_decoding(self.N,Lc,info)
        
        return info,llr
  

#%%
if __name__=="__main__":

    #initial constant
    M=4
    K=8
    N=int(np.log2(M))*K
    EsNodB=0
    const=Myconstruction(N,M,EsNodB)
    N=const.N
    epoch=10**1
    c=np.zeros(const.N)
    for _ in range(epoch):
        info,llr=const.main_func()
 
        d=np.zeros(len(llr))
        #print(llr)
        #np.savetxt("llr",llr)
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        d[(2*info-1)*llr<0]=0
        d[(2*info-1)*llr>=0]=1
        #print(c)
        c=c+d
    
    print(c)
    
    const=monte_carlo()
    print(const.main_const(N,K,EsNodB,M))
    
    