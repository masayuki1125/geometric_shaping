from ctypes import LibraryLoader
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
from polar_code import iGA
from polar_code import monte_carlo_construction
from turbo_code import turbo_construction
from turbo_code import turbo_encode
from turbo_code import turbo_decode
from modulation import modulation
from modulation.BICM import make_BICM
from modulation.BICM import BICM_ID
from channel import AWGN
class Mysystem:
    def __init__(self,M,K):
        #make instance
        self.M=M
        self.K=K
        #self.N=self.K*int(np.log2(self.M))
        self.N=self.K*2
        const_var=3 #1:MC 2:iGA 3:RCA
        
        self.type=1#1:separated scheme 2:Block intlv(No intlv in arikan polar decoder) 3:No intlv(Block intlv in arikan polar decoder) 4:rand intlv
        
        #for construction
        if const_var==1:
            self.const=monte_carlo_construction.monte_carlo()
            self.const_name="_MC"
        elif const_var==2:
            self.const=iGA.Improved_GA()
            self.const_name="_iGA"
        elif const_var==3:
            self.const=RCA.RCA()
            self.const_name="_RCA"
        
        #ENCの数
        self.enc_num=int(np.log2(M**(1/2)))
    
        if self.N%self.enc_num!=0:
            print("encoder is not mod(enc_num)")
        self.N_sep=self.N//self.enc_num
        self.K_sep=self.K//self.enc_num #とりあえず適当なKで初期化する
        
        
       #coding
        self.cd=[]
        self.ec=[]
        self.dc=[]
        for i in range(self.enc_num):
            self.cd+=[polar_construction.coding(self.N_sep,self.K_sep)]
            
            #CRCを調整する
            if self.enc_num==2:
                self.cd[i].CRC_polynomial =np.array([1,0,0,1,1])
            elif self.enc_num==4:
                self.cd[i].CRC_polynomial =np.array([1,0,1,0,0,1,1,0,1])
            else:
                print("unsupported encoder number")
            
            self.ec+=[polar_encode.encoding(self.cd[i])]
            self.dc+=[polar_decode.decoding(self.cd[i],self.ec[i])]
            
        #get decoder var
        self.decoder_ver=self.cd[0].decoder_ver
        #modulation
        self.modem=modulation.QAMModem(self.M)
        #channel
        self.ch=AWGN._AWGN()
        
        #interleaver design
        self.BICM_int,self.BICM_deint=self.make_BICM_int(self.N,self.M,self.type)
        #np.savetxt("BICM_int",self.BICM_int,fmt="%.0f")
        
        self.filename=self.make_filename()
    
    def make_filename(self):
        #filename
        filename="polar_{}_{}_{}QAM".format(self.N,self.K,self.M)
        filename=filename+self.const_name
        if self.cd[0].systematic_polar==True:
            filename="systematic_"+filename
            
        #decoder type
        if self.cd[0].decoder_ver==0:
            filename+="_SC"
        elif self.cd[0].decoder_ver==2:
            filename+="_CA_SCL"
                    
        #provisional
        filename+="_type{}".format(self.type)
        
        #output filename to confirm which program I run
        print(filename)
        
        return filename
    
    def reverse_bits(self,N):
        res=np.zeros(N,dtype=int)

        for i in range(N):
            tmp=format (i,'b')
            tmp=tmp.zfill(int(np.log2(N))+1)[:0:-1]
            #print(tmp) 
            res[i]=self.reverse(i,N)
        return res

    @staticmethod
    def reverse(n,N):
        tmp=format (n,'b')
        tmp=tmp.zfill(int(np.log2(N))+1)[:0:-1]
        res=int(tmp,2) 
        return res
        
    def make_BICM_int(self,N,M,type):
        
        BICM_int=np.arange(N,dtype=int)
        #modify BICM int from simplified to arikan decoder order
        
        if type==1:#1:separated scheme 
            #type3と同じで、ブロックインターリーブする
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            BICM_int=np.ravel(BICM_int,order='F')
            pass #specific file is needed
        elif type==2:#2:No intlv in arikan polar decoder
            print("err type2")
            
        elif type==3:#3:Block intlv in arikan polar decoder
            print("err type3")
            #BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            #BICM_int=np.ravel(BICM_int,order='F')
        elif type==4:#4:rand intlv
            print("err type4")
            #bit reversal order
            #bit_reversal_sequence=self.cd.bit_reversal_sequence
            #BICM_int=BICM_int[bit_reversal_sequence]
            
            #tmp,_=make_BICM(N)
            #BICM_int=BICM_int[tmp]
        elif type==5:#2:No intlv +rand intlv for each channel
            print("err type5")
            #bit reversal order
            #bit_reversal_sequence=self.cd.bit_reversal_sequence
            #BICM_int=BICM_int[bit_reversal_sequence]
            
            #tmp,_=make_BICM(N//int(np.log2(M**(1/2))))
            #BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            #for i in range (int(np.log2(M**(1/2)))):
            #    BICM_int[i]=BICM_int[i][tmp]
            #BICM_int=np.ravel(BICM_int,order='C')
        elif type==6:#凍結ビットを低SNRに設定する
            print("err type6")
            #self.adaptive_intlv=True
            #ass#specific file is needed
        elif type==7:#compound polar codes
            #use block interleaver
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            BICM_int=np.ravel(BICM_int,order='F')
            
        else:
            print("interleaver type error")
        BICM_deint=np.argsort(BICM_int)
        #np.savetxt("deint",BICM_deint,fmt='%.0f')
        #print(BICM_int)
        #print(BICM_deint) 
        return BICM_int,BICM_deint
    
    def make_key(self,EST_cwd,pre_dec_num,key_num):
        '''
        推定符号語を用いることにより、より良いLLRを受信シンボルから導出する
        key_num:シンボル内の何ビット目のビットについてかの情報
        pre_dec_num:今まで何回復号したのかの情報
        EST_cwd:今まで復号し終わった推定符号語
        '''
        
        #code_book = self.modem.code_book
        symb_len=self.N//int(np.log2(self.M))

        zeros = np.zeros((self.M//2,symb_len)) #zero_iと同じ行列縦軸M/2、横軸シンボル長
        ones = np.zeros(zeros.shape)

        bin_seq = self.modem.de2bin(self.modem.m)
        
        #1シンボルあたり、決まっているビットが2ビット隣同士で存在する
        
        for symb_idx in range(symb_len):
            count_zero=0
            count_one=0
            zero_symb_idx=0
            one_symb_idx=0
            
            for bin_symb in bin_seq:    
                #print(bin_symb)
                #print(bin_symb[dec_num+1])
                #今復号したいインデックスのビットが0か1か確認する
                if bin_symb[key_num] == '0':
                    count_zero+=1
                    #EST_cwdと同じビットになっているもののみカウント
                    if bin_symb[pre_dec_num] == str(int(EST_cwd[2*symb_idx]))\
                        and bin_symb[pre_dec_num+2] == str(int(EST_cwd[2*symb_idx+1])):
                        zeros[zero_symb_idx,count_zero]=1
                        
                elif bin_symb[key_num] == '1':
                    count_one+=1
                    #EST_cwdと同じビットになっているもののみカウント
                    if bin_symb[pre_dec_num] == str(int(EST_cwd[2*symb_idx]))\
                        and bin_symb[pre_dec_num+2] == str(int(EST_cwd[2*symb_idx+1])):
                        ones[one_symb_idx,count_one]=1
                else:
                    print("program error")
            
        #print(zeros)
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        
        return zeros,ones
    
    def determine_keys(self,EST_info,pre_dec_num,key_num):
        #再エンコードする
        u_massage=self.ec[pre_dec_num].generate_U(EST_info)
        EST_cwd=self.ec[pre_dec_num].encode(u_massage)
        
        #推定符号語から、Keyの行列を生成する
        zero_key,one_key=self.make_key(EST_cwd,pre_dec_num,key_num)
        
        return zero_key,one_key
           
    def main_func(self,EsNodB):
        #すべてのデコーダ一律で凍結ビットを得る
        if self.decoder_ver==2:
            self.CRC_len=len(self.cd[0].CRC_polynomial)-1  
            frozen_bits,info_bits=self.const.main_const_sep(self.N,self.K+self.CRC_len,EsNodB,self.M,BICM_int=self.BICM_int)
        else:
            frozen_bits,info_bits=self.const.main_const_sep(self.N,self.K,EsNodB,self.M,BICM_int=self.BICM_int)
            
        #check
        for i in range(self.N):
            if (np.any(i==frozen_bits) or np.any(i==info_bits))==False:
                raise ValueError("The frozen set or info set is overlapped")
        
        #凍結ビットによって、情報ビットの長さを変える
        for i in range(self.enc_num):
            a=frozen_bits>=i*self.N_sep
            #print(a)
            b=frozen_bits<(i+1)*self.N_sep
            #print(b)
            c=a*b
            #print(c)
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            
            d=info_bits>=i*self.N_sep
            e=info_bits<(i+1)*self.N_sep
            f=d*e
            
            frozen_bits_sep=frozen_bits[c]%self.N_sep
            info_bits_sep=info_bits[f]%self.N_sep
            
            #print("len",len(info_bits_sep))
            
            #frozen and info chech
            if (len(frozen_bits_sep)+len(info_bits_sep))!=self.N_sep:
                print("frozen_bit and info_bit len err")
            
            for j in range(self.N_sep):
                if np.any(frozen_bits_sep==i):
                    break
                elif np.any(info_bits_sep==i):
                    break
                else:
                    print(j)
                    print("frozen or info is missed")
                    print(frozen_bits_sep)
                    print(info_bits_sep)
                    from IPython.core.debugger import Pdb; Pdb().set_trace()
                    
            self.cd[i].design_SNR=EsNodB    
            self.cd[i].frozen_bits=frozen_bits_sep
            self.ec[i].frozen_bits=frozen_bits_sep
            self.dc[i].frozen_bits=frozen_bits_sep
            self.cd[i].info_bits=info_bits_sep
            self.ec[i].info_bits=info_bits_sep
            self.dc[i].info_bits=info_bits_sep
            
            if self.decoder_ver==2:
                res=len(info_bits_sep)-self.CRC_len
            else:
                res=len(info_bits_sep)
            
            self.cd[i].K=res
            self.ec[i].K=res
            self.dc[i].K=res
            
            #print(res)
            
            #print(len(info_bits_sep)/self.N_sep)
            
        #for iGA and RCA and monte_carlo construction
                
        EsNo = 10 ** (EsNodB / 10)
        No=1/EsNo

        info=np.empty(0,dtype=int)
        cwd=np.empty(0,dtype=int)
        
        #info_use=np.empty(0,dtype=int)
        for i in range(self.enc_num):
            info_sep,cwd_sep=self.ec[i].polar_encode()
            #print(len(info_sep))
            info=np.concatenate([info,info_sep])
            cwd=np.concatenate([cwd,cwd_sep])
        
        if self.type==7:
            cwd=np.reshape(cwd,[int(np.log2(M**(1/2))),-1],order='C')
            cwd=(cwd.T@np.array([[1,0],[1,1]]))%2
            cwd=np.ravel(cwd,order='F')
        else:
            pass
                
        cwd=cwd[self.BICM_int] #interleave
        TX_conste=self.modem.modulate(cwd)
        #print(TX_conste)
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        if self.type==7:
            tmp=True
            Lc,mat_list=self.modem.demodulate(RX_conste,No,tmp)
        else:
            Lc=self.modem.demodulate(RX_conste,No)
        tmp_Lc=Lc
            
        Lc=Lc[self.BICM_deint] #de interleave
        
        if np.all([tmp_Lc]==Lc[self.BICM_int]):
            print("Lc ok")
        
        EST_info=np.empty(0)
        for i in range(self.enc_num):
            if self.type==7:
                #compound polar codes
                if i==0:
                    llr=self.dc[0].chk(Lc[i*self.N_sep:(i+1)*self.N_sep],Lc[(i+1)*self.N_sep:(i+2)*self.N_sep])
                elif i==1:
                    #今まで復号したEST_infoから、次のEST_infoを復号する
                    
                    #euclid distanceの三次元行列から、使う分のnumとdenumを用意する
                    num=mat_list[0]
                    denum=mat_list[1]
                    print(num.shape)
                    
                    zero_key=np.zeros(num.shape)
                    one_key=np.zeros(zero_key.shape)
                    for j in [1,3]: #同一シンボル内で後半の2ビットを復号したいので、1番目と3番めのインデックスを復号する
                        zero_key_j,one_key_j=self.determine_keys(EST_info,i-1,j) #zeros_key,ones_keyはx軸が1シンボル内で取りうるビット列のインデックス、y軸がシンボル長の二次元配列
                        zero_key[:,:,j]=zero_key_j
                        one_key[:,:,j]=one_key_j
                                     
                    #num=np.stack([mat_list[0][:,:,[0,1]],mat_list[0][:,:,[2,3]]],axis=1)
                    #denum=np.concatenate([mat_list[1][:,:,[0,1]],mat_list[1][:,:,[2,3]]],axis=1)
                    print(num.shape)
                    
                    ##check
                    '''
                    num_post = self.modem.calc_exp(num,No)
                    denum_post = self.modem.calc_exp(denum,No)
                    print("numpostshape",num_post.shape)
                    llr = np.transpose(num_post[0]) - np.transpose(denum_post[0])
                    print("shape",llr.shape)
                    print("llr",llr)
                    np.savetxt("Lc",Lc[self.BICM_int])
                    print("Lc",Lc[self.BICM_int])
                    result=np.ravel(llr,order='F')
                    
                    if np.any(result[self.BICM_deint]!=Lc):
                        print("llr error!")
                        print(result)
                        print(Lc[self.BICM_int])
                    else:
                        print("llr is correct")
                    '''
                    ##check end
                    num_post=self.modem.calc_exp(num*zero_key,No)
                    denum_post=self.modem.calc_exp(denum*one_key,No)
                    
                    #二次元配列のLLRを作成
                    llr = np.transpose(num_post[0]) - np.transpose(denum_post[0])
                    #1次元のベクトルに変換
                    llr=np.ravel(llr,order='F')
                    
                    #使うビットだけ取り出す
                    llr=llr[self.BICM_deint]
                    
                    if np.any[:len(llr)//2]!=0:
                        print("llr error!")
            
                    llr=llr[len(llr)//2:]
                    
                    print(llr)
                    
            else:
                #separated polar codes
                llr=Lc[i*self.N_sep:(i+1)*self.N_sep]
                pass  
            
            EST_info_sep=self.dc[i].polar_decode(llr)
            
            EST_info=np.concatenate([EST_info,EST_info_sep])
            
            #if i==0:
                #EST_info=self.dc[i].polar_decode(Lc[i*self.N_sep:(i+1)*self.N_sep])
        
        
        #print("err",np.sum(info_use!=EST_info_use))
        #print(len(info_use))
        #print(len(EST_info))
        
        return info,EST_info
    
if __name__=='__main__':
    K=512 #symbol数
    M=16
    
    EsNodB=10.0
    print("EsNodB",EsNodB)
    system=Mysystem(M,K)
    print("\n")
    print(system.N,system.K)
    
    MAXCNT=5
    count_err=0
    count_all=0
    while count_err<MAXCNT:
        count_all+=1
        info,EST_info=system.main_func(EsNodB)
        print("\r"+str(np.sum(info!=EST_info))+str(count_all)+str(count_err),end="")
        if np.any(info!=EST_info):
            count_err+=1
    print("result")
    print(count_err/count_all)