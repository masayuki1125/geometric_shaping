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
        
        self.type=7#1:separated scheme 2:Block intlv(No intlv in arikan polar decoder) 3:No intlv(Block intlv in arikan polar decoder) 4:rand intlv
        
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
            
            #復号した推定符号ごと対応するマッピングパターンを出力する
            self.make_corr_cwd()
            
            BICM_int=np.reshape(BICM_int,[int(np.log2(M**(1/2))),-1],order='C')
            BICM_int=np.ravel(BICM_int,order='F')
            
        else:
            print("interleaver type error")
        BICM_deint=np.argsort(BICM_int)
        #np.savetxt("deint",BICM_deint,fmt='%.0f')
        #print(BICM_int)
        #print(BICM_deint) 
        return BICM_int,BICM_deint
    
    def make_corr_cwd(self):
        
        base_num=2
        
        #base_numがlのとき、l×lの生成行列を用意する
        G_0=np.array([[1,0],[1,1]])
        
        if len(G_0)!=base_num:
            print("G_0 is wrong!")
        
        #corr_cwd_keys[0]は、1ビット目が0だったときの可能性のある符号語のリスト
        #corr_cwd_keys[1]は、1ビット目が1だったときの可能性のある符号語のリスト
        
        self.corr_cwd_keys=[[],[]]
        
        #base_num列のすべてのパターンの情報ビットを生成する
        decs=[dec for dec in range(2**2)]
        bin_out = [np.binary_repr(d, width=base_num) for d in decs] #文字列の情報ビットのリスト
        bin_out= [np.array(list(bin_out_str),dtype=int) for bin_out_str in bin_out] #int型の情報ビットの行列
        
        for info in bin_out:
            #print(info)
            #mapperと同じコードで符号化する
            info=np.reshape(info,[int(np.log2(self.M**(1/2))),-1],order='C')
            map=(info.T@G_0)%2
            map=np.ravel(map,order='F')
            #print(cwd)
            if info[0]==0:
                map=np.array(map,dtype=str)#文字列に変換
                map=''.join(map)#文字列の行列を1つの文字列に変換
                
                self.corr_cwd_keys[0]+=[map]
            elif info[0]==1:
                map=np.array(map,dtype=str)#文字列に変換
                map=''.join(map)#文字列の行列を1つの文字列に変換
                
                self.corr_cwd_keys[1]+=[map]
            else:
                print("error")
        print("corresponding codewords")        
        print(self.corr_cwd_keys)
    
    def make_key(self,EST_cwd,key_num,No):
        '''
        推定符号語を用いることにより、より良いLLRを受信シンボルから導出する
        key_num:シンボル内の何ビット目のビットについてかの情報
        pre_dec_num:今まで何回復号したのかの情報
        EST_cwd:今まで復号し終わった推定符号語
        '''
        
        symb_len=self.N//int(np.log2(self.M))
        
        new_num = np.zeros((2,symb_len)) #zero_iと同じ行列縦軸M/2、横軸シンボル長
        new_denum = np.zeros(new_num.shape) 

        bin_seq = self.modem.de2bin(self.modem.m)
        
        #不要なkeynumが入力された場合、nan行列を返す
        if key_num==0 or key_num==2:
            res=np.zeros(symb_len)
            res[:]=np.nan
            return res
        
        #推定符号語をインターリーブし、シンボルごとの順番にする
        tmp=EST_cwd[self.BICM_int]
        
        tmp=np.reshape(tmp,[int(np.log2(self.M)),-1],order='F')
        
        #check
        if np.any(np.isnan(tmp[0])): #一つでもnanがあった場合、誤り
            print("tmp[0] err")
        elif np.any(np.isnan(tmp[2])):#一つでもnanがあった場合、誤り
            print("tmp[2] err")
        elif np.all(np.isnan(tmp[1]))==False:#すべてがnan出なかった場合、誤り
            print("tmp[1] err")
        elif np.all(np.isnan(tmp[3]))==False:#すべてがnan出なかった場合、誤り
            print("tmp[3] err")
        else:
            print("cwd is ok")
        
        #情報ビットを取り出す
        tmp=tmp[0::2]

        #tmp=np.ravel(self.txcwd,order='F')
        if np.sum(self.txcwd[0::2]!=tmp)!=0:
            print("EST cwd is not equal to cwd!")
            from IPython.core.debugger import Pdb; Pdb().set_trace()

        #tmpをEST_cwdに代入する
        EST_cwd_2D=tmp
        
        #EST_cwd_2D[::-1,:]=EST_cwd_2D
        
        #code_book = self.modem.code_book
        
        
        #1シンボルあたり、決まっているビットが2ビット隣同士で存在する
        
        #print("pre_dec_num",pre_dec_num)
        #print("key_num",key_num)
        
        
        for symb_idx in range(symb_len):
            count_zero=0
            count_one=0
            count_zero_key=0
            count_one_key=0
            #print("map",self.txmap[:,symb_idx])
            #print("cwd",self.txcwd[:,symb_idx])
            
            
            #check
            txcwd=self.txcwd[:,symb_idx]
            txmap=self.txmap[:,symb_idx]
            
            if txcwd[0]!=EST_cwd_2D[0,symb_idx] or txcwd[2]!=EST_cwd_2D[1,symb_idx]:
                print("cwd error")
            else:
                #print("cwd is ok")
                pass
            
            txcwd=np.reshape(txcwd,[int(np.log2(self.M**(1/2))),-1],order='F')
            
            #print(txcwd)
            EST_map=(txcwd.T@np.array([[1,0],[1,1]]))%2
            #print(EST_map)
            EST_map=np.ravel(EST_map,order='C')
            
            #check
            if np.any(txmap!=EST_map):
                print("error map")
                print(txmap)
                print(EST_map)
                from IPython.core.debugger import Pdb; Pdb().set_trace()
            else:
                #print("map is ok")
                pass
            
            #print(EST_cwd[2*symb_idx])
            #print(EST_cwd[2*symb_idx+1])
            
            for bin_symb in bin_seq:
                
                writing=0
                
                #シンボルのインデックスを前半と後半で分ける
                former_bin=bin_symb[:len(bin_symb)//2]
                latter_bin=bin_symb[len(bin_symb)//2:]
                
                #former_binとlatter_binが対応するマッピングになっている場合、writingを立てる
                if former_bin in self.corr_cwd_keys[int(EST_cwd_2D[0,symb_idx])] and latter_bin in self.corr_cwd_keys[int(EST_cwd_2D[1,symb_idx])]:
                    writing=1
                    #print(bin_symb)
                    
                else:
                    pass


                if bin_symb[key_num] == '0':
                    #EST_cwdと同じビットになっているもののみカウント
                    #writingが立っている場合、その行列のインデックスに1を立てる
                    if writing==1:
                        #print("pass")
                        new_num[count_zero_key,symb_idx]=self.num[count_zero,symb_idx,key_num]
                        count_zero_key+=1
                    else:
                        pass
                    #インデックスをインクリメントする
                    count_zero+=1
                        
                elif bin_symb[key_num] == '1':
                    #EST_cwdと同じビットになっているもののみカウント
                    if writing==1:
                        #print("pass")
                        
                        new_denum[count_one_key,symb_idx]=self.denum[count_one,symb_idx,key_num]
                        count_one_key+=1
                        
                        #ones[count_one,symb_idx]=1
                    else:
                        pass
                    
                    #インデックスをインクリメントする
                    count_one+=1
                        
                else:
                    print("program error")
                    
            #check
            #print("correct map",txmap)
            #print(self.num[:,symb_idx,key_num].shape)
            #print(self.num[:,symb_idx,key_num])
            #print(self.denum[:,symb_idx,key_num])
            
            #tmp_num=self.num[:,symb_idx,key_num][zeros[:,symb_idx]]
            #tmp_denum=self.denum[:,symb_idx,key_num][ones[:,symb_idx]]
            
            #print("selected",self.num[:,symb_idx,key_num]*zeros[:,symb_idx])
            #print(self.denum[:,symb_idx,key_num]*ones[:,symb_idx])
            
            #res_num=np.exp(-1*np.array(tmp_num)/self.No)
            #res_denum=np.exp(-1*np.array(tmp_denum)/self.No)
            
            #print("exp")
            #print(res_num)
            #print(res_denum)
            
            #res_num=np.sum(res_num)
            #res_denum=np.sum(res_denum)
            #res_num=np.clip(res_num,10**(-15),10**15)
            #res_num=np.log(res_num)
            #res_denum=np.clip(res_denum,10**(-15),10**15)
            #res_denum=np.log(res_denum)
            #print(res_num)
            #print(res_denum)
            
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            
            #check
            if count_zero!=8 or count_one!=8:
                print("count_error")
        
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
        #print(zeros.shape)
        
        
        #if np.any(np.sum(zeros,axis=0)!=2) or np.any(np.sum(ones,axis=0)!=2):
            #print("1 is not 2 error!!")
            #print(np.sum(zeros,axis=0))
            #print(np.sum(ones,axis=0))
            
        if np.any(new_num==0) or np.any(new_denum==0):
            print("not filled keys!")
            print(self.one_key)
            print(self.zero_key)
        
        #print(zeros)
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        
        #llrの計算をする
        num_post=self.modem.calc_exp(new_num,No)
        denum_post=self.modem.calc_exp(new_denum,No)
        
        llr = np.transpose(num_post[0]) - np.transpose(denum_post[0])
        
        return llr
    
    '''
    def determine_keys(self,EST_info,pre_dec_num,key_num):
        
        #error check
        
        #tmp=np.sum(EST_info!=self.info_use)
        #print("former est_info error",tmp)
        
        #再エンコードする
        u_massage=self.ec[pre_dec_num].generate_U(EST_info)
        EST_cwd=self.ec[pre_dec_num].encode(u_massage[self.ec[pre_dec_num].bit_reversal_sequence])
        #print(np.sum(self.cwd_use!=EST_cwd))
        
        
       #u_massage=self.ec[pre_dec_num].generate_U(EST_info)
       #EST_cwd=self.ec[pre_dec_num].encode(u_massage[self.ec[pre_dec_num].bit_reversal_sequence])
        
        if np.any(EST_cwd!=self.cwd_1D[:len(self.cwd_1D)//2]):
            print("EST_cwd 1 error!")
            #print(EST_cwd)
            #print(self.cwd_1)
            print("error count",np.sum(EST_cwd!=self.cwd_use))
            from IPython.core.debugger import Pdb; Pdb().set_trace()
            
        #推定符号語から、Keyの行列を生成する
        self.make_key(EST_cwd,key_num)
        #zero_key,one_key=self.make_key(EST_cwd,key_num)
        
        #return zero_key,one_key
    '''
           
    def main_func(self,EsNodB):
        #すべてのデコーダ一律で凍結ビットを得る
        if self.decoder_ver==2:
            self.CRC_len=len(self.cd[0].CRC_polynomial)-1  
            frozen_bits,info_bits=self.const.main_const(self.N,self.K+self.CRC_len,EsNodB,self.M,BICM_int=self.BICM_int,type=self.type)
        else:
            frozen_bits,info_bits=self.const.main_const(self.N,self.K,EsNodB,self.M,BICM_int=self.BICM_int,type=self.type)
            
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
        
        #check
        self.No=No

        info=np.empty(0,dtype=int)
        cwd=np.empty(0,dtype=int)
        
        #info_use=np.empty(0,dtype=int)
        for i in range(self.enc_num):
            info_sep,cwd_sep=self.ec[i].polar_encode()
            #print(len(info_sep))
            
            #err check
            if i==0:
                #self.info_use=info_sep
                self.cwd_use=cwd_sep
                #cwd check
                #u_massage=self.ec[i].generate_U(info_sep)
                #tmp_cwd=self.ec[i].encode(u_massage[self.ec[i].bit_reversal_sequence])
                #if np.any(tmp_cwd!=cwd_sep):
                    #print("error gen cwd !")
                    #print(np.sum(tmp_cwd!=cwd_sep))
                    
                #else:
                    #print("cwd is ok")

            
            info=np.concatenate([info,info_sep])
            cwd=np.concatenate([cwd,cwd_sep])

        #check
        self.cwd_1D=cwd
        
        if self.type==7:
            cwd=np.reshape(cwd,[int(np.log2(self.M**(1/2))),-1],order='C')
            map=(cwd.T@np.array([[1,0],[1,1]]))%2
            map=np.ravel(map,order='F')
            
        else:
            map=cwd
        
        '''
        a=np.arange(len(self.BICM_int))
        b=np.arange(len(self.BICM_int))[self.BICM_int]
        np.savetxt("int",self.BICM_int,fmt="%.0f")
        np.savetxt("deint",self.BICM_deint,fmt="%.0f")
        print(a)
        print(b)
        '''
        
        #cwd check
        #self.cwd_1=cwd[:len(cwd)//2]
        
        map=map[self.BICM_int] #interleave
        
        self.txmap=np.reshape(map,[int(np.log2(self.M)),-1],order='F')
        self.txcwd=np.reshape(cwd.ravel(order='C')[self.BICM_int],[int(np.log2(self.M)),-1],order='F')
        
        TX_conste=self.modem.modulate(map)
        #print(TX_conste)
        RX_conste=self.ch.add_AWGN(TX_conste,No)
        if self.type==7:
            tmp=True
            Lc,mat_list=self.modem.demodulate(RX_conste,No,tmp)
            
            #euclid distanceの三次元行列をクラス変数にする
            self.num=mat_list[0]
            self.denum=mat_list[1]
            
        else:
            Lc=self.modem.demodulate(RX_conste,No)
        tmp_Lc=Lc
            
        Lc=Lc[self.BICM_deint] #de interleave
        
        if np.all([tmp_Lc]==Lc[self.BICM_int]):
            pass
        else:
            print("Lc error")
        
        EST_info=np.empty(0)
        for i in range(self.enc_num):
            if self.type==7:
                #compound polar codes
                if i==0:
                    llr=self.dc[0].chk(Lc[i*self.N_sep:(i+1)*self.N_sep],Lc[(i+1)*self.N_sep:(i+2)*self.N_sep])
                elif i==1:
                    #今まで復号したEST_infoから、次のEST_cwdを復号する  
                    
                    EST_cwd=np.zeros(self.N)
                    EST_cwd[:]=np.nan
                    K_sep_st=0
                    for pre_dec_num in range(i):#今まで復号した復号器の個数分の符号語を生成する
                        #再エンコードする
                        
                        K_sep_ed=len(self.ec[pre_dec_num].info_bits)
                        u_massage=self.ec[pre_dec_num].generate_U(EST_info[K_sep_st:K_sep_ed])
                        EST_cwd_sep=self.ec[pre_dec_num].encode(u_massage[self.ec[pre_dec_num].bit_reversal_sequence])
                        K_sep_st=K_sep_ed
                        EST_cwd[pre_dec_num*self.N_sep:(pre_dec_num+1)*self.N_sep]=EST_cwd_sep
                        
                    #インターリーブする
                    #EST_cwd=EST_cwd[self.BICM_int]
                    
                    #print(EST_cwd)
                        
                    ##check
                    if np.any(EST_cwd[:len(self.cwd_1D)//2]!=self.cwd_1D[:len(self.cwd_1D)//2]):
                        print("EST_cwd 1 error!")
                        #print(EST_cwd)
                        #print(self.cwd_1)
                        print("error count",np.sum(EST_cwd!=self.cwd_use))
                        from IPython.core.debugger import Pdb; Pdb().set_trace()
                    
                    #受信信号店からLLRを再計算する
                    llr=np.zeros(self.N)
                    llr[:]=np.nan
                    
                    for key_num in range(int(np.log2(self.M))): #同一シンボル内で後半の2ビットを復号したいので、1番目と3番めのインデックスを復号する
                        llr_sep=self.make_key(EST_cwd,key_num,No) #zeros_key,ones_keyはx軸が1シンボル内で取りうるビット列のインデックス、y軸がシンボル長の二次元配
                        llr[key_num::int(np.log2(self.M))]=llr_sep
                    
                    
                    #print(len(llr))
                    #for j in [0,2]:
                        #zero_key[:,:,j]=np.ones(zero_key[:,:,j].shape)
                        #one_key[:,:,j]=np.ones(zero_key[:,:,j].shape)
                    #num=np.stack([mat_list[0][:,:,[0,1]],mat_list[0][:,:,[2,3]]],axis=1)
                    #denum=np.concatenate([mat_list[1][:,:,[0,1]],mat_list[1][:,:,[2,3]]],axis=1)
                    #print(num.shape)
                    
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
                    #num_post=self.modem.calc_exp(self.zero_key,No)
                    #denum_post=self.modem.calc_exp(self.one_key,No)
                    #print(num_post)
                    #二次元配列のLLRを作成
                    #llr = np.transpose(num_post[0]) - np.transpose(denum_post[0])
                    #1次元のベクトルに変換
                    #llr=np.ravel(llr,order='F')
                    
                    #使うビットだけ取り出す
                    #print(llr)
                    
                    llr=llr[self.BICM_deint]
                    
                    if np.all(np.isnan(llr[:len(llr)//2]))!=True:
                        print("llr error")
                        
                    llr=llr[len(llr)//2:]
                    
                    #check
                    if np.any(np.isnan(llr)):
                        print("llr is nan error!")
                        from IPython.core.debugger import Pdb; Pdb().set_trace()
                    #print(llr)             
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
        #print(np.sum(info!=EST_info))
        
        return info,EST_info
    
if __name__=='__main__':
    K=512 #symbol数
    M=16
    
    EsNodB=8.0
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