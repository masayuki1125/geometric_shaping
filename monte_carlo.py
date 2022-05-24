import ray
import pickle
import sys
import numpy as np
from main import Mysystem

ray.init()

@ray.remote
def output(dumped,EbNodB):
    '''
    #あるSNRで計算結果を出力する関数を作成
    #cd.main_func must input 'EbNodB' and output 1D 'codeword' and 'EST_codeword'
    '''

    #de-seriallize file
    cd=pickle.loads(dumped)
    #seed値の設定
    np.random.seed()

    #prepare some constants
    MAX_ALL=10**2
    MAX_ERR=1
    count_bitall=0
    count_biterr=0
    count_all=0
    count_err=0
    

    while count_all<MAX_ALL and count_err<MAX_ERR:
        #print("\r"+str(count_err),end="")
        information,EST_information=cd.main_func(EbNodB)
        
        #calculate block error rate
        if np.any(information!=EST_information):
            count_err+=1
        count_all+=1

        #calculate bit error rate 
        count_biterr+=np.sum(information!=EST_information)
        count_bitall+=len(information)

    return count_err,count_all,count_biterr,count_bitall
# In[11]:


class MC():
    def __init__(self,K):
        self.K=K
        self.TX_antenna=1
        self.RX_antenna=1
        self.MAX_ERR=100
        self.EbNodB_start=0
        self.EbNodB_end=20
        self.EbNodB_range=np.arange(self.EbNodB_start,self.EbNodB_end,0.5) #0.5dBごとに測定

    #特定のNに関する出力
    def monte_carlo_get_ids(self,dumped):
        '''
        input:main_func
        -----------
        dumped:seriallized file 
        main_func: must input 'EbNodB' and output 1D 'codeword' and 'EST_codeword'
        -----------
        output:result_ids(2Darray x:SNR_number y:MAX_ERR)
        '''

        print("from"+str(self.EbNodB_start)+"to"+str(self.EbNodB_end))
        
        result_ids=[[] for i in range(len(self.EbNodB_range))]

        for i,EbNodB in enumerate(self.EbNodB_range):
            
            for j in range(self.MAX_ERR):
                #multiprocess    
                result_ids[i].append(output.remote(dumped,EbNodB))  # 並列演算
                #resultは長さ1のリストの中にBLER,BERの2つのarrayのtupleが入った配列
        
        return result_ids
    
    def monte_carlo_calc(self,result_ids_array,M_list):

        #prepare constant
        tmp_num=self.MAX_ERR
        tmp_ids=[]

        #Nのリストに対して実行する
        for i,M in enumerate(M_list):
            #特定のNに対して実行する
            #特定のNのBER,BLER系列
            BLER=np.zeros(len(self.EbNodB_range))
            BER=np.zeros(len(self.EbNodB_range))

            for j,EbNodB in enumerate(self.EbNodB_range):#i=certain SNR
                
                #特定のSNRごとに実行する
                while sum(np.isin(result_ids_array[i][j], tmp_ids)) == len(result_ids_array[i][j]):#j番目のNの、i番目のSNRの計算が終わったら実行
                    finished_ids, running_ids = ray.wait(result_ids_array[i], num_returns=tmp_num, timeout=None)
                    tmp_num+=1
                    tmp_ids=finished_ids

                result=ray.get(result_ids_array[i][j])
                #resultには同じSNRのリストが入る
                count_err=0
                count_all=0
                count_biterr=0
                count_bitall=0
                
                for k in range(self.MAX_ERR):
                    tmp1,tmp2,tmp3,tmp4=result[k]
                    count_err+=tmp1
                    count_all+=tmp2
                    count_biterr+=tmp3
                    count_bitall+=tmp4

                BLER[j]=count_err/count_all
                BER[j]=count_biterr/count_bitall

                print("\r"+"EbNodB="+str(EbNodB)+",BLER="+str(BLER[j])+",BER="+str(BER[j]),end="")
                
                if count_err/count_all<10**-5:
                    print("finish")
                    break
            
            #特定のNについて終わったら出力
            st=savetxt(M,self.K)
            st.savetxt(BLER,BER)



# In[ ]:


#毎回書き換える関数
class savetxt():
  
  def __init__(self,M,K):
    self.mysys=Mysystem(M,K)
    self.mc=MC(K)
    '''
    if M==16:
        mc.EbNodB_start+=5
        mc.EbNodB_end+=5
        mc.EbNodB_range=np.arange(mc.EbNodB_start,mc.EbNodB_end,0.5)
    elif M==256:
        mc.EbNodB_start+=10
        mc.EbNodB_end+=10
        mc.EbNodB_range=np.arange(mc.EbNodB_start,mc.EbNodB_end,0.5)
    '''

  def savetxt(self,BLER,BER):
    with open(self.mysys.filename,'w') as f:

        #print("#N="+str(self.mysis.cd.N),file=f)
        #print("#K="+str(self.mysis.cd.K),file=f)
        #print("#list_size="+str(self.cd.list_size),file=f)
        #print("#Strong User SNR="+str(self.cd.EbNodB1),file=f)
        #print("#power allocation beta="+str(self.cd.beta),file=f)
        #print("#TX_antenna="+str(self.mc.TX_antenna),file=f)
        #print("#RX_antenna="+str(self.mc.RX_antenna),file=f)
        #print("#modulation_symbol="+str(self.ch.M),file=f)
        #print("#MAX_BLERR="+str(self.mc.MAX_ERR),file=f)
        #print("#construction="+str(self.mysys.const),file=f)
        print("#EsNodB,BLER,BER",file=f)  
        for i in range(len(self.mc.EbNodB_range)):
            print(str(self.mc.EbNodB_range[i]),str(BLER[i]),str(BER[i]),file=f)

# In[ ]:
if __name__=="__main__":
    K=128
    print("K=",K)
    mc=MC(K)
    M_list=[256]
    result_ids_array=[]
    
    for M in M_list:
        
        '''
        if M==16:
            mc.EbNodB_start+=5
            mc.EbNodB_end+=5
            mc.EbNodB_range=np.arange(mc.EbNodB_start,mc.EbNodB_end,0.5)
        elif M==256:
            mc.EbNodB_start+=10
            mc.EbNodB_end+=10
            mc.EbNodB_range=np.arange(mc.EbNodB_start,mc.EbNodB_end,0.5)
        '''
        print(mc.EbNodB_range)
        
        cd=Mysystem(M,K)
        dumped=pickle.dumps(cd)
        print("M",M)
        result_ids_array.append(mc.monte_carlo_get_ids(dumped))

    mc.monte_carlo_calc(result_ids_array,M_list)
