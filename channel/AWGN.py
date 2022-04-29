import numpy as np
import cupy as cp
import math
import time


#QAMは工事中

class _AWGN():
    def __init__(self,beta=1,M=2):
        '''
        input constant about channel
        -----------
        M:変調多値数
        TX_antenna:送信側アンテナ数
        RX_antenna:受信側アンテナ数
        '''
        super().__init__()

        self.M=M
        self.TX_antenna=1
        self.RX_antenna=1
        self.beta=beta

    def generate_QAM(self,info):
        if self.M==2:
            const=2*info-1
            
        elif self.M==4:
            const1=2*info[::2]-1
            const2=2*info[1::2]-1
            const=(self.beta)**(1/2)*const1+(1-self.beta)**(1/2)*const2*(-1*const1)
            #print(const)
        return const
    
    '''
    def generate_PAM(self,info1,info2,beta=0.2):
        const1=(beta)**(1/2)*(2*info1-1)
        const2=((1-beta))**(1/2)*(2*info2-1)
        const=const1+const2
        
        return const
    '''

    def add_AWGN(self,constellation,No,beta):

        # AWGN雑音の生成
        noise = np.random.normal(0, math.sqrt(No / 2), (len(constellation))) \
                #+ 1j * np.random.normal(0, math.sqrt(No / 2), (len(constellation)))

        # AWGN通信路 = 送信シンボル間干渉が生じないような通信路で送信
        RX_constellation = constellation + noise 

        # 以下のprint関数の出力を表示すると、Noとほぼ一致するはず
        #print(np.dot(noise[0, :], np.conj(noise[0, :]))/bit_num)

        return RX_constellation
    
    def add_Rayleigh(self,constellation,No,beta):
    
        # AWGN雑音の生成
        noise = np.random.normal(0, math.sqrt(No / 2), (len(constellation))) \
                + 1j * np.random.normal(0, math.sqrt(No / 2), (len(constellation)))
                
        interference=np.random.randint(0,2,len(constellation))

        # AWGN通信路 = 送信シンボル間干渉が生じないような通信路で送信
        RX_constellation = (1-beta)**(1/2)*constellation + noise +(beta)**(1/2)*interference

        # 以下のprint関数の出力を表示すると、Noとほぼ一致するはず
        #print(np.dot(noise[0, :], np.conj(noise[0, :]))/bit_num)

        return RX_constellation
    
    @staticmethod
    def calc_exp(x,A,No):
        #解が0にならないように計算する
        res=np.zeros(len(x))
        for i in range(len(x)):
            if (x[i]-A)**2/No<30:
                res[i]=np.exp(-1*(x[i]-A)**2/No)
            else:
                res[i]=10**(-15)
        return res
    
    def calc_LLR(self,x,No):
        A1=self.calc_exp(x,-(1-self.beta)**(1/2)-(self.beta)**(1/2),No)
        A2=self.calc_exp(x,-(1-self.beta)**(1/2)+(self.beta)**(1/2),No)
        A3=self.calc_exp(x,(1-self.beta)**(1/2)-(self.beta)**(1/2),No)
        A4=self.calc_exp(x,(1-self.beta)**(1/2)+(self.beta)**(1/2),No)
        
        Lc=np.zeros(int(math.log2(self.M))*len(x))
        Lc[::2]=np.log((A3+A4)/(A1+A2))
        Lc[1::2]=np.log((A2+A3)/(A1+A4))
        
        #print(Lc)
        #print(y2)
        #print(y1)
        return Lc
    
    def demodulate(self,RX_constellation,No):
        if self.M==2:
            y=RX_constellation.real
            Lc=4*y/No
        
        elif self.M==4:
            y=RX_constellation.real
            #print("y=",y)
            Lc=self.calc_LLR(y,No)
        
        #print(Lc)
        
        return Lc

    def generate_LLR(self,information,EbNodB,Rayleigh=False):
        '''
        information:1D sequence
        EbNodB:EsNodB
        --------
        output:LLR of channel output
        '''
        # Additive Gaussian White Noiseの生成する際のパラメータ設定
        EbNo = 10 ** (EbNodB / 10)
        No=1/EbNo #Eb=1(fixed)

        constellation=self.generate_QAM(information)
        if Rayleigh==False:
            RX_constellation=self.add_AWGN(constellation,No,self.beta)
        else:
            RX_constellation=self.add_Rayleigh(constellation,No,self.beta)
            
        Lc=self.demodulate(RX_constellation,No)
        #print(Lc)
        return Lc

if __name__=="__main__":
    beta=0.01
    M=4
    def f(const1,const2):
        return (1-beta)**(1/2)*const1+(beta)**(1/2)*const2*(-1*const1)
    #const1,const2
    a=f(-1,-1) #00
    b=f(1,-1) #10
    c=f(-1,1) #01
    d=f(1,1) #11
    print(a,c,d,b)
    print(a-c,c-d,d-b)
    print(a**2+b**2+c**2+d**2)
    #print(a)
    ch=_AWGN(beta)
    time_start = time.time()  
    information=np.ones(100)
    res=ch.generate_LLR(information,100)
    #print(res)
    res=np.sign(res)
    EST_information=(res+1)//2
    #print(EST_information)
    #print(information)
    print(np.sum(information!=EST_information))
    #print(ch.channel(information,100))
    
    K=100
    MAX_ERR=100
    
    for i in range(1,99):
        beta=1/100  
        ch=_AWGN(beta)  
        for EbNodB in range(10,11):
            print(EbNodB)
            count_err=0
            count_all=0
            while count_err<MAX_ERR:
                information=np.random.randint(0,2,K)
                #information=np.zeros(K)
                res=ch.generate_LLR(information,EbNodB)
                res=np.sign(res)
                EST_information=(res+1)//2
                #print(EST_information)
                #print(information)
                #print(np.sum(information[::2]!=EST_information[::2]),np.sum(information[1::2]!=EST_information[1::2]))
                count_err+=np.sum(information!=EST_information)
                #print(count_err)
                count_all+=K

            print(count_err/count_all)
        
    

    time_end = time.time()  
    time_cost = time_end - time_start  
    print('time cost:', time_cost, 's')
  