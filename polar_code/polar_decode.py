import numpy as np
import math

class decoding():
  
  def __init__(self,myPC,myPC_ec):
    #successeeded constants
    self.N=myPC.N
    self.K=myPC.K
    self.decoder_var=myPC.decoder_ver

    self.systematic_polar=myPC.systematic_polar
    
    #specific constants
    self.list_size=4
    self.itr_num=int(math.log2(self.N))
    
    #functions
    self.CRC_gen=myPC.CRC_gen
    self.CRC_polynomial=myPC.CRC_polynomial
    self.info_bits=myPC.info_bits
    self.frozen_bits=myPC.frozen_bits
    self.bit_reversal_sequence=myPC.bit_reversal_sequence
    
    self.generate_U=myPC_ec.generate_U
    self.encode=myPC_ec.encode
    
  @staticmethod
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

  def SC_decoding(self,Lc,info=False):
    #initialize constant    
    llr=np.zeros((self.itr_num+1,self.N))
    EST_codeword=np.zeros((self.itr_num+1,self.N))
    llr[0]=Lc

    #put decoding result into llr[logN]

    depth=0
    length=0
    before_process=0# 0:left 1:right 2:up 3:leaf

    while True:
  
      #left node operation
      if before_process!=2 and before_process!=3 and length%2**(self.itr_num-depth)==0:
        depth+=1
        before_process=0

        tmp1=llr[depth-1,length:length+2**(self.itr_num-depth)]
        tmp2=llr[depth-1,length+2**(self.itr_num-depth):length+2**(self.itr_num-depth+1)]

        llr[depth,length:length+self.N//(2**depth)]=self.chk(tmp1,tmp2)

      #right node operation 
      elif before_process!=1 and length%2**(self.itr_num-depth)==2**(self.itr_num-depth-1):
        
        #print(length%2**(self.itr_num-depth))
        #print(2**(self.itr_num-depth-1))
        
        depth+=1
        before_process=1

        tmp1=llr[depth-1,length-2**(self.itr_num-depth):length]
        tmp2=llr[depth-1,length:length+2**(self.itr_num-depth)]

        llr[depth,length:length+2**(self.itr_num-depth)]=tmp2+(1-2*EST_codeword[depth,length-2**(self.itr_num-depth):length])*tmp1

      #up node operation
      elif before_process!=0 and length!=0 and length%2**(self.itr_num-depth)==0:#今いるdepthより一個下のノードから、upすべきか判断する
      
        tmp1=EST_codeword[depth+1,length-2**(self.itr_num-depth):length-2**(self.itr_num-depth-1)]
        tmp2=EST_codeword[depth+1,length-2**(self.itr_num-depth-1):length]

        EST_codeword[depth,length-2**(self.itr_num-depth):length-2**(self.itr_num-depth-1)]=(tmp1+tmp2)%2
        EST_codeword[depth,length-2**(self.itr_num-depth-1):length]=tmp2

        depth-=1
        before_process=2
      
      else:
        print("error!")

      #leaf node operation
      if depth==self.itr_num:

        #frozen_bit or not
        if np.any(self.frozen_bits==length):
          EST_codeword[depth,length]=0
        
        #info_bit operation
        else :
          EST_codeword[depth,length]=(-1*np.sign(llr[depth,length])+1)//2
        
        length+=1 #go to next length

        depth-=1 #back to depth
        before_process=3
        
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        #print(llr)
        #print(EST_codeword)
        
        if length==self.N:
          break
    
    res=llr[self.itr_num]
    #print(res)
    #np.savetxt("llr",res)
    #from IPython.core.debugger import Pdb; Pdb().set_trace()
      
    if self.systematic_polar==True:
      #re encode polar
      u_message=self.generate_U(res[self.info_bits])
      res=self.encode(u_message[self.bit_reversal_sequence])
    
    return res
  
  @staticmethod
  def calc_BM(u_tilde,llr):
    if u_tilde*llr>30:
      return u_tilde*llr
    else:
      return math.log(1+math.exp(u_tilde*llr))

  def SCL_decoding(self,Lc):

    #initialize constant    
    llr=np.zeros((self.list_size,self.itr_num+1,self.N))
    EST_codeword=np.zeros((self.list_size,self.itr_num+1,self.N))
    llr[0,0]=Lc
    PML=np.full(self.list_size,10.0**10) #path metric of each list 
    PML[0]=0 

    #put decoding result into llr[L,logN]

    #prepere constant
    depth=0
    length=0
    before_process=0# 0:left 1:right 2:up 3:leaf
    branch=1#the number of branchs. 1 firstly, and increase up to list size 
    BM=np.full((self.list_size,2),10.0**10)#branch metrics
    # low BM is better

    while True:
      
      #interior node operation
  
      #left node operation
      if before_process!=2 and before_process!=3 and length%2**(self.itr_num-depth)==0:
        depth+=1
        before_process=0

        tmp1=llr[:,depth-1,length:length+2**(self.itr_num-depth)]
        tmp2=llr[:,depth-1,length+2**(self.itr_num-depth):length+2**(self.itr_num-depth+1)]

        #carculate each list index
        for i in range(branch):
          llr[i,depth,length:length+self.N//(2**depth)]=self.chk(tmp1[i],tmp2[i])

      #right node operation 
      elif before_process!=1 and length%2**(self.itr_num-depth)==2**(self.itr_num-depth-1):
        
        depth+=1
        before_process=1

        tmp1=llr[:,depth-1,length-2**(self.itr_num-depth):length]
        tmp2=llr[:,depth-1,length:length+2**(self.itr_num-depth)]

        #carculate each list index
        for i in range(branch):
          llr[i,depth,length:length+2**(self.itr_num-depth)]=tmp2[i]+(1-2*EST_codeword[i,depth,length-2**(self.itr_num-depth):length])*tmp1[i]

      #up node operation
      elif before_process!=0 and length!=0 and length%2**(self.itr_num-depth)==0:#今いるdepthより一個下のノードから、upすべきか判断する
      
        tmp1=EST_codeword[:,depth+1,length-2**(self.itr_num-depth):length-2**(self.itr_num-depth-1)]
        tmp2=EST_codeword[:,depth+1,length-2**(self.itr_num-depth-1):length]

        #carculate each list index
        for i in range(branch):
          EST_codeword[i,depth,length-2**(self.itr_num-depth):length-2**(self.itr_num-depth-1)]=(tmp1[i]+tmp2[i])%2
          EST_codeword[i,depth,length-2**(self.itr_num-depth-1):length]=tmp2[i]

        depth-=1
        before_process=2

      #leaf node operation
      if depth==self.itr_num:
        
        #frozen_bit or not
        if np.any(self.frozen_bits==length):

          #decide each list index
          for i in range(branch):
            EST_codeword[i,depth,length]=0
          
          #update path metric
          u_tilde=-1#because frozen_bit is 0
          for i in range(branch):
            PML[i]=PML[i]+self.calc_BM(u_tilde,llr[i,depth,length])
        
        #info_bit operation
        else :

          #decide each list index
          for i in range(branch):

            u_tilde=-1*np.sign(llr[i,depth,length])#[-1,1]
            #llr<0 -> u_tilde=1 u_hat=1 // u_hat(sub)=0
            #llr>0 -> u_tilde=-1 u_hat=0 // u_hat(sub)=1
            
            #decide main u_hat 
            tmp0=self.calc_BM(u_tilde,llr[i,depth,length])
            tmp1=self.calc_BM(-1*u_tilde,llr[i,depth,length])
            BM[i,int((u_tilde+1)//2)]=tmp0

            #decide sub u_hat
            BM[i,int((-1*u_tilde+1)//2)]=tmp1

          #branch*2 path number
          #update PM

          #update BM to PML 2d array
          BM[0:branch]+=np.tile(PML[0:branch,None],(1,2))

          #update branch
          branch=branch*2
          if branch>self.list_size:
            branch = self.list_size 

          #trim PML 2d array and update PML
          PML[0:branch]=np.sort(np.ravel(BM))[0:branch]
          list_num=np.argsort((np.ravel(BM)))[0:branch]//2#i番目のPMを持つノードが何番目のリストから来たのか特定する
          u_hat=np.argsort((np.ravel(BM)))[0:branch]%2##i番目のPMを持つノードがu_hatが0か1か特定する
          
          #copy before data
          #選ばれたパスの中で、何番目のリストが何番目のリストからの派生なのかを計算する
          #その後、llr，EST_codewordの値をコピーし、今計算しているリーフノードの値も代入する
          llr[0:branch]=llr[list_num,:,:]#listを並び替えru
          EST_codeword[0:branch]=EST_codeword[list_num,:,:]
          EST_codeword[0:branch,depth,length]=u_hat
                  
        length+=1 #go to next length

        depth-=1 #back to depth
        before_process=3
        
        if length==self.N:
          break
    
    res_list_num=0
    res=np.zeros((self.list_size,self.N))
    #print(res.shape)
    #print(res[i].shape)
         
      #set candidates
    for i in range(self.list_size):
      res[i,:]=EST_codeword[i,self.itr_num]
        
    #for systematic_polar
    if self.systematic_polar==True:
      #re encode polar
      for i in range(self.list_size):
        #print(i)
        #print(res[i][self.info_bits])
        u_message=self.generate_U(res[i,:][self.info_bits])
        res[i,:]=self.encode(u_message[self.bit_reversal_sequence])
    
    #CRC_check
    if self.decoder_var==2:
      for i in range(self.list_size):
        EST_CRC_info=res[i][self.info_bits]
        _,check=self.CRC_gen(EST_CRC_info,self.CRC_polynomial)
        #print(check)
        if check==True:
          res_list_num=i
          break
      
      #else:
        #print("no codeword")
    #print("CRC_err")

    #print("\r",PML,end="")
    
    return res[res_list_num]

  def polar_decode(self,Lc,info=False):
    '''
    polar_decode
    Lc: LLR fom channel
    decoder_var:int [0,1] (default:0)
    0:simpified SC decoder
    1:simplified SCL decoder
    '''
    if self.decoder_var==0:
      EST_codeword=self.SC_decoding(Lc)

    elif self.decoder_var==1 or self.decoder_var==2:
      EST_codeword=self.SCL_decoding(Lc)
        
      res=EST_codeword[self.info_bits]
    return res
  
if __name__=="__main__":
  from polar_construction import coding
  from polar_decode import decoding
  
  