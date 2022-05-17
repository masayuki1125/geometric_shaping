import numpy as np
import cupy as cp
import math

class encoding():
  def __init__(self,myPC):
    
    #constants
    self.N=myPC.N
    self.K=myPC.K
    self.decoder_var=myPC.decoder_ver
    self.systematic_polar=myPC.systematic_polar
    
    #functions
    self.CRC_gen=myPC.CRC_gen
    self.CRC_polynomial=myPC.CRC_polynomial
    self.info_bits=myPC.info_bits
    self.frozen_bits=myPC.frozen_bits
    self.bit_reversal_sequence=myPC.bit_reversal_sequence
    
  
  def generate_information(self):
    #generate information
    
    info=np.random.randint(0,2,self.K)
    #print(information)

    if self.decoder_var==0 or self.decoder_var==1:
      return info
    elif self.decoder_var==2:
      CRC_info,_=self.CRC_gen(info,self.CRC_polynomial)

      ##check CRC_info
      _,check=self.CRC_gen(CRC_info,self.CRC_polynomial)
      if check!=True:
        print("CRC_info error")
      
      return CRC_info
      

  def generate_U(self,information):
    u_message=np.zeros(self.N)
    u_message[self.info_bits]=information
    return u_message

  def encode(self,u_message):
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

        codeword = np.concatenate([self.encode(u1u2), self.encode(u2)])
    return codeword

  def systematic_encode(self,information):
    X=np.zeros((self.N,int(math.log2(self.N))+1))
    X[self.frozen_bits,0]=0
    X[self.info_bits,int(math.log2(self.N))]=information

    for i in reversed(range(1,self.N+1)):
      if np.any(i-1==self.info_bits):
        s=int(math.log2(self.N))+1
        delta=-1
      else:
        s=1
        delta=1

      #binary representation
      tmp=format (i-1,'b')
      b=tmp.zfill(int(math.log2(self.N))+1)
        
      for j in range(1,int(math.log2(self.N))+1):
        t=s+delta*j
        l=min(t,t-delta)
        kai=2**(int(math.log2(self.N))-l)
        #print(l)
        if int(b[l])==0:
          #print("kai")
          #print(i-kai-1)
          #print(i-1)
          X[i-1,t-1]=(X[i-1,t-delta-1]+X[i+kai-1,t-delta-1])%2
        
        else:
          #print("b")
          #print("kai")
          #print(i-kai)
          X[i-1,t-1]=X[i-1,t-delta-1]
        
    #print(X)

    #check
    x=X[:,int(math.log2(self.N))]
    y=X[:,0]
    codeword=self.encode(y[self.bit_reversal_sequence])
    if np.any(codeword!=x):
      print(codeword)
      print("err")
    
    return x

  def polar_encode(self):
    info=self.generate_information()
    
    #for systematic polar code encode 
    if self.systematic_polar==True:
      cwd=self.systematic_encode(info)
    
    else:
      u_message=self.generate_U(info)
      cwd=self.encode(u_message[self.bit_reversal_sequence])
      #bool型なので、int型に変更する
    #codeword=u_message@self.Gres%2
    return info,cwd.astype(np.int)

if __name__=="__main__":
  
  from polar_construction import coding
  
  N=4
  #for k in range(N):
  cd=coding(N,N)
  cd.systematic_polar=False
  ec=encoding(cd)
  info,cwd=ec.polar_encode()
  print(info,cwd)
    
    