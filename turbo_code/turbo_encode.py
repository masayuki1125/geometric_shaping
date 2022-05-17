import numpy as np
import math

class encoding():
    
  def __init__(self,cd):
    
    self.rsc=cd.rsc
    self.N=cd.N
    self.K=cd.K
    self.turbo_int=cd.turbo_int
    
    self.n_out=cd.n_out
    self.n_tail_bits=cd.n_tail_bits
    self.ind_top=cd.ind_top
    self.ind_bot=cd.ind_bot
    
  def generate_information(self):
    information=np.random.randint(0,2,self.K)
    return information

  def encoding(self, information):
    #print("info is")
    #print(len(information))
    # Get code bits from each encoder.
    ctop = self.rsc.encode(information)
    cbot = self.rsc.encode(information[self.turbo_int])
    
    systematic=ctop[:2*self.K:2]
    
    ptop=ctop[1:2*self.K:2]
    ptop=ptop[self.ind_top]
    ptop=np.concatenate([ptop,ctop[2*self.K:]])
    pbot = cbot[1:2*self.K:2]
    pbot=pbot[self.ind_bot]
    pbot=np.concatenate([pbot,cbot[2*self.K:]])
    '''
    codeword, pos = -np.ones(self.n_out * self.K + self.n_tail_bits, dtype=int), 0
    #符号長はパンクチャされる文だけ減らす
    for k in range(self.K):#top情報ビットを入れる
        codeword[pos : pos + self.rsc.n_out] = ctop[self.rsc.n_out * k : self.rsc.n_out * (k + 1)]
        pos += self.rsc.n_out
        codeword[pos : pos + self.rsc.n_out - 1] = cbot[self.rsc.n_out * k + 1 : self.rsc.n_out * (k + 1)]
        pos += self.rsc.n_out - 1
    
    codeword[pos : pos + self.rsc.n_out * self.rsc.mem_len] = ctop[self.rsc.n_out * self.K :]#tail bitを入れる
    codeword[pos + self.rsc.n_out * self.rsc.mem_len :] = cbot[self.rsc.n_out * self.K :]#tail bitを入れる
    '''
    cwd=np.concatenate([systematic,ptop,pbot])
    #print("cwd_len",len(cwd))
    return cwd
   
  def turbo_encode(self):
        info=self.generate_information()
        cwd=self.encoding(info)
        
        return info,cwd

if __name__=="__main__":
    from turbo_construction import coding
    N=1024
    K=16
    cd=coding(N,K)
    
    ec=encoding(cd)
    
    info,cwd=ec.turbo_encode()
    print(len(info))
    print(len(cwd))