import numpy as np
import math

class decoding():
    
  def __init__(self,cd):
    self.K=cd.K
    self.n_out=cd.n_out
    self.turbo_int=cd.turbo_int
    self.turbo_deint=cd.turbo_deint
    self.rsc=cd.rsc
    self.ind_top=cd.ind_top
    self.ind_bot=cd.ind_bot
    self.R=cd.R
    self.n_tail_bits=cd.n_tail_bits
    self.output_all_iterations=False #:default:false 
    self.max_itr=cd.max_itr
    
  def decoding(self, Lc):

    # Systematic bit LLRs for each decoder
    lambda_s = Lc[0 : self.K]
    in_lambda_s = lambda_s[self.turbo_int]
    
    tail=self.n_tail_bits//2
    #print(tail)

    ptop_first=self.K
    
    ptop_last=ptop_first+len(self.ind_top)+tail
    pbot_first=ptop_last
    pbot_last=pbot_first+len(self.ind_bot)+tail
    
    ptop=Lc[ptop_first:ptop_last-tail]
    
    pbot=Lc[pbot_first:pbot_last-tail]

    ptop_llrs=np.zeros(self.K)
    ptop_llrs[self.ind_top]=ptop
    
    pbot_llrs=np.zeros(self.K)
    pbot_llrs[self.ind_bot]=pbot
    
    ctop_llrs=np.zeros(2*self.K)
    cbot_llrs=np.zeros(2*self.K)
    
    ctop_llrs[:2*self.K:2]=lambda_s
    ctop_llrs[1:2*self.K:2]=ptop_llrs
    
    ctop_llrs=np.concatenate([ctop_llrs,Lc[ptop_last-tail:ptop_last]])
    
    cbot_llrs[:2*self.K:2]=in_lambda_s
    cbot_llrs[1:2*self.K:2]=pbot_llrs
    
    cbot_llrs=np.concatenate([cbot_llrs,Lc[pbot_last-tail:pbot_last]])
    
    # Main loop for turbo iterations
    if self.output_all_iterations:#output 2D EST_information
      EST_information=np.zeros((self.K,self.max_itr))
    
    
    lambda_e, in_lambda_e = np.zeros(self.K), np.zeros(self.K)
    for i in range(self.max_itr):
      res = self.rsc.decode_bcjr(ctop_llrs, in_lambda_e[self.turbo_deint])
      lambda_e = res- in_lambda_e[self.turbo_deint] - lambda_s
      in_res = self.rsc.decode_bcjr(cbot_llrs, lambda_e[self.turbo_int])
      in_lambda_e = in_res - lambda_e[self.turbo_int] - in_lambda_s
      
      if self.output_all_iterations:#output 2D EST_information
        res = in_res[self.turbo_deint]
        EST_information[:,i] = (res < 0).astype(int)

    # Final post-decoding LLRs and hard decisions
    if self.output_all_iterations==False:
      res = in_res[self.turbo_deint]
      EST_information = (res < 0).astype(int)
      
    #print(EST_information.shape)

    return EST_information

  def turbo_decode(self,Lc):
    EST_information=self.decoding(Lc)
    return EST_information

if __name__=="__main__":
    from turbo_construction import coding
    from turbo_encode import encoding
    N=1024
    K=512
    cd=coding(N,K)
    
    ec=encoding(cd)
    dc=decoding(cd)
    
    info,cwd=ec.turbo_encode()
    
    
    info=dc.decoding(cwd)
    
    
    