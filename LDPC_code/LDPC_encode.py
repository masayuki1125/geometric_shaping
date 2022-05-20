import numpy as np

class encoding():
      
    def __init__(self,cd):
        
        self.encoder_var=0
        self.N=cd.N
        self.K=cd.K
        self.tG=cd.tG
        self.H=cd.H

    def generate_information(self):
        #generate information
        information=np.random.randint(0,2,self.K)
        return information

    def LDPC_encode(self):
        information=self.generate_information()
        codeword=self.tG@information%2
        
        if np.any(codeword@self.H.T%2!=0):
            print("encoder error!")
        #else:
            #print("pass")
        
        return information,codeword

if __name__=="__main__":
    from LDPC_construction import coding
    N=1024
    K=512
    cd=coding(N,K)
    #print(cd.H)
    
    ec=encoding(cd)
    ec.encode()