import numpy as np
import math
import os

def srandom_interleave(N,M):
        
    mod=(M)**(1/2)//2
    s=math.floor(math.sqrt(N))-5
    print(s)
    #step 1 generate random sequence
    vector=np.arange(N,dtype='int')
    np.random.shuffle(vector)

    itr=True
    count=0
    while itr:
        #intialize for each iteration
        heap=np.zeros(N,dtype='int')
        position=np.arange(N,dtype='int')

        #step2 set first vector to heap
        heap[0]=vector[0]
        position=np.delete(position,0)

        #step3 bubble sort 
        #set to ith heap
        for i in range(1,N):
            #serch jth valid position
            for pos,j in enumerate(position):
                # confirm valid or not
                for k in range(1,s+1):
                    if i-k>=0 and (abs(heap[i-k]-vector[j])+abs(i-k-j))<=s or (vector[j]%mod)!=(i%mod):
                        '''
                        i-k>=0 : for the part i<s 
                        (abs(heap[i-k]-vector[j]))<=s : srandom interleaver
                        vector[j]//mod!=i//mod : mod M interleaver(such as odd-even)
                        '''
                        #vector[j] is invalid and next vector[j+1]
                        break

                #vector[j] is valid and set to heap[i]
                else:
                    heap[i]=vector[j]
                    position=np.delete(position,pos)
                    break
            #if dont exit num at heap[i]
            else:
                #set invalid sequence to the top and next iteration
                tmp=vector[position]
                np.random.shuffle(tmp)
                vector[0:N-i]=tmp
                vector[N-i:N]=heap[0:i]
                break

        #if all the heap num is valid, end iteration
        else:
            itr=False
        
        #print(heap)
        #print(vector)
        print("\r","itr",count,end="")
        count+=1
    
    return heap

def make_BICM(N,M):
    
    #make BICM directory
    # directory make
    current_directory="/home/kaneko/Dropbox/programming/geometric_shaping/modulation"
    #current_directory=os.getcwd()
    dir_name="BICM_interleaver"
    dir_name=current_directory+"/"+dir_name
    
    try:
        os.makedirs(dir_name)
    except FileExistsError:
        pass
    
    filename="length{}_mod{}".format(N,int((M)**(1/2)//2))

    #if file exists, then load txt file
    filename=dir_name+"/"+filename
    
    try:
        BICM_int=np.loadtxt(filename,dtype='int')
    except FileNotFoundError:
        BICM_int=srandom_interleave(N,M)
        #export file
        np.savetxt(filename,BICM_int,fmt='%d')
    
    BICM_deint=np.argsort(BICM_int)
    
    #check
    a=np.arange(N)
    b=a[BICM_int]
    c=b[BICM_deint]
    if np.any(a!=c):
        print("BICM interleaver error!")
    
    
    return BICM_int,BICM_deint