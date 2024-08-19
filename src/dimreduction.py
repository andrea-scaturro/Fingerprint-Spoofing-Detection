import numpy 
import matplotlib
import matplotlib.pyplot as plt
import scipy


from dataset.dataset import *



def PCA(D, m):
    
    mu = D.mean(1)
    DC = D - mu.reshape((mu.size, 1))
    C = DC @ DC.T / (DC.shape[1] - 1)
    
    U, s, Vh  = numpy.linalg.svd(C)
    P = U[:, 0:m]    

    return P




def SbSw(D, L):
    
    SB = 0
    SW = 0
    mu = mcol(D.mean(1))  
    for i in range(L.max() + 1):

        DCls = D[:, L == i]  
        muCls = mcol(DCls.mean(1)) 
        SW += numpy.dot(DCls - muCls, (DCls - muCls).T)  
        SB += DCls.shape[1] * numpy.dot(muCls - mu, (muCls - mu).T)  

    SW /= D.shape[1] 
    SB /= D.shape[1]
    return SB, SW



def LDA(D, L, m):
    SB, SW = SbSw(D, L)
    s, U = scipy.linalg.eigh(SB, SW) 
    return U[:, ::-1][:, 0:m] 
    

