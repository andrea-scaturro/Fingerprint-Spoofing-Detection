import numpy 
import matplotlib
import matplotlib.pyplot as plt
import scipy

from dataset.dataset import *
from .dimreduction import *
from .classifier_pca_lda import *
from .plotter import *




def statistics(D,L):

    mu = D.mean(1).reshape((D.shape[0], 1))  # D.mean fa la media per ogni feature, rispetto alla colonna
    print('Mean:')                           # facciamo il reshape per avere un vettore colonna
    print(mu)                                
    print()

    DC = D - mu                             # senza reshape avremmo avuto problemi col broadcasting
    
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    print('Covariance:')
    print(C)
    print()

    var = D.var(1)
    std = D.std(1)
    print('Variance:', var)
    print('Std. dev.:', std)
    print()
    
    for cls in [0,1]:
        print('Class', cls)
        DCls = D[:, L==cls]
        mu = DCls.mean(1).reshape(DCls.shape[0], 1)
        print('Mean:')
        print(mu)
        C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
        print('Covariance:')
        print(C)
        var = DCls.var(1)
        std = DCls.std(1)
        print('Variance:', var)
        print('Std. dev.:', std)
        print()

    
def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = ((D-mu) @ (D-mu).T) / float(D.shape[1])
    return mu, C
        
    
