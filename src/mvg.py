import numpy 
import matplotlib
import matplotlib.pyplot as plt
import scipy

from dataset.dataset import *
from .dimreduction import *
from .classifier_pca_lda import *
from .plotter import *
from .gaussian_analysis import *





    #-----------Binary tasks: log-likelihood ratios and MVG---------------


def mvg(DTR, LTR, DVAL, LVAL, threshold=0,m =6 ):
        
        
        D = DTR[:, LTR==0]
        L = LTR[LTR == 0] 

        D2 = DTR[:, LTR==1]
        L2 = LTR[LTR == 1] 

        m1,c1 = compute_mu_C(D)
        m2,c2 = compute_mu_C(D2)

        ll = compute_ll(DVAL,m1,c1)
        ll2 = compute_ll(DVAL,m2,c2)

        llr= ll2-ll

        PVAL = numpy.zeros(shape=llr.shape[0], dtype=numpy.int32)
        # if threshold ==0:
        #      threshold = compute_threshold()

        PVAL[llr >= threshold] =1
        PVAL[llr < threshold] = 0
        error, _ = count_errors( LVAL, PVAL)
        
        return error, PVAL, llr



def mvg_tied(DTR, LTR, DVAL, LVAL, threshold=0, m=6):
        
        D = DTR[:, LTR==0]
        L = LTR[LTR == 0] 

        D2 = DTR[:, LTR==1]
        L2 = LTR[LTR == 1] 

        m1,c1 = compute_mu_C(D)
        m2,c2 = compute_mu_C(D2)

        if threshold:
             threshold = compute_threshold()
        CC = numpy.zeros((DTR.shape[0], DTR.shape[0])) 
        for label in numpy.unique(LTR): #ad ogni iterazione prendiamo i dati associati ad una etichetta
        
            D = DTR[:, LTR==label]
            L = LTR[LTR == label]  
        
            m,C = compute_mu_C(D)  #questa funzione ti trova la media e la cov e la mette nel dizionario    
            CC += C* D.shape[1] 
        
        
        CC *= 1/(DTR.shape[1])


        ll = compute_ll(DVAL,m1,CC)
        ll2 = compute_ll(DVAL,m2,CC)

        llr= ll2-ll
        
        PVAL = numpy.zeros(shape=llr.shape[0], dtype=numpy.int32)       
        PVAL[llr >= threshold] =1
        PVAL[llr < threshold] = 0
        error, _ = count_errors( LVAL, PVAL)
        return error, PVAL, llr




def mvg_nb(DTR, LTR, DVAL, LVAL, threshold=0,m=6):

        
        D = DTR[:, LTR==0]
        D2 = DTR[:, LTR==1] 

        m1,c1 = compute_mu_C(D)
        m2,c2 = compute_mu_C(D2)

        C = c1 @ numpy.eye(c1.shape[0])
        C2 = c2 @ numpy.eye(c2.shape[0])

        ll = compute_ll(DVAL,m1,C)
        ll2 = compute_ll(DVAL,m2,C2)
        llr= ll2-ll
        
        PVAL = numpy.zeros(shape=llr.shape[0], dtype=numpy.int32)
       
        if threshold:
             threshold = compute_threshold()

        PVAL[llr >= threshold] =1
        PVAL[llr < threshold] = 0
        error, _ = count_errors( LVAL, PVAL)
        return error,PVAL, llr


#threshold lab 5 
def compute_threshold():

    p1=0.5
    p2=0.5

    t = - numpy.log((p1/(p2)))
    
    return t




def pearson_correlation(DTR,LTR,DVAL,LVAL):
     
    Corr=[]
    for label in numpy.unique(LTR):
           
        D = DTR[:, LTR==label]
        L = LTR[LTR == label]

        _,C = compute_mu_C(D)
        corr = C / ( vcol(C.diagonal()**0.5) * vrow(C.diagonal()**0.5) )
        Corr.append(corr)
    
    return Corr


