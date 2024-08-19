import numpy 
import matplotlib
import matplotlib.pyplot as plt
import scipy

from dataset.dataset import *
from .dimreduction import *
from .plotter import *


def predict_labels(DVAL,LVAL, threshold):
    # Inizializza un array di etichette previste
    PVAL = numpy.zeros(shape=LVAL.shape, dtype=numpy.int32)
    
    # Confronta i campioni proiettati con la soglia e assegna le etichette previste
    PVAL[DVAL[0] >= threshold] =1
    PVAL[DVAL[0] < threshold] = 0
    
    return PVAL

def count_errors(LVAL, PVAL):
    # Conta il numero di errori confrontando le etichette reali con le etichette previste
    num_errors = numpy.sum(LVAL.astype(numpy.int32) != PVAL)
    num_right = numpy.sum(LVAL.astype(numpy.int32) == PVAL)
    return num_errors,num_right



def pca_stats(D, L, m):
     
    mu = D.mean(1)
    D = D - mu.reshape((mu.size, 1))

    P1 = -PCA(D, m)  # Calcola solo la prima direzione PCA
    
    # Proietta i dati di addestramento e di validazione utilizzando la direzione PCA
    D_pca = numpy.dot(P1.T, D)
    plot_hist(D_pca, L,'pca')
    
    

def lda_stats(D, L, m):

    mu = D.mean(1)
    D = D - mu.reshape((mu.size, 1))

    P2=LDA(D, L, m) 
    D_lda=numpy.dot(P2.T, D)
    plot_hist(D_lda, L,'lda')
    


    
def pca_classifier(DTR, LTR,DVAL, LVAL, m):
     
    mu = DTR.mean()
    DTR = DTR - mu.reshape((mu.size, 1))

    mu = DVAL.mean(1)
    DVAL = DVAL - mu.reshape((mu.size, 1))
 
    P1 = PCA(DTR, 1)  # Calcola solo la prima direzione PCA
    
    # Proietta i dati di addestramento e di validazione utilizzando la direzione PCA
    DTR_pca = numpy.dot(P1.T, DTR)
    DVAL_pca = numpy.dot(P1.T, DVAL)

    #plot_hist_lab3(DTR_pca, LTR)
    #plot_hist_lab3(DVAL_pca, LVAL)
    
    threshold = (DTR_pca[0, LTR==0].mean() + DTR_pca[0, LTR==1].mean()) / 2.0
    PVAL = predict_labels(DVAL_pca,LVAL,threshold)
    error,_= count_errors(LVAL,PVAL)
    return error
   



def lda_classifier(DTR, LTR,DVAL, LVAL, m):

    P2=LDA(DTR, LTR, m)
    
    DVAL_lda=numpy.dot(P2.T, DVAL)
    DTR_lda=numpy.dot(P2.T, DTR)

    #plot_hist_lab3(DTR_lda, LTR, 'LDA')
    #plot_hist_lab3(DVAL_lda, LVAL, 'LDA')

    
    #pt1
    threshold = (DTR_lda[0, LTR==0].mean() + DTR_lda[0, LTR==1].mean()) / 2.0
    PVAL = predict_labels(DVAL_lda,LVAL,threshold)
    error,_= count_errors(LVAL,PVAL)
    return error

    #PVAL = compute_threshold(DTR_lda, LTR, DVAL_lda, LVAL)
    #return PVAL




def pca_lda_classifier(DTR, LTR, DVAL, LVAL, m):

    P1 = PCA(DTR, m)  # Calcola solo la prima direzione PCA
    
    
    # Proietta i dati di addestramento e di validazione utilizzando la direzione PCA
    DTR_pca = numpy.dot(P1.T, DTR)
    DVAL_pca = numpy.dot(P1.T, DVAL)    

    plot_hist_lab3(DTR_pca, LTR, 'PCA')
    plot_hist_lab3(DVAL_pca, LVAL, 'PCA')

    #LDA

    P2=LDA(DTR_pca, LTR, m)

    DVAL_lda=numpy.dot(P2.T, DVAL_pca)
    DTR_lda=numpy.dot(P2.T, DTR_pca)

    plot_hist_lab3(DTR_lda, LTR, 'LDA')
    plot_hist_lab3(DVAL_lda, LVAL, 'LDA')


    #threshold = (DTR_lda[0, LTR==1].mean() + DTR_lda[0, LTR==2].mean()) / 2.0

    PVAL = compute_threshold(DTR_lda, LTR, DVAL_lda, LVAL)

    return PVAL





def compute_threshold(DTR,LTR,DVAL,LVAL):
    
    threshold = (DTR[0, LTR==0].mean() + DTR[0, LTR==1].mean()) / 2.0
    
    i=-5
    min_error=2000
    while i<=5:

        PVAL = predict_labels(DVAL,LVAL, i)
        n1,n2=count_errors(LVAL,PVAL)
        if n1< min_error:
            min_error = n1
            thresh = i
            P = PVAL
        i+=0.00001
       

    return P