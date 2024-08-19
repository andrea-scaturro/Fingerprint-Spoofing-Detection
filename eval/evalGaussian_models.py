import numpy 
import matplotlib
import matplotlib.pyplot as plt
import scipy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'trainData.txt'))




from dataset import *
from src.dimreduction import *
from src.classifier_pca_lda import *
from src.plotter import *
from src.gaussian_analysis import *
from src.mvg import *



#======================= Main Program ==========================

if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load(data_path)
    m=6

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    error, _,_      = mvg(DTR, LTR, DVAL, LVAL)
    error_Tied, _,_ = mvg_tied(DTR, LTR, DVAL, LVAL)
    error_Nb, _,_   = mvg_nb(DTR, LTR, DVAL, LVAL)
    error_LDA       = lda_classifier(DTR, LTR, DVAL, LVAL,m)
    
    print("\nError Rate: " ,   (error     / DVAL.shape[1])*100 ,"%" )    
    print("Error Rate Tied: " ,(error_Tied/ DVAL.shape[1])*100, "%")
    print("Error Rate NB: " ,  (error_Nb  / DVAL.shape[1])*100, "%")
    print("Error Rate LDA: " , (error_LDA / DVAL.shape[1])*100, "%")


    # for 1-2-3-4 feature
    print("\nfeature 1 to 4:")

    DTR_f  = DTR[0:4, :]
    DVAL_f = DVAL[0:4, :]

    error, _,_      = mvg(DTR_f, LTR, DVAL_f, LVAL)
    error_Tied, _,_ = mvg_tied(DTR_f, LTR, DVAL_f, LVAL)
    error_Nb, _,_   = mvg_nb(DTR_f, LTR, DVAL_f, LVAL)
    error_LDA       = lda_classifier(DTR_f, LTR, DVAL_f, LVAL,m)
    
    print("\nError Rate: " ,   (error     / DVAL_f.shape[1])*100 ,"%" )    
    print("Error Rate Tied: " ,(error_Tied/ DVAL_f.shape[1])*100, "%")
    print("Error Rate NB: " ,  (error_Nb  / DVAL_f.shape[1])*100, "%")
    print("Error Rate LDA: " , (error_LDA / DVAL_f.shape[1])*100, "%")


    # for 1-2 feature
    print("\nfeature 1-2:")

    DTR_12  = DTR[0:2, :]
    DVAL_12 = DVAL[0:2, :]

    error, _,_      = mvg(DTR_12, LTR, DVAL_12, LVAL)
    error_Tied, _,_ = mvg_tied(DTR_12, LTR, DVAL_12, LVAL)
    error_Nb, _,_   = mvg_nb(DTR_12, LTR, DVAL_12, LVAL)
    error_LDA       = lda_classifier(DTR_12, LTR, DVAL_12, LVAL,m)
    
    print("\nError Rate: " ,   (error     / DVAL_12.shape[1])*100 ,"%" )    
    print("Error Rate Tied: " ,(error_Tied/ DVAL_12.shape[1])*100, "%")
    print("Error Rate NB: " ,  (error_Nb  / DVAL_12.shape[1])*100, "%")
    


    # for 3-4 feature
    print("\nfeature 3-4:")

    DTR_34  = DTR[2:4, :]
    DVAL_34 = DVAL[2:4, :]

    error, _,_      = mvg(DTR_34, LTR, DVAL_34, LVAL)
    error_Tied, _,_ = mvg_tied(DTR_34, LTR, DVAL_34, LVAL)
    error_Nb, _,_   = mvg_nb(DTR_34, LTR, DVAL_34, LVAL)
    error_LDA       = lda_classifier(DTR_34, LTR, DVAL_34, LVAL,m)
    
    print("\nError Rate: " ,   (error     / DVAL_34.shape[1])*100 ,"%" )    
    print("Error Rate Tied: " ,(error_Tied/ DVAL_34.shape[1])*100, "%")
    print("Error Rate NB: " ,  (error_Nb  / DVAL_34.shape[1])*100, "%")
    


    # for 5-6 feature
    print("\nfeature 5-6:")

    DTR_56  = DTR[4:6, :]
    DVAL_56 = DVAL[4:6, :]

    error, _,_      = mvg(DTR_56, LTR, DVAL_56, LVAL)
    error_Tied, _,_ = mvg_tied(DTR_56, LTR, DVAL_56, LVAL)
    error_Nb, _,_   = mvg_nb(DTR_56, LTR, DVAL_56, LVAL)
    error_LDA       = lda_classifier(DTR_56, LTR, DVAL_56, LVAL,m)
    
    print("\nError Rate: " ,   (error     / DVAL_56.shape[1])*100 ,"%" )    
    print("Error Rate Tied: " ,(error_Tied/ DVAL_56.shape[1])*100, "%")
    print("Error Rate NB: " ,  (error_Nb  / DVAL_56.shape[1])*100, "%")
    


    
    # appling pca

    P1 = -PCA(DTR, 5)  
    D_pca = numpy.dot(P1.T, DTR)
    DVAL_pca = numpy.dot(P1.T, DVAL)

    print("\nPCA:")

    error, _,_      = mvg(D_pca, LTR, DVAL_pca, LVAL)
    error_Tied, _,_ = mvg_tied(D_pca, LTR, DVAL_pca, LVAL)
    error_Nb, _,_   = mvg_nb(D_pca, LTR, DVAL_pca, LVAL)
    
    
    print("\nError Rate: " ,   (error     / DVAL_pca.shape[1])*100 ,"%" )    
    print("Error Rate Tied: " ,(error_Tied/ DVAL_pca.shape[1])*100, "%")
    print("Error Rate NB: " ,  (error_Nb  / DVAL_pca.shape[1])*100, "%")
    




    res = pearson_correlation(DTR, LTR, DVAL,LVAL)
    numpy.set_printoptions(precision=4)
    numpy.set_printoptions(suppress=True)
    print("\nCorrelation Matrix Class1: \n")
    print(res[0])
    
    print("\nCorrelation Matrix Class2: \n")
    print(res[1])
  