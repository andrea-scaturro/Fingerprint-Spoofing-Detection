import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'trainData.txt'))

from tqdm import tqdm
from src.bayes_decision import *
from src.classifier_pca_lda import *
from dataset import *
from src.dimreduction import *
from src.gaussian_analysis import *
from src.mvg import *
from src.logistic_regression import *
from src.plotter import *
from src.svm import *

# ======================= Main Program ==========================



def plot_3minDCF(effPriorLogOdds, min_dcf_values_1, min_dcf_values_2, min_dcf_values_3, title=''):

    # cancello il grafico precedente
    plt.clf()

    plt.xscale("log", base=10)
    # Tracciare il grafico

    plt.plot(effPriorLogOdds, min_dcf_values_1, label='min DCF pi: 0.1', color='r')
    plt.plot(effPriorLogOdds, min_dcf_values_2, label='min DCF pi: 0.5', color='g')
    plt.plot(effPriorLogOdds, min_dcf_values_3, label='min DCF pi: 0.9', color='b')
    plt.ylim([0, 1.1])
    #plt.xlim([-4, 4])

    plt.xlabel('prior log-odds')
    plt.ylabel('DCF Value')
    plt.title('Bayes Error Plot: '+title)
    plt.legend()
    # plt.grid(True)
    #plt.savefig('plot/plot_logistic_regression/plot3minDCF/logistic_regression_'+title+'.pdf')
    plt.show()


def plot_3DCF(effPriorLogOdds, dcf_values_1, dcf_values_2, dcf_values_3, title=''):

    # cancello il grafico precedente
    plt.clf()
    
    plt.xscale("log", base=10)
    # Tracciare il grafico

    plt.plot(effPriorLogOdds, dcf_values_1, label='DCF pi: 0.1', color='r')
    plt.plot(effPriorLogOdds, dcf_values_2, label='DCF pi: 0.5', color='g')
    plt.plot(effPriorLogOdds, dcf_values_3, label='DCF pi: 0.9', color='b')
    plt.ylim([0, 1.1])
    #plt.xlim([-4, 4])

    plt.xlabel('prior log-odds')
    plt.ylabel('DCF Value')
    plt.title('Bayes Error Plot: '+title)
    plt.legend()
    # plt.grid(True)
    #plt.savefig('plot/plot_logistic_regression/plot3DCF/logistic_regression_'+title+'.pdf')
    plt.show()



def plot_4DCF_MinDCF(effPriorLogOdds,pi,title=''):


    DCF_values_g4 = []
    DCF_values_g3 = []
    DCF_values_g2 = []
    DCF_values_g1 = []

    minDCF_values_g4 = []
    minDCF_values_g3 = []
    minDCF_values_g2 = []
    minDCF_values_g1 = []


    for g in [np.exp(-4),np.exp(-3),np.exp(-2),np.exp(-1)]:
        for e in effPriorLogOdds:
            
            K =1
            
            kernelFunc = rbfKernel(g)
            eps = 0
            fScore = train_dual_SVM_kernel(DTR, LTR, e, kernelFunc, eps)
            SVAL = fScore(DVAL)


            minDCF = compute_minDCF(DVAL, LVAL, SVAL, pi, 1, 1)
            DCF = compute_actDCF(DVAL, LVAL, SVAL, pi, 1, 1)
            
            if(np.log(g) == -4):
                
                minDCF_values_g4.append(minDCF)
                DCF_values_g4.append(DCF)

            if(np.log(g) == -3):
                
                minDCF_values_g3.append(minDCF)
                DCF_values_g3.append(DCF)

            if(np.log(g) == -2):
                
                minDCF_values_g2.append(minDCF)
                DCF_values_g2.append(DCF)

            if(np.log(g) == -1):
                
                minDCF_values_g1.append(minDCF)
                DCF_values_g1.append(DCF)
    


    # cancello il grafico precedente
    plt.clf()
    
    plt.xscale("log", base=10)
    # Tracciare il grafico

    plt.plot(effPriorLogOdds, DCF_values_g4, label='DCF $\gamma: e^{-4}$', color='r')
    plt.plot(effPriorLogOdds, DCF_values_g3, label='DCF $\gamma: e^{-3}$', color='g')
    plt.plot(effPriorLogOdds, DCF_values_g2, label='DCF $\gamma: e^{-2}$', color='b')
    plt.plot(effPriorLogOdds, DCF_values_g1, label='DCF $\gamma: e^{-1}$', color='y')



    plt.plot(effPriorLogOdds, minDCF_values_g4, label='minDCF $\gamma: e^{-4}$', color='#FF6347')  # Tomato
    plt.plot(effPriorLogOdds, minDCF_values_g3, label='minDCF $\gamma: e^{-3}$', color='#32CD32')  # LimeGreen
    plt.plot(effPriorLogOdds, minDCF_values_g2, label='minDCF $\gamma: e^{-2}$', color='#1E90FF')  # DodgerBlue
    plt.plot(effPriorLogOdds, minDCF_values_g1, label='minDCF $\gamma: e^{-1}$', color='#FFD700')  # Gold



    plt.ylim([0, 1.1])
    #plt.xlim([-4, 4])

    plt.xlabel('prior log-odds')
    plt.ylabel('DCF - MinDCF')
    plt.title('Bayes Error Plot: '+title)
    plt.legend()
    # plt.grid(True)
    path = 'plot/plot_svm/svm_RBF/bayes_error_plot/svm_'+title+'.pdf'
    #plt.savefig(path)
    plt.show()




    
def svm_linear_eval(C,DTR,LTR,DVAL,LVAL,pi,title=''):

    
    DCF_values = []
    DCF_values2 = []
    DCF_values3 = []

    minDCF_values = []
    minDCF_values2 = []
    minDCF_values3 = []

    for c in tqdm(C):   
        
        K =1
        
        w, b = train_dual_SVM_linear(DTR, LTR, c, K)
        SVAL = (vrow(w) @ DVAL + b).ravel()
        PVAL = (SVAL > 0) * 1
        err = (PVAL != LVAL).sum() / float(LVAL.size)

        minDCF   = compute_minDCF(DVAL, LVAL, SVAL, pi, 1, 1)
        minDCF_2 = compute_minDCF(DVAL, LVAL, SVAL, 0.5, 1, 1)
        minDCF_3 = compute_minDCF(DVAL, LVAL, SVAL, 0.9, 1, 1)


        DCF = compute_actDCF(DVAL, LVAL, SVAL, pi, 1, 1)
        DCF_2 = compute_actDCF(DVAL, LVAL, SVAL, 0.5, 1, 1)
        DCF_3 = compute_actDCF(DVAL, LVAL, SVAL, 0.9, 1, 1)

        minDCF_values.append(minDCF)
        minDCF_values2.append(minDCF_2)
        minDCF_values3.append(minDCF_3)

        DCF_values.append(DCF)
        DCF_values2.append(DCF_2)
        DCF_values3.append(DCF_3)


    print("\nDCF "+title+" computed! ")
    print("minDCF: %.4f" % min(minDCF_values))
    print("DCF: %.4f" % min(DCF_values))

    path = 'plot/plot_svm/svm_linear/bayes_error_plot/svm_'+title+'.pdf'
    bayes_error_plot_2(C,DCF_values,minDCF_values, title)

    #plot_3DCF(C,DCF_values,DCF_values2,DCF_values3, title)
    #plot_3minDCF(C,minDCF_values,minDCF_values2,minDCF_values3, title)
    





def svm_polynomial_eval(C,DTR,LTR,DVAL,LVAL,pi,title=''):

    
    DCF_values = []
    DCF_values2 = []
    DCF_values3 = []

    minDCF_values = []
    minDCF_values2 = []
    minDCF_values3 = []

    K =1
        
    kernelFunc = polyKernel(2, 1)
    eps = 0
    
    for c in tqdm(C):   
        
        fScore = train_dual_SVM_kernel(DTR, LTR, c, kernelFunc, eps)
        SVAL = fScore(DVAL)


        minDCF= compute_minDCF(DVAL, LVAL, SVAL, pi, 1, 1)
        minDCF_2 = compute_minDCF(DVAL, LVAL, SVAL, 0.5, 1, 1)
        minDCF_3 = compute_minDCF(DVAL, LVAL, SVAL, 0.9, 1, 1)


        DCF = compute_actDCF(DVAL, LVAL, SVAL, pi, 1, 1)
        DCF_2 = compute_actDCF(DVAL, LVAL, SVAL, 0.5, 1, 1)
        DCF_3 = compute_actDCF(DVAL, LVAL, SVAL, 0.9, 1, 1)

        minDCF_values.append(minDCF)
        minDCF_values2.append(minDCF_2)
        minDCF_values3.append(minDCF_3)

        DCF_values.append(DCF)
        DCF_values2.append(DCF_2)
        DCF_values3.append(DCF_3)


    print("\nDCF "+title+" computed! ")
    print("\tminDCF: %.4f" % min(minDCF_values))
    print("\tDCF: %.4f" % min(DCF_values))

    #path = 'plot/plot_svm/svm_polynomial/bayes_error_plot/svm_'+title+'.pdf'
    bayes_error_plot_2(C,DCF_values,minDCF_values, title)

    #plot_3DCF(C,DCF_values,DCF_values2,DCF_values3, title)
    #plot_3minDCF(C,minDCF_values,minDCF_values2,minDCF_values3, title)










def svm_RBF_eval(C,gamma,DTR,LTR,DVAL,LVAL,pi,title=''):

    
    DCF_values = []
    DCF_values2 = []
    DCF_values3 = []

    minDCF_values = []
    minDCF_values2 = []
    minDCF_values3 = []

    min =100
    minAct =0
    c_min = 0

    for c in tqdm(C):   
        
        K =1
        
        kernelFunc = rbfKernel(gamma)
        eps = 1
        fScore = train_dual_SVM_kernel(DTR, LTR, c, kernelFunc, eps)
        SVAL = fScore(DVAL)


        minDCF = compute_minDCF(DVAL, LVAL, SVAL, pi, 1, 1)
        minDCF_2 = compute_minDCF(DVAL, LVAL, SVAL, 0.5, 1, 1)
        minDCF_3= compute_minDCF(DVAL, LVAL, SVAL, 0.9, 1, 1)


        DCF = compute_actDCF(DVAL, LVAL, SVAL, pi, 1, 1)
        DCF_2 = compute_actDCF(DVAL, LVAL, SVAL, 0.5, 1, 1)
        DCF_3 = compute_actDCF(DVAL, LVAL, SVAL, 0.9, 1, 1)

        minDCF_values.append(minDCF)
        minDCF_values2.append(minDCF_2)
        minDCF_values3.append(minDCF_3)

        DCF_values.append(DCF)
        DCF_values2.append(DCF_2)
        DCF_values3.append(DCF_3)

        if minDCF < min:
            min = minDCF
            minAct = DCF
            c_min = c

    title = 'RBF_gamma=%d' % numpy.log(gamma)

    

    #path = 'plot/plot_svm/svm_RBF/bayes_error_plot/svm_'+title+'.pdf'
    #bayes_error_plot_2(C,DCF_values,minDCF_values, title)

    #plot_3DCF(C,DCF_values,DCF_values2,DCF_values3, title)
    #plot_3minDCF(C,minDCF_values,minDCF_values2,minDCF_values3, title)
    return min,minAct,c_min





if __name__ == '__main__':

    
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load(data_path)
    m = 6


    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    C = np.logspace(-5, 0, 11)
    pi = 0.1

    
    print()
    print("SVM: \n")
    svm_linear_eval(C,DTR,LTR,DVAL,LVAL,pi,'Linear')



    #centered
    mu = DTR.mean()
    DTR_mean = DTR - mu
    DVAL_mean = DVAL - mu

    print()
    print("SVM: Linear Centered \n")
    svm_linear_eval(C,DTR_mean,LTR,DVAL_mean,LVAL,pi,'Linear Centered')



    # # SVM Poly
    print()
    print("SVM: Polynomial \n")
    svm_polynomial_eval(C,DTR,LTR,DVAL,LVAL,pi,'Polynomial')


    #SVM RBK
    C2 =numpy.logspace(-3, 2, 11)
    print()
    print("SVM: RBF \n")
    for gamma in [np.exp(-2),np.exp(-4), np.exp(-3), np.exp(-1) ]:
        print("gamma: ", gamma)
        minDCF,actDCF,c = svm_RBF_eval(C2,gamma,DTR,LTR,DVAL,LVAL,pi)
        
        print("\nDCF computed! ")
        print("\tminDCF: %.4f" % (minDCF))
        print("\tDCF: %.4f" % (actDCF))
        print("\tc: %f" % (c))

    plot_4DCF_MinDCF(C2,pi,"RBF")



