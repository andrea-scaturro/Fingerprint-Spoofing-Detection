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


def logistic_regression_eval(lamda,DTR,LTR,DVAL,LVAL,pi,title=''):

    
    minDCF_values = []
    DCF_values = []

    DCF_values2 = []
    DCF_values3 = []

    minDCF_values2 = []
    minDCF_values3 = []

    l_min =0
    minDcfPrint = 10
    minActDCFPrint =0
    
    for l in tqdm(lamda):    

        params_withgrad = trainLogReg_sol2_withgrad(DTR, LTR, l)
        
        w = params_withgrad[0][0:-1]
        b = params_withgrad[0][-1]
        sllr_1 = sllr_actual(DTR, LTR, DVAL, LVAL, w, b)
        sllr_2 = sllr_min(DVAL, LVAL, w, b, pi)

        minDCF= compute_minDCF(DVAL, LVAL, vcol(sllr_2), pi, 1, 1)
        minDCF_2 = compute_minDCF(DVAL, LVAL, vcol(sllr_2), 0.5, 1, 1)
        minDCF_3 =compute_minDCF(DVAL, LVAL, vcol(sllr_2), 0.9, 1, 1)


        DCF = compute_actDCF(DVAL, LVAL, vcol(sllr_1), pi, 1, 1)
        DCF_2 = compute_actDCF(DVAL, LVAL, vcol(sllr_1), 0.5, 1, 1)
        DCF_3 = compute_actDCF(DVAL, LVAL, vcol(sllr_1), 0.9, 1, 1)


        minDCF_values.append(minDCF)
        minDCF_values2.append(minDCF_2)
        minDCF_values3.append(minDCF_3)


        DCF_values.append(DCF)
        DCF_values2.append(DCF_2)
        DCF_values3.append(DCF_3)

        if minDCF < minDcfPrint:
            minDcfPrint = minDCF
            l_min = l
            minActDCFPrint = DCF


    
    # plot

    #bayes_error_plot_2(lamda,DCF_values, minDCF_values,title)
    
    
    #plot_3DCF(lamda,DCF_values,DCF_values2,DCF_values3,title)
    #plot_3minDCF(lamda,minDCF_values,minDCF_values2,minDCF_values3,title)
    
    return minDcfPrint,minActDCFPrint,l_min



    
def logistic_regression_weighted_eval(lamda,DTR,LTR,DVAL,LVAL,pi,title=''):

    
    DCF_values = []
    DCF_values2 = []
    DCF_values3 = []

    minDCF_values = []
    minDCF_values2 = []
    minDCF_values3 = []

    l_min =0
    minDcfPrint = 10
    minActDCFPrint =0

    for l in tqdm(lamda):    
        
        params_withgrad = trainLogReg_sol1_withgrad_weight(DTR, LTR, l, pi)

        w = params_withgrad[0][0:-1]
        b = params_withgrad[0][-1]
        sllr_1 = sllr_weight(DVAL, LVAL, w, b, pi)

        minDCF = compute_minDCF(DVAL, LVAL, vcol(sllr_1), pi, 1, 1)
        minDCF_2= compute_minDCF(DVAL, LVAL, vcol(sllr_1), 0.5, 1, 1)
        minDCF_3 = compute_minDCF(DVAL, LVAL, vcol(sllr_1), 0.9, 1, 1)


        DCF = compute_actDCF(DVAL, LVAL, vcol(sllr_1), pi, 1, 1)
        DCF_2 = compute_actDCF(DVAL, LVAL, vcol(sllr_1), 0.5, 1, 1)
        DCF_3 = compute_actDCF(DVAL, LVAL, vcol(sllr_1), 0.9, 1, 1)

        minDCF_values.append(minDCF)
        minDCF_values2.append(minDCF_2)
        minDCF_values3.append(minDCF_3)

        DCF_values.append(DCF)
        DCF_values2.append(DCF_2)
        DCF_values3.append(DCF_3)

        if minDCF < minDcfPrint:
            minDcfPrint = minDCF
            l_min = l
            minActDCFPrint = DCF



    #bayes_error_plot_2(lamda,DCF_values,minDCF_values, 'Weighted')
    #plot_3DCF(lamda,DCF_values,DCF_values2,DCF_values3, title)
    #plot_3minDCF(lamda,DCF_values,DCF_values2,DCF_values3, title)
    
    return minDcfPrint,minActDCFPrint,l_min

if __name__ == '__main__':


    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load(data_path)
    m = 6

    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    lamda = np.logspace(-4, 2, 13)
    pi = 0.1


    
    

    print()
    print("Logistic Regression: \n")
    minDCF, dcf,l = logistic_regression_eval(lamda,DTR,LTR,DVAL,LVAL,pi)

    print("\nDCF computed! ")
    print("minDCF: %.4f" % minDCF)
    print("DCF: %.4f" % dcf)
    print("l: %.4f" % l)


    # FOR 50 SAMPLE

    DTR_50, LTR_50 = DTR[:, ::50], LTR[::50]

    print()
    print("Logistic Regression: 50 Sample \n")
    minDCF, dcf,l = logistic_regression_eval(lamda,DTR_50,LTR_50,DVAL,LVAL,pi,'50 SAMPLE')

    print("\nDCF 50 SAMPLE computed! ")
    print("minDCF: %.4f" % minDCF)
    print("DCF: %.4f" % dcf)
    print("l: %.4f" % l)


    # WEIGHTED

    print()
    print("Logistic Regression: Weighted\n")
    minDCF, dcf,l = logistic_regression_weighted_eval(lamda,DTR,LTR,DVAL,LVAL,pi,'WEIGHTED')

    print("\nDCF WEIGHTED computed! ")
    print("minDCF: %.4f" % minDCF)
    print("DCF: %.4f" % dcf)
    print("l: %.4f" % l)
    


    # CENTERED

    mu = DTR.mean()
    DTR_mean = DTR - mu
    DVAL_mean = DVAL - mu

    print()
    print("Logistic Regression: Centered\n")
    minDCF, dcf,l = logistic_regression_eval(lamda,DTR_mean,LTR,DVAL_mean,LVAL,pi,'CENTERED')

    print("\nDCF CENTERED computed! ")
    print("minDCF: %.4f" % minDCF)
    print("DCF: %.4f" % dcf)
    print("l: %.4f" % l)


    # Z-Norm

    means = vcol(np.mean(DTR, axis=1))
    stds = vcol(np.std(DTR, axis=1))

    DTR_m = (DTR - means) / stds
    DVAL_m = (DVAL - means) / stds

    print()
    print("Logistic Regression: Z-Norm\n")
    lminDCF, dcf,l = logistic_regression_eval(lamda,DTR_m,LTR,DVAL_m,LVAL,pi,'Z-Norm')

    print("\nDCF Z-Norm computed! ")
    print("minDCF: %.4f" % minDCF)
    print("DCF: %.4f" % dcf)
    print("l: %.4f" % l)


    # PCA m=5

    apply_PCA = PCA(DTR, 5)
    DTR_pca = numpy.dot(apply_PCA.T, DTR)

    apply_PCA = PCA(DVAL, 5)
    DVAL_pca = numpy.dot(apply_PCA.T, DVAL)
    print()
    print("Logistic Regression: PCA\n")
    minDCF, dcf,l = logistic_regression_eval(lamda,DTR_pca,LTR,DVAL_pca,LVAL,pi,'PCA')

    print("\nDCF PCA computed! ")
    print("minDCF: %.4f" % minDCF)
    print("DCF: %.4f" % dcf)
    print("l: %.4f" % l)


     # QUADRATIC
    
    DTR_quadratic = quadratic(DTR)
    DVAL_quadratic = quadratic(DVAL)
    print()
    print("Logistic Regression: Quadratic \n")
    minDCF, dcf,l = logistic_regression_eval(lamda,DTR_quadratic,LTR,DVAL_quadratic,LVAL,pi,'QUADRATIC')

    print("\nDCF QUADRATIC computed! ")
    print("minDCF: %.4f" % minDCF)
    print("DCF: %.4f" % dcf)
    print("l: %.4f" % l)



    # CENTERED + QUADRATIC

    mu = DTR.mean()
    DTR_mean = DTR - mu
    DVAL_mean = DVAL - mu
    DTR_quadratic = quadratic(DTR_mean)
    DVAL_quadratic = quadratic(DVAL_mean)
    print()
    print("Logistic Regression: Centerd + Quadratic \n")
    minDCF, dcf,l = logistic_regression_eval(lamda,DTR_quadratic,LTR,DVAL_quadratic,LVAL,pi,'CENTERED+QUADRATIC')

    print("\nDCF CENTERED+QUADRATIC' computed! ")
    print("minDCF: %.4f" % minDCF)
    print("DCF: %.4f" % dcf)
    print("l: %.4f" % l)    


    # Z-Norm + QUADRATIC

    means = vcol(np.mean(DTR, axis=1))
    stds = vcol(np.std(DTR, axis=1))

    DTR_m = (DTR - means) / stds
    DVAL_m = (DVAL - means) / stds
    DTR_quadratic = quadratic(DTR_m)
    DVAL_quadratic = quadratic(DVAL_m)

    print()
    print("Logistic Regression: Z-Norm + Quadratic \n")
    lminDCF, dcf,l = logistic_regression_eval(lamda,DTR_quadratic,LTR,DVAL_quadratic,LVAL,pi,'Z-Norm + QUADRATIC')

    print("\nDCF Z-Norm + QUADRATIC computed! ")
    print("minDCF: %.4f" % minDCF)
    print("DCF: %.4f" % dcf)
    print("l: %.4f" % l)


    # PCA m=5 + QUADRATIC
    
    apply_PCA = PCA(DTR, 5)
    DTR_pca = numpy.dot(apply_PCA.T, DTR)

    apply_PCA = PCA(DVAL, 5)
    DVAL_pca = numpy.dot(apply_PCA.T, DVAL)

    DTR_quadratic = quadratic(DTR_pca)
    DVAL_quadratic = quadratic(DVAL_pca)

    print()
    print("Logistic Regression: PCA + Quadratic\n")
    minDCF, dcf,l = logistic_regression_eval(lamda,DTR_quadratic,LTR,DVAL_quadratic,LVAL,pi,'PCA+QUADRATIC')

    print("\nDCF PCA+QUADRATIC computed! ")
    print("minDCF: %.4f" % minDCF)
    print("DCF: %.4f" % dcf)
    print("l: %.4f" % l)


    


# stats

    t = compute_threshold()
    _, _, llr = mvg(DTR, LTR, DVAL, LVAL, t)
    _, _, llr_tied = mvg_tied(DTR, LTR, DVAL, LVAL, t)
    _, _, llr_nb = mvg_nb(DTR, LTR, DVAL, LVAL, t)

    minDCF_g = compute_minDCF(DVAL, LVAL, llr, pi, 1, 1)
    minDCF_t= compute_minDCF(DVAL, LVAL, llr_tied, pi, 1, 1)
    minDCF_nb = compute_minDCF(DVAL, LVAL, llr_nb, pi, 1, 1)

    actDCF_g  = compute_actDCF(DVAL, LVAL, llr, pi, 1, 1)
    actDCF_t  = compute_actDCF(DVAL, LVAL, llr_tied, pi, 1, 1)
    actDCF_nb = compute_actDCF(DVAL, LVAL, llr_nb, pi, 1, 1)

    print("\nGaussian Classifier:")
    print("minDCF: %.4f" % (minDCF_g))
    print("actDCF: %.4f" % (actDCF_g))

    print("minDCF_tied: %.4f" % (minDCF_t))
    print("actDCF_tied: %.4f" % (actDCF_t))

    print("minDCF_naive_bayes: %.4f" % (minDCF_nb))
    print("actDCF_naive_bayes: %.4f" % (actDCF_nb))



