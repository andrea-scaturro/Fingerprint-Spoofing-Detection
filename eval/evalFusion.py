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
from src.gmm import *
from src.score_calibration import *
from src.fusion import *



# ======================= Main Program ==========================

if __name__ == '__main__':

    D, L = load(data_path)
    m = 6

    # DTR and LTR are model training data and labels
    # DVAL and LVAL are validation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # print()
    # print("Fusion: Gmm - SVM - LogReg\n")
    # fusion_all(DTR,LTR,DVAL,LVAL)

    pT=0.1

    gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], 8, type = 'diagonal', psiEig = 0.01)
    gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], 32, type = 'diagonal', psiEig = 0.01)
                        
    SLLR_gmm = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
        

    gamma = np.exp(-2)
    kernelFunc = rbfKernel(gamma)
    
    fScore = train_dual_SVM_kernel(DTR, LTR, 31.622776601683793, kernelFunc, 1)
    SLLR_svm = fScore(DVAL)
        
        

    DTR_quadratic  = quadratic(DTR)
    DVAL_quadratic = quadratic(DVAL)
                        
    params_withgrad = trainLogReg_sol1_withgrad_weight(DTR_quadratic, LTR,0.03162277660168379, pT)  
    w = params_withgrad[0][0:-1]
    b = params_withgrad[0][-1]

    SLLR_logReg =  (w.T @ DVAL_quadratic + b - numpy.log(pT / (1-pT))).ravel()

    print()
    print("Fusion: Gmm - SVM - LogReg\n")
    fusion_all2(DVAL,LVAL,SLLR_gmm,SLLR_svm,SLLR_logReg)

    print()
    print("Fusion: Gmm - SVM\n")
    fusion_2Classify(DTR,LTR,DVAL,LVAL,SLLR_gmm,SLLR_svm)

    print()
    print("Fusion: Gmm - LogReg\n")
    fusion_2Classify(DTR,LTR,DVAL,LVAL,SLLR_gmm,SLLR_logReg)

    print()
    print("Fusion: SVM - LogReg\n")
    fusion_2Classify(DTR,LTR,DVAL,LVAL,SLLR_svm,SLLR_logReg)
    print()
    
    actDCF_values,minDCF_values = fusion_all_plot(DTR,LTR,DVAL,LVAL)

    
    title =  "Fusion: Gmm - SVM"
    actDCF_valuesGMM_SVM,minDCF_valuesGMM_SVM =fusion_2Classify_plot(DTR,LTR,DVAL,LVAL,SLLR_gmm,SLLR_svm,title)

    
    title ="Fusion: Gmm - LogReg"
    actDCF_valuesGMM_Log,minDCF_valuesGMM_Log =fusion_2Classify_plot(DTR,LTR,DVAL,LVAL,SLLR_gmm,SLLR_logReg,title)


    title = "Fusion: SVM - LogReg"
    actDCF_valuesSVM_Log,minDCF_valuesSVM_Log =fusion_2Classify_plot(DTR,LTR,DVAL,LVAL,SLLR_svm,SLLR_logReg,title)


    plt.clf()
    colors =    ['#8B0000', '#006400', '#00008B', '#FFD700', '#8B008B'] 
    effPriorLogOdds = np.linspace(-3, 3, 21)
    plt.plot(effPriorLogOdds, actDCF_values, label='DCF all',linestyle='-',  color=colors[0])
    plt.plot(effPriorLogOdds, minDCF_values, label='min DCF all', linestyle='--',color=colors[0])

    plt.plot(effPriorLogOdds, actDCF_valuesGMM_SVM, label='DCF GMM-SVM',linestyle='-',  color=colors[1])
    plt.plot(effPriorLogOdds, minDCF_valuesGMM_SVM, label='min DCF GMM-SVM', linestyle='--',color=colors[1])

    plt.plot(effPriorLogOdds, actDCF_valuesGMM_Log, label='DCF GMM-Log',linestyle='-',  color=colors[2])
    plt.plot(effPriorLogOdds, minDCF_valuesGMM_Log, label='min DCF GMM-Log', linestyle='--',color=colors[2])

    plt.plot(effPriorLogOdds, actDCF_valuesSVM_Log, label='DCF SVM-Log',linestyle='-',  color=colors[3])
    plt.plot(effPriorLogOdds, minDCF_valuesSVM_Log, label='min DCF SVM-Log', linestyle='--',color=colors[3])
   
    #plt.ylim([0, 1.1])
    plt.ylim([0, 0.8])
    plt.xlim([-3, 3])

    plt.xlabel('priorLogOdds')
    plt.ylabel('DCF Value')
    plt.title('Bayes Error Plot: Fusion')
    plt.legend()
    #plt.grid(True)
    

    plt.savefig('plot/plot_fusion/Compare.pdf')

    #plt.show()
