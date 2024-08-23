import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm
import sys
import os


from src.bayes_decision import *
from src.classifier_pca_lda import *
from dataset.dataset import *
from src.dimreduction import *
from src.gaussian_analysis import *
from src.mvg import *
from src.logistic_regression import *
from src.plotter import *
from src.svm import *
from src.gmm import *
from src.score_calibration import *
from src.stats import *
from src.fusion import *
from eval.evalLogistic_regression import *




class Application:

    def __init__(self, pi, Cfn, Cfp):
        self.pi = pi
        self.Cfn = Cfn
        self.Cfp = Cfp


applications_data = [
        (0.1, 1.0, 1.0),
        (0.9, 1.0, 1.0),
        (0.5, 1.0, 1.0),
        (0.1, 1.0, 9.0),
        (0.1, 9.0, 1.0)
        ]

applications = [Application(pi, Cfn, Cfp) for pi, Cfn, Cfp in applications_data]



def fusion_allC(LVAL, SLLR_gmm,SLLR_svm,SLLR_logReg):    

    minActDCF = 1
    minimumDCF = 1
    pT = 0.5

    KFOLD = 5    
                
   
    fusedScores = [] 
    fusedLabels = [] 
        
    # Train KFOLD times the fusion model
    for i in range(KFOLD):
              
            
            SCAL1, SVAL1 = extract_train_val_folds_from_ary(SLLR_gmm, i)
            SCAL2, SVAL2 = extract_train_val_folds_from_ary(SLLR_svm, i)
            SCAL3, SVAL3 = extract_train_val_folds_from_ary(SLLR_logReg, i)
            

            LCAL, LLVAL = extract_train_val_folds_from_ary(LVAL, i)

            
            SCAL = numpy.vstack([SCAL1, SCAL2,SCAL3])
            
            params = trainLogReg_sol1_withgrad_weight((SCAL), LCAL, 0, pT)
            w = params[0][0:-1]
            b = params[0][-1]
            
            
            # Build the validation scores "feature" matrix
            SVAL = numpy.vstack([SVAL1, SVAL2,SVAL3])
            
            # Apply the model to the validation fold
            calibrated_SVAL =  (w.T @ SVAL + b - numpy.log(pT / (1-pT))).ravel()

            
            fusedScores.append(calibrated_SVAL)
            fusedLabels.append(LLVAL)

    
    fusedScores = numpy.hstack(fusedScores)
    fusedLabels = numpy.hstack(fusedLabels)
        
    return fusedScores,fusedLabels


def evalFusion_application(DVAL,LVAL,llr):

    plt.clf()  
    effPriorLogOdds = np.linspace(-3, 3, 21)
    colors =    ['#8B0000', '#006400', '#00008B', '#FFD700', '#8B008B'] 
    
    

    for i in range(len(applications)):

        dcf_values = []
        min_dcf_values = []

        for p in effPriorLogOdds:
            pi = 1 / (1 + np.exp(-p))
            dcf = compute_actDCF(DVAL,LVAL,llr,pi, applications[i].Cfn, applications[i].Cfp)
            min_dcf = compute_minDCF(DVAL,LVAL,llr,pi,applications[i].Cfn, applications[i].Cfp)

            dcf_values.append(dcf)
            min_dcf_values.append(min_dcf)

    
        plt.plot(effPriorLogOdds, dcf_values, label=f'Application : {i+1} DCF ',linestyle='-' ,color=colors[i])
        plt.plot(effPriorLogOdds, min_dcf_values, label=f'Application : {i+1} min DCF', linestyle='--',color=colors[i])
    
    plt.ylim([0, 0.8])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF Value')
    plt.title('Compare Applications')
    plt.legend()
    #plt.grid(True)
    plt.savefig('plot/plot_eval/compareApplication.pdf')

    plt.show()



def fusion_2Classify(LVAL,sllr1,sllr2):    


    KFOLD = 5
    pT  =0.5

    fusedScores = [] 
    fusedLabels = [] 


    #  Train KFOLD times the fusion model
    for foldIdx in range(KFOLD):
    # keep 1 fold for validation, use the remaining ones for training       
            
        SCAL1, SVAL1 = extract_train_val_folds_from_ary(sllr1, foldIdx)
        SCAL2, SVAL2 = extract_train_val_folds_from_ary(sllr2, foldIdx)
            

        LCAL, LLVAL = extract_train_val_folds_from_ary(LVAL, foldIdx)

            
        SCAL = numpy.vstack([SCAL1, SCAL2])
            
        params = trainLogReg_sol1_withgrad_weight((SCAL), LCAL, 0, pT)
        w = params[0][0:-1]
        b = params[0][-1]
        
        SVAL = numpy.vstack([SVAL1, SVAL2])    
        calibrated_SVAL =  (w.T @ SVAL + b - numpy.log(pT / (1-pT))).ravel()

        fusedScores.append(calibrated_SVAL)
        fusedLabels.append(LLVAL)


    fusedScores = numpy.hstack(fusedScores)
    fusedLabels = numpy.hstack(fusedLabels)


    return fusedScores,fusedLabels




def computeErrorPlot(DVAL,LVAL,llr1,llr2,llr3 ,title='',path=''):
     
    effPriorLogOdds = np.linspace(-3, 3, 21)
    
    # Array per memorizzare i valori DCF e min DCF
    dcf_values1 = []
    dcf_values3 = []
    dcf_values2 = []

    mindcf_values1 = []
    mindcf_values3 = []
    mindcf_values2 = []

    # Calcolo dei valori DCF e min DCF per ciascun prior efficace
    for p in effPriorLogOdds:
        pi = 1 / (1 + np.exp(-p))

        dcf1 = compute_actDCF(DVAL,LVAL,llr1,pi, 1, 1)
        dcf2 = compute_actDCF(DVAL,LVAL,llr2,pi, 1, 1)
        dcf3 = compute_actDCF(DVAL,LVAL,llr3,pi, 1, 1)

        mindcf1 = compute_minDCF(DVAL,LVAL,llr1,pi, 1, 1)
        mindcf2 = compute_minDCF(DVAL,LVAL,llr2,pi, 1, 1)
        mindcf3 = compute_minDCF(DVAL,LVAL,llr3,pi, 1, 1)
        

        dcf_values1.append(dcf1)
        dcf_values2.append(dcf2)
        dcf_values3.append(dcf3)

        mindcf_values1.append(mindcf1)
        mindcf_values2.append(mindcf2)
        mindcf_values3.append(mindcf3)


    # Tracciare il grafico


    plt.plot(effPriorLogOdds, mindcf_values1, label='minDCF_GMM',linestyle='--', color='r')
    plt.plot(effPriorLogOdds, mindcf_values2, label='minDCF_SVM',linestyle='--', color='b')
    plt.plot(effPriorLogOdds, mindcf_values3, label='minDCF_Log',linestyle='--' ,color='g')

    plt.plot(effPriorLogOdds, dcf_values1, label='DCF_GMM ',linestyle='-', color='r')
    plt.plot(effPriorLogOdds, dcf_values2, label='DCF_SVM', linestyle='-',color='b')
    plt.plot(effPriorLogOdds, dcf_values3, label='DCF_Log', linestyle='-',color='g')


    plt.ylim([0, 0.8])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF Value')
    plt.title(title)
    plt.legend()
    #plt.grid(True)

    if path != '':

        plt.savefig(path)

    plt.show()



def evaluateGMM(DTR,LTR,DVAL,LVAL):
     
    for type in ['full', 'diagonal']:

        print()
        print(type.upper(),":\n")

        for numC in [1,2,4,8,16,32]:

            for numC2 in [1,2,4,8,16,32]:

                gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], numC, type =type, psiEig = 0.01)
                gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], numC2, type = type, psiEig = 0.01)
            

                SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
            
            
                minDCF=compute_minDCF(DVAL,LVAL,SLLR,0.1)
                dcf = compute_actDCF(DVAL,LVAL,SLLR,0.1)
                
                print ('numC = %d   numC2 = %d     minDCF = %.4f  DCF = %.4f' % (numC, numC2 ,minDCF, dcf))



         

def evaluateSVMLinear(DTR,LTR,DVAL,LVAL):

    DCF_values = []
    minDCF_values = []
 

    min = 10
    dcf =0
    cMin =0

    C = np.logspace(-5, 2, 11)

    for c in tqdm(C):   
        
        K =1
        
        w, b = train_dual_SVM_linear(DTR, LTR, c, K)
        SVAL = (vrow(w) @ DVAL + b).ravel()
        

        minDCF   = compute_minDCF(DVAL, LVAL, SVAL, pi, 1, 1)
        DCF = compute_actDCF(DVAL, LVAL, SVAL, pi, 1, 1)

        minDCF_values.append(minDCF)
        DCF_values.append(DCF)
       
        if minDCF < min:
            min = minDCF
            dcf = DCF
            cMin = c
    
    return min,dcf,cMin




def evaluateSVMPoly(DTR,LTR,DVAL,LVAL):

    DCF_values = []
    minDCF_values = []
    K =1

    min = 10
    dcf =0
    cMin =0


    C = np.logspace(-5, 2, 11)
        
    kernelFunc = polyKernel(2, 1)
    eps = 1
    
    for c in tqdm(C):   
        
        fScore = train_dual_SVM_kernel(DTR, LTR, c, kernelFunc, eps)
        SVAL = fScore(DVAL)

        minDCF= compute_minDCF(DVAL, LVAL, SVAL, pi, 1, 1)
        DCF = compute_actDCF(DVAL, LVAL, SVAL, pi, 1, 1)
    
        minDCF_values.append(minDCF)
        DCF_values.append(DCF)
    
        if minDCF < min:
            min = minDCF
            dcf = DCF
            cMin = c
    
    return min,dcf,cMin



def evaluateSVMRBF(DTR,LTR,DVAL,LVAL):

    DCF_values = []
    minDCF_values = []
  
    min =100
    minAct =0
    c_min = 0
    g =0

    C = np.logspace(-5, 2, 11)

    for gamma in [np.exp(-2),np.exp(-4), np.exp(-3), np.exp(-1) ]:
        for c in tqdm(C):   
            
            K =1
            
            kernelFunc = rbfKernel(gamma)
            eps = 1
            fScore = train_dual_SVM_kernel(DTR, LTR, c, kernelFunc, eps)
            SVAL = fScore(DVAL)


            minDCF = compute_minDCF(DVAL, LVAL, SVAL, pi, 1, 1)
            DCF = compute_actDCF(DVAL, LVAL, SVAL, pi, 1, 1)
    

            minDCF_values.append(minDCF)
            DCF_values.append(DCF)

            if minDCF < min:
                min = minDCF
                minAct = DCF
                c_min = c
                g = gamma

    
        return min,minAct,c_min,g


def evaluateSVM(DTR,LTR,DVAL,LVAL):

    minDCF_lin,DCF_lin,C_lin = evaluateSVMLinear(DTR,LTR,DVAL,LVAL)
    minDCF_poly,DCF_poly,C_poly = evaluateSVMPoly(DTR,LTR,DVAL,LVAL)
    minDCF_rbf,DCF_rbf,C_rbf,gamma = evaluateSVMRBF(DTR,LTR,DVAL,LVAL)

    print()
    print('SVM Linear: \n')
    print('\tminDCF: ', minDCF_lin)
    print('\tDCF: ', DCF_lin)
    print('\tC: ', C_lin)

    print()
    print('SVM Polynomial: \n')
    print('\tminDCF: ', minDCF_poly)
    print('\tDCF: ', DCF_poly)
    print('\tC: ', C_poly)

    print()
    print('SVM RBF: \n')
    print('\tminDCF: ', minDCF_rbf)
    print('\tDCF: ', DCF_rbf)
    print('\tC: ', C_rbf)
    print('\tgamma: ', gamma)





def evaluateLogReg(DTR,LTR,DVAL,LVAL):


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



# ======================= Main Program ==========================

if __name__ == '__main__':

    D, L = load('dataset/trainData.txt')
    
    DTE, LTE = load('dataset/evalData.txt')


    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    pi = 0.1 # target application

    # compute scores for the classifiers
    gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], 8, type = 'diagonal', psiEig = 0.01)
    gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], 32, type = 'diagonal', psiEig = 0.01)
                            
    SLLR_gmm = logpdf_GMM(DTE, gmm1) - logpdf_GMM(DTE, gmm0)
            


    gamma = np.exp(-2)
    kernelFunc = rbfKernel(gamma)
        
    fScore = train_dual_SVM_kernel(DTR, LTR, 31.622776601683793, kernelFunc, 1)
    
    SLLR_svm = fScore(DTE)        

    
    DTR_quadratic  = quadratic(DTR)
    DVAL_quadratic = quadratic(DTE)
                            
    params_withgrad = trainLogReg_sol1_withgrad_weight(DTR_quadratic, LTR,0.03162277660168379, pi)  
    w = params_withgrad[0][0:-1]
    b = params_withgrad[0][-1]

    SLLR_logReg =  (w.T @ DVAL_quadratic + b - numpy.log(pi / (1-pi))).ravel()





    fusedScore_allClassisy,fusedLabel_allClassify = fusion_allC(LTE, SLLR_gmm,SLLR_svm,SLLR_logReg)

    
    
    fusedScoreGMM_SVM, fusedLabelGMM_SVM = fusion_2Classify(LTE, SLLR_gmm,SLLR_svm)
    fusedScoreGMM_Log, fusedLabelGMM_Log = fusion_2Classify(LTE, SLLR_gmm,SLLR_logReg)
    fusedScoreSVM_Log, fusedLabelSVM_Log = fusion_2Classify(LTE, SLLR_svm,SLLR_logReg)
    

    #eval fusion of the 3 classifier 
    print()
    print('Fusion All Classify\n')
    print('\tminDCF: ' , compute_minDCF(DTE,fusedLabel_allClassify,fusedScore_allClassisy,pi))
    print('\tactDCF: ' , compute_actDCF(DTE,fusedLabel_allClassify,fusedScore_allClassisy,pi))

    path = 'plot/plot_eval/Valuation Fuse A'
    bayes_error_plot(DTE,fusedLabel_allClassify,fusedScore_allClassisy,'Valuation Fuse A',path)
    evalFusion_application(DTE,fusedLabel_allClassify,fusedScore_allClassisy)

    # eval GMM - SVM

    print()
    print('Fusion GMM - SVM \n')
    
    print('\tactDCF: ' , compute_actDCF(DTE,fusedLabelGMM_SVM,fusedScoreGMM_SVM,pi))

    # eval GMM - Logistc Regression
    print()
    print('Fusion GMM - Logistic Regression \n')
    
    print('\tactDCF: ' , compute_actDCF(DTE,fusedLabelGMM_Log,fusedScoreGMM_Log,pi))

    # eval SVM - Logistc Regression
    print()
    print('Fusion SVM - Logistic Regression \n')
    
    print('\tactDCF: ' , compute_actDCF(DTE,fusedLabelSVM_Log,fusedScoreSVM_Log,pi))

    path = 'plot/plot_eval/Valuation Fuse 2'
    bayes_error_plotDCF(DTE,fusedLabel_allClassify,fusedScoreGMM_SVM,fusedScoreGMM_Log,fusedScoreSVM_Log,fusedScore_allClassisy,'Valuation Fuse 2',path)


    scoreCalibrateGMM, labelsCalibrateGMM = scoreCalibration(DTR,LTR,DTE,LTE,'gmm')
    scoreCalibrateSVM, labelsCalibrateSVM = scoreCalibration(DTR,LTR,DTE,LTE,'svm')
    scoreCalibrateLogReg, labelsCalibrateLogReg = scoreCalibration(DTR,LTR,DTE,LTE,'logReg')

    print()
    print('Gmm\n')
    print('\tminDCF: ' , compute_minDCF(DTE,labelsCalibrateGMM,scoreCalibrateGMM,pi))
    print('\tactDCF: ' , compute_actDCF(DTE,labelsCalibrateGMM,scoreCalibrateGMM,pi))


    print()
    print('SVM\n')
    print('\tminDCF: ' , compute_minDCF(DTE,labelsCalibrateSVM,scoreCalibrateSVM,pi))
    print('\tactDCF: ' , compute_actDCF(DTE,labelsCalibrateSVM,scoreCalibrateSVM,pi))


    print()
    print('Logistc Regression')
    print('\tminDCF: ' , compute_minDCF(DTE,labelsCalibrateLogReg,scoreCalibrateLogReg,pi))
    print('\tactDCF: ' , compute_actDCF(DTE,labelsCalibrateLogReg,scoreCalibrateLogReg,pi))
    
    path = 'plot/plot_eval/Valuation Classify'
    computeErrorPlot(DTE,LTE,SLLR_gmm,SLLR_svm,SLLR_logReg,'Valuation Classify',path)

    path = 'plot/plot_eval/Valuation ClassifyCal'
    computeErrorPlot(DTE,labelsCalibrateGMM,scoreCalibrateGMM,scoreCalibrateSVM,scoreCalibrateLogReg,'Valuation Classify Calibrate',path)


    print()
    print("Evaluate: SVM")
    evaluateSVM(DTR,LTR,DTE,LTE)

    print()
    print("Evaluate: GMM")
    evaluateGMM(DTR,LTR,DTE,LTE)

    print()
    print("Evaluate: Logistic Regression")
    evaluateLogReg(DTR,LTR,DTE,LTE)