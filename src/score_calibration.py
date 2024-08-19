
import numpy
import matplotlib
import matplotlib.pyplot as plt

from .bayes_decision import *
from .logistic_regression import *
from .gmm import *
from .svm import *



def bayesErrorPlotCalibration(DVAL,LVAL,llr,model,title):

    minDcf_values=[]
    dcf_values=[]
    effPriorLogOdds = numpy.linspace(-3,3,21)

    for e in effPriorLogOdds:

        pi = 1 / (1 + np.exp(-e))
                        
        minDCF= compute_minDCF(DVAL, LVAL, llr,pi)
        dcf = compute_actDCF(DVAL, LVAL, llr, pi)
                
        minDcf_values.append(minDCF)
        dcf_values.append(dcf)


    path= 'plot/plot_scoreCalibration/'+title+'.pdf'
    bayes_error_plot_2(effPriorLogOdds,dcf_values,minDcf_values,title,path)




def extract_train_val_folds_from_ary(X, i, KFOLD):
    return numpy.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != i]), X[i::KFOLD]

def scoreCalibration_findPt(DTR,LTR,DVAL,LVAL, model='gmm'):

    

    m = 10
    a = 20
    p=0
    
    for pT in np.arange(0.1, 1.00, 0.1):
    
        if model == 'gmm':
        
            
            gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], 8, type = 'diagonal', psiEig = 0.01)
            gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], 32, type = 'diagonal', psiEig = 0.01)
                        
            SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)

            minDCF =  compute_minDCF(DVAL,LVAL,SLLR, 0.1, 1.0, 1.0)
            dcf    =  compute_actDCF(DVAL,LVAL,SLLR, 0.1, 1.0, 1.0)
        

        if model == 'svm':

            gamma = np.exp(-2)
            kernelFunc = rbfKernel(gamma)
            eps = 1
            fScore = train_dual_SVM_kernel(DTR, LTR, 31.622776601683793, kernelFunc, eps)
            SLLR = fScore(DVAL)

            minDCF =  compute_minDCF(DVAL,LVAL,SLLR, 0.1, 1.0, 1.0)
            dcf    =  compute_actDCF(DVAL,LVAL,SLLR, 0.1, 1.0, 1.0)
        
        
        if model == 'logReg':

            DTR_quadratic= quadratic(DTR)
            DVAL_quadratic = quadratic(DVAL)
                        
            params_withgrad = trainLogReg_sol1_withgrad_weight(DTR_quadratic, LTR,0.03162277660168379, 0.1)  
            w = params_withgrad[0][0:-1]
            b = params_withgrad[0][-1]

            SLLR =  (w.T @ DVAL_quadratic + b - numpy.log(0.1 / (1-0.1))).ravel()

            minDCF =  compute_minDCF(DVAL_quadratic,LVAL,SLLR, 0.1, 1.0, 1.0)
            dcf    =  compute_actDCF(DVAL_quadratic,LVAL,SLLR, 0.1, 1.0, 1.0)
    
        
        KFOLD = 5

        
        
        

        
        print()
        print(model)
        print()
        print ('minDCF: %.3f' % minDCF) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
        print ('actDCF: %.3f' % dcf)
        
        
        calibrateScores = [] # We will add to the list the scores computed for each fold
        labelsCalibrate = []

        # Train KFOLD times the calibration model
        for i in range(KFOLD):
            
            SCAL, SVAL = extract_train_val_folds_from_ary(SLLR, i,KFOLD)
            LCAL, LLVAL = extract_train_val_folds_from_ary(LVAL, i,KFOLD)

            
            params = trainLogReg_sol1_withgrad_weight(vrow(SCAL), LCAL, 0, pT)
            w = params[0][0:-1]
            b = params[0][-1]
        

            calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
            calibrateScores.append(calibrated_SVAL)
            labelsCalibrate.append(LLVAL)

           

        calibrateScores = numpy.hstack(calibrateScores)
        labelsCalibrate = numpy.hstack(labelsCalibrate)


        if model == 'logReg':

            DTR_quadratic= quadratic(DTR)
            DVAL_quadratic = quadratic(DVAL)

            minDCF_cal = compute_minDCF( DVAL_quadratic,labelsCalibrate,calibrateScores, 0.1, 1.0, 1.0)
            dcf_cal  =  compute_actDCF(DVAL_quadratic,labelsCalibrate,calibrateScores, 0.1, 1.0, 1.0)

        else:

            minDCF_cal = compute_minDCF( DVAL,labelsCalibrate,calibrateScores, 0.1, 1.0, 1.0)
            dcf_cal  =  compute_actDCF(DVAL,labelsCalibrate,calibrateScores, 0.1, 1.0, 1.0)

        print ('minDCF, calibrate: %.3f' % minDCF_cal) 
        print ('actDCF, calibrate: %.3f' % dcf_cal)
        print( ' pt: ',pT)

        if (dcf_cal - minDCF_cal) <( a - m):
                m = minDCF_cal
                a = dcf_cal 
                p=pT

    print ('minDCF, calibrate: %.3f' % minDCF_cal) 
    print ('actDCF, calibrate: %.3f' % dcf_cal)

    print('\t m: ',m)
    print('\t a: ',a)
    print('\t pT: ',p)





def scoreCalibration(DTR,LTR,DVAL,LVAL, model='gmm'):

    
        if model == 'gmm':
        
            pT = 0.8
            gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], 8, type = 'diagonal', psiEig = 0.01)
            gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], 32, type = 'diagonal', psiEig = 0.01)
                        
            SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
        

        if model == 'svm':

            pT = 0.3
            gamma = np.exp(-2)
            kernelFunc = rbfKernel(gamma)
            eps = 1
            fScore = train_dual_SVM_kernel(DTR, LTR, 31.622776601683793, kernelFunc, eps)
            SLLR = fScore(DVAL)
        
        
        if model == 'logReg':

            pT = 0.2
            DTR= quadratic(DTR)
            DVAL = quadratic(DVAL)
                        
            params_withgrad = trainLogReg_sol1_withgrad_weight(DTR, LTR,0.03162277660168379, 0.1)  
            w = params_withgrad[0][0:-1]
            b = params_withgrad[0][-1]

            SLLR =  (w.T @ DVAL + b - numpy.log(0.1 / (1-0.1))).ravel()
        
    
        
        KFOLD = 5

        minDCF =  compute_minDCF(DVAL,LVAL,SLLR, 0.1, 1.0, 1.0)
        dcf    =  compute_actDCF(DVAL,LVAL,SLLR, 0.1, 1.0, 1.0)
            

        
        # print()
        # print(model)
        # print()
        # print ('minDCF: %.3f' % minDCF) 
        # print ('actDCF: %.3f' % dcf)
        
        
        calibrateScores = [] # We will add to the list the scores computed for each fold
        labelsCalibrate = []

        # Train KFOLD times the calibration model
        for i in range(KFOLD):
            
            SCAL, SVAL = extract_train_val_folds_from_ary(SLLR, i,KFOLD)
            LCAL, LLVAL = extract_train_val_folds_from_ary(LVAL, i,KFOLD)

            
            params = trainLogReg_sol1_withgrad_weight(vrow(SCAL), LCAL, 0, pT)
            w = params[0][0:-1]
            b = params[0][-1]
        

            calibrated_SVAL =  (w.T @ vrow(SVAL) + b - numpy.log(pT / (1-pT))).ravel()
            calibrateScores.append(calibrated_SVAL)
            labelsCalibrate.append(LLVAL)

           

        calibrateScores = numpy.hstack(calibrateScores)
        labelsCalibrate = numpy.hstack(labelsCalibrate)

        # bayesErrorPlotCalibration(DVAL,LVAL,SLLR,model,model)
        # bayesErrorPlotCalibration(DVAL,labelsCalibrate,calibrateScores,model,model+' Calibrate')

        return calibrateScores,labelsCalibrate
    
    
    
   
    

    