
import numpy
import matplotlib
import matplotlib.pyplot as plt

from .bayes_decision import *
from .logistic_regression import *
from .gmm import *
from .svm import *


def extract_train_val_folds_from_ary(X, i, KFOLD = 5):
    return numpy.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != i]), X[i::KFOLD]


def fusion_all(DTR,LTR,DVAL,LVAL):    

    minActDCF = 1
    minimumDCF = 1
    pi  =0

    for pT in np.arange(0.1, 1.00, 0.1):

        KFOLD = 5

        
                
        gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], 8, type = 'diagonal', psiEig = 0.01)
        gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], 32, type = 'diagonal', psiEig = 0.01)
                            
        SLLR_gmm = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
            

        gamma = np.exp(-2)
        kernelFunc = rbfKernel(gamma)
        
        fScore = train_dual_SVM_kernel(DTR, LTR, 31.622776601683793, kernelFunc, 1)
        SLLR_svm = fScore(DVAL)
            
            

        DTR_quadratic  = quadratic(DTR)
        DVAL_quadratic = quadratic(DVAL)
                            
        params_withgrad = trainLogReg_sol1_withgrad_weight(DTR_quadratic, LTR,0.03162277660168379, 0.1)  
        w = params_withgrad[0][0:-1]
        b = params_withgrad[0][-1]

        SLLR_logReg =  (w.T @ DVAL_quadratic + b - numpy.log(0.1 / (1-0.1))).ravel()

        
        fusedScores = [] 
        fusedLabels = [] 
        
        # Train KFOLD times the fusion model
        for foldIdx in range(KFOLD):
            # keep 1 fold for validation, use the remaining ones for training       
            
            SCAL1, SVAL1 = extract_train_val_folds_from_ary(SLLR_gmm, foldIdx)
            SCAL2, SVAL2 = extract_train_val_folds_from_ary(SLLR_svm, foldIdx)
            SCAL3, SVAL3 = extract_train_val_folds_from_ary(SLLR_logReg, foldIdx)
            

            LCAL, LLVAL = extract_train_val_folds_from_ary(LVAL, foldIdx)

            
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

        # Evaluate the performance on pooled scores - we need to use the label vector fusedLabels since it's aligned to calScores_sys_2 (plot on same figure as system 1 and system 2)
        
        actDCF = compute_actDCF(DVAL,fusedLabels,fusedScores, 0.1, 1.0, 1.0)
        minDCF = compute_minDCF(DVAL,fusedLabels,fusedScores, 0.1, 1.0, 1.0)
        print("\t pT: ",pT)
        print ('\t\tminDCF         : %.3f' % compute_minDCF(DVAL,fusedLabels,fusedScores, 0.1, 1.0, 1.0)) 
        print ('\t\tactDCF         : %.3f' % compute_actDCF(DVAL,fusedLabels,fusedScores, 0.1, 1.0, 1.0))
        
        
        if actDCF <  minActDCF:
            minActDCF = actDCF
            minimumDCF = minDCF
            pi = pT
  
    print()
    print("\t actDCF", minActDCF)
    print("\t minDCF", minimumDCF)
    print("\t pT", pi)


def fusion_all2(DVAL,LVAL,sllr1,sllr2,sllr3):
    
    KFOLD = 5

    minActDCF = 1
    minimumDCF = 1
    pi  =0

    for pT in np.arange(0.1, 1.00, 0.1):
    

        fusedScores = [] 
        fusedLabels = [] 


        #  Train KFOLD times the fusion model
        for foldIdx in range(KFOLD):
            # keep 1 fold for validation, use the remaining ones for training       
            
            SCAL1, SVAL1 = extract_train_val_folds_from_ary(sllr1, foldIdx)
            SCAL2, SVAL2 = extract_train_val_folds_from_ary(sllr2, foldIdx)
            SCAL3, SVAL3 = extract_train_val_folds_from_ary(sllr3, foldIdx)
            

            LCAL, LLVAL = extract_train_val_folds_from_ary(LVAL, foldIdx)

            
            SCAL = numpy.vstack([SCAL1, SCAL2,SCAL3])
            
            params = trainLogReg_sol1_withgrad_weight((SCAL), LCAL, 0, pT)
            w = params[0][0:-1]
            b = params[0][-1]
            
            
            # Build the validation scores "feature" matrix
            SVAL = numpy.vstack([SVAL1, SVAL2,SVAL3])
            
            # Apply the model to the validation fold
            calibrated_SVAL =  (w.T @ SVAL + b - numpy.log(pT / (1-pT))).ravel()

            # Add the scores of this validation fold to the cores list
            fusedScores.append(calibrated_SVAL)
            
            # Add the corresponding labels to preserve alignment between scores and labels
            fusedLabels.append(LLVAL)

        # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)        
        fusedScores = numpy.hstack(fusedScores)
        fusedLabels = numpy.hstack(fusedLabels)

        # Evaluate the performance on pooled scores - we need to use the label vector fusedLabels since it's aligned to calScores_sys_2 (plot on same figure as system 1 and system 2)

        print("\t pT", pT)
        print ('\t\tminDCF         : %.3f' % compute_minDCF(DVAL,fusedLabels,fusedScores, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
        print ('\t\tactDCF         : %.3f' % compute_actDCF(DVAL,fusedLabels,fusedScores, 0.1, 1.0, 1.0))

        actDCF = compute_actDCF(DVAL,fusedLabels,fusedScores, 0.1, 1.0, 1.0)
        minDCF = compute_minDCF(DVAL,fusedLabels,fusedScores, 0.1, 1.0, 1.0)
        
        if actDCF <  minActDCF:
            minActDCF = actDCF
            minimumDCF = minDCF
            pi = pT
  
    print()
    print("\t actDCF", minActDCF)
    print("\t minDCF", minimumDCF)
    print("\t pT", pi)

  
    

def fusion_2Classify(DTR,LTR,DVAL,LVAL,sllr1,sllr2):    


    KFOLD = 5

    minActDCF = 1
    minimumDCF = 1
    pi  =0

    for pT in np.arange(0.1, 1.00, 0.1):
    

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
            
            
            # Build the validation scores "feature" matrix
            SVAL = numpy.vstack([SVAL1, SVAL2])
            
            # Apply the model to the validation fold
            calibrated_SVAL =  (w.T @ SVAL + b - numpy.log(pT / (1-pT))).ravel()

            # Add the scores of this validation fold to the cores list
            fusedScores.append(calibrated_SVAL)
            
            # Add the corresponding labels to preserve alignment between scores and labels
            fusedLabels.append(LLVAL)

        # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)        
        fusedScores = numpy.hstack(fusedScores)
        fusedLabels = numpy.hstack(fusedLabels)

        # Evaluate the performance on pooled scores - we need to use the label vector fusedLabels since it's aligned to calScores_sys_2 (plot on same figure as system 1 and system 2)

        print("\t pT", pT)
        print ('\t\tminDCF         : %.3f' % compute_minDCF(DVAL,fusedLabels,fusedScores, 0.1, 1.0, 1.0)) # Calibration may change minDCF due to being fold-dependent (thus it's not globally affine anymore)
        print ('\t\tactDCF         : %.3f' % compute_actDCF(DVAL,fusedLabels,fusedScores, 0.1, 1.0, 1.0))

        actDCF = compute_actDCF(DVAL,fusedLabels,fusedScores, 0.1, 1.0, 1.0)
        minDCF = compute_minDCF(DVAL,fusedLabels,fusedScores, 0.1, 1.0, 1.0)
        
        if actDCF <  minActDCF:
            minActDCF = actDCF
            minimumDCF = minDCF
            pi = pT
  
    print()
    print("\t actDCF", minActDCF)
    print("\t minDCF", minimumDCF)
    print("\t pT", pi)

  


def fusion_all_plot(DTR,LTR,DVAL,LVAL):    

    
    minDCF_values=[]
    actDCF_values=[]

    pT  =0.5
    effPriorLogOdds = np.linspace(-3, 3, 21)

    for e in effPriorLogOdds:

        pi = 1 / (1 + np.exp(-e))

        KFOLD = 5

        gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], 8, type = 'diagonal', psiEig = 0.01)
        gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], 32, type = 'diagonal', psiEig = 0.01)
                            
        SLLR_gmm = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
            

        gamma = np.exp(-2)
        kernelFunc = rbfKernel(gamma)
        
        fScore = train_dual_SVM_kernel(DTR, LTR, 31.622776601683793, kernelFunc, 1)
        SLLR_svm = fScore(DVAL)
            
            

        DTR_quadratic  = quadratic(DTR)
        DVAL_quadratic = quadratic(DVAL)
                            
        params_withgrad = trainLogReg_sol1_withgrad_weight(DTR_quadratic, LTR,0.03162277660168379, 0.1)  
        w = params_withgrad[0][0:-1]
        b = params_withgrad[0][-1]

        SLLR_logReg =  (w.T @ DVAL_quadratic + b - numpy.log(0.1 / (1-0.1))).ravel()

        
        fusedScores = [] 
        fusedLabels = [] 
        
        # Train KFOLD times the fusion model
        for foldIdx in range(KFOLD):
            # keep 1 fold for validation, use the remaining ones for training       
            
            SCAL1, SVAL1 = extract_train_val_folds_from_ary(SLLR_gmm, foldIdx)
            SCAL2, SVAL2 = extract_train_val_folds_from_ary(SLLR_svm, foldIdx)
            SCAL3, SVAL3 = extract_train_val_folds_from_ary(SLLR_logReg, foldIdx)
            

            LCAL, LLVAL = extract_train_val_folds_from_ary(LVAL, foldIdx)

            
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

        # Evaluate the performance on pooled scores - we need to use the label vector fusedLabels since it's aligned to calScores_sys_2 (plot on same figure as system 1 and system 2)
        
        actDCF = compute_actDCF(DVAL,fusedLabels,fusedScores, pi, 1.0, 1.0)
        minDCF = compute_minDCF(DVAL,fusedLabels,fusedScores, pi, 1.0, 1.0)

        minDCF_values.append(minDCF)
        actDCF_values.append(actDCF)
    path = 'plot/plot_fusion/fusionAll.pdf'
    bayes_error_plot_2(effPriorLogOdds,actDCF_values,minDCF_values,'Fusion All',path)
    return actDCF_values,minDCF_values
  




def fusion_2Classify_plot(DTR,LTR,DVAL,LVAL,sllr1,sllr2,title=''):    


    KFOLD = 5
    pT =0.5
    minDCF_values=[]
    actDCF_values=[]

    effPriorLogOdds = np.linspace(-3, 3, 21)

    for e in effPriorLogOdds:

        pi = 1 / (1 + np.exp(-e))
    

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
            
            
            # Build the validation scores "feature" matrix
            SVAL = numpy.vstack([SVAL1, SVAL2])
            
            # Apply the model to the validation fold
            calibrated_SVAL =  (w.T @ SVAL + b - numpy.log(pT / (1-pT))).ravel()

            # Add the scores of this validation fold to the cores list
            fusedScores.append(calibrated_SVAL)
            
            # Add the corresponding labels to preserve alignment between scores and labels
            fusedLabels.append(LLVAL)

        # Re-build the score and label arrays (pooling) - these contains an entry for every element in the original dataset (but the order of the samples is different)        
        fusedScores = numpy.hstack(fusedScores)
        fusedLabels = numpy.hstack(fusedLabels)

        # Evaluate the performance on pooled scores - we need to use the label vector fusedLabels since it's aligned to calScores_sys_2 (plot on same figure as system 1 and system 2)


        actDCF = compute_actDCF(DVAL,fusedLabels,fusedScores, pi, 1.0, 1.0)
        minDCF = compute_minDCF(DVAL,fusedLabels,fusedScores, pi, 1.0, 1.0)
        
        minDCF_values.append(minDCF)
        actDCF_values.append(actDCF)

    path = 'plot/plot_fusion/'+title+'.pdf'
    bayes_error_plot_2(effPriorLogOdds,actDCF_values,minDCF_values,title,path)
    return actDCF_values,minDCF_values
        

  