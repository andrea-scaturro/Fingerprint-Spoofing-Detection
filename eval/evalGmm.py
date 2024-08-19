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



def eval_NumGaussianComponent():

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

                


#compare best models
def eval_models():


    minDCF_gmm    =[]
    minDCF_svm    =[]
    minDCF_logReg =[]
    minDCF_mvg    =[]

    actDCF_gmm    =[]
    actDCF_svm    =[]
    actDCF_logReg =[]
    actDCF_mvg    =[]

    colors =    ['#8B0000', '#006400', '#00008B', '#FFD700', '#8B008B'] 

    effPriorLogOdds = np.linspace(-4, 4, 21)
    for model in ['gmm','svm', 'logReg']:

        for e in tqdm(effPriorLogOdds, desc=f'model: {model}'):    

            if model == 'gmm':

                pi = 1 / (1 + np.exp(-e))
                gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], 8, type = 'diagonal', psiEig = 0.01)
                gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], 32, type = 'diagonal', psiEig = 0.01)
                    
                SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)

                minDCF=compute_minDCF(DVAL,LVAL,SLLR,pi)
                dcf = compute_actDCF(DVAL,LVAL,SLLR,pi)
                
                minDCF_gmm.append(minDCF)
                actDCF_gmm.append(dcf)
                
            elif model == 'svm':
                        
                pi = 1 / (1 + np.exp(-e))
                gamma = np.exp(-2)
                kernelFunc = rbfKernel(gamma)
                eps = 1
                fScore = train_dual_SVM_kernel(DTR, LTR, 31.622776601683793, kernelFunc, eps)
                SLLR = fScore(DVAL)

                minDCF=compute_minDCF(DVAL,LVAL,SLLR,pi)
                dcf = compute_actDCF(DVAL,LVAL,SLLR,pi)
                
                minDCF_svm.append(minDCF)
                actDCF_svm.append(dcf)


            elif model == 'logReg':

                pi = 1 / (1 + np.exp(-e))
                epsilon = 1e-10
                pi = max(min(pi, 1 - epsilon), epsilon)

                DTR_quadratic = quadratic(DTR)
                DVAL_quadratic = quadratic(DVAL)
                        
                params_withgrad = trainLogReg_sol1_withgrad_weight(DTR_quadratic, LTR,0.03162277660168379, pi)  
                w = params_withgrad[0][0:-1]
                b = params_withgrad[0][-1]

                SLLR =  (w.T @ DVAL_quadratic + b - numpy.log(pi / (1-pi))).ravel()

                minDCF = compute_minDCF(DVAL_quadratic, LVAL, SLLR,pi)
                dcf = compute_actDCF(DVAL_quadratic, LVAL, SLLR, pi)

                minDCF_logReg.append(minDCF)
                actDCF_logReg.append(dcf)
                    

            # elif model == 'mvg':

            #     _,_,SLLR = mvg(DTR, LTR, DVAL, LVAL)

            #     pi = 1 / (1 + np.exp(-e))

            #     minDCF=compute_minDCF(DVAL,LVAL,SLLR,pi)
            #     dcf = compute_actDCF(DVAL,LVAL,SLLR,pi)

            #     minDCF_mvg.append(minDCF)
            #     actDCF_mvg.append(dcf)


    #plt.xscale("log", base=10)
    plt.plot(effPriorLogOdds, minDCF_gmm, label=f'min DCF : gmm', linestyle='--',color=colors[0])
    plt.plot(effPriorLogOdds, actDCF_gmm,    label=f'       DCF : gmm',linestyle='-', color=colors[0])

    plt.plot(effPriorLogOdds, minDCF_svm, label=f'min DCF : svm', linestyle='--',color=colors[1])
    plt.plot(effPriorLogOdds, actDCF_svm,    label=f'       DCF : svm',linestyle='-', color=colors[1])

    plt.plot(effPriorLogOdds, minDCF_logReg, label=f'min DCF : logReg', linestyle='--', color= colors[2])
    plt.plot(effPriorLogOdds, actDCF_logReg,    label=f'       DCF : logReg',linestyle='-', color=colors[2])

    # plt.plot(effPriorLogOdds, minDCF_mvg, label=f'min DCF : mvg', linestyle='--', color= colors[3])
    # plt.plot(effPriorLogOdds, actDCF_mvg,    label=f'       DCF : lmvg',linestyle='-', color=colors[3])

    
            
    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])

    plt.ylabel('minDCF/DCF Value')
    plt.xlabel('logOdds')
    plt.title('Models eval')
        
    plt.legend(fontsize=10)
    # plt.grid(True)
    plt.savefig('plot/error_plot/BayesErrorPlot_BestModel.pdf')
    plt.show()

        
                

    

def eval_application(DTR,LTR):

    type = 'diagonal'
    effPriorLogOdds = np.linspace(-4, 4, 21)
    
    colors =    ['#8B0000', '#006400', '#00008B', '#FFD700', '#8B008B']  # Colori scuri

    models = ['gmm','mvg','logReg','svm']

    for model in models:

        
        # cancello il grafico precedente
        plt.clf()  

        for i in range(len(applications)):

            dcf_values =[]
            minDcf_values =[]

            print()
            for e in tqdm(effPriorLogOdds, desc=f'Processing application {i+1}/{len(applications)}'):    

                
                if model == 'gmm':

                    pi = 1 / (1 + np.exp(-e))
                    gmm0 = train_GMM_LBG_EM(DTR[:, LTR==0], 8, type = type, psiEig = 0.01)
                    gmm1 = train_GMM_LBG_EM(DTR[:, LTR==1], 32, type = type, psiEig = 0.01)
                    
                    SLLR = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)

                    minDCF=compute_minDCF(DVAL,LVAL,SLLR,pi,applications[i].Cfp, applications[i].Cfn)
                    dcf = compute_actDCF(DVAL,LVAL,SLLR,pi, applications[i].Cfp, applications[i].Cfn)
                
                
                if model == 'svm':
                    
                    pi = 1 / (1 + np.exp(-e))
                    gamma = np.exp(-2)
                    kernelFunc = rbfKernel(gamma)
                    
                    fScore = train_dual_SVM_kernel(DTR, LTR, 31.622776601683793, kernelFunc)
                    SLLR = fScore(DVAL)

                    minDCF=compute_minDCF(DVAL,LVAL,SLLR,pi,applications[i].Cfp, applications[i].Cfn)
                    dcf = compute_actDCF(DVAL,LVAL,SLLR,pi, applications[i].Cfp, applications[i].Cfn)


                if model == 'logReg':

                    pi = 1 / (1 + np.exp(-e))

                    DTR_quadratic = quadratic(DTR)
                    DVAL_quadratic = quadratic(DVAL)
                     
                    params_withgrad = trainLogReg_sol1_withgrad_weight(DTR_quadratic, LTR, 0.03162277660168379, pi)  
                    w = params_withgrad[0][0:-1]
                    b = params_withgrad[0][-1]

                    SLLR =  (w.T @ DVAL_quadratic + b - numpy.log(pi / (1-pi))).ravel()

                    minDCF = compute_minDCF(DVAL_quadratic, LVAL, SLLR,pi,applications[i].Cfp, applications[i].Cfn)
                    dcf = compute_actDCF(DVAL_quadratic, LVAL, SLLR, pi, applications[i].Cfp, applications[i].Cfn)
                

                if model == 'mvg':

                    _,_,SLLR = mvg(DTR, LTR, DVAL, LVAL)

                    pi = 1 / (1 + np.exp(-e))

                    minDCF=compute_minDCF(DVAL,LVAL,SLLR,pi,applications[i].Cfp, applications[i].Cfn)
                    dcf = compute_actDCF(DVAL,LVAL,SLLR,pi, applications[i].Cfp, applications[i].Cfn)
            
                
                
                minDcf_values.append(minDCF)
                dcf_values.append(dcf)

            #plt.xscale("log", base=10)
            plt.plot(effPriorLogOdds, minDcf_values, label=f'min DCF : {i+1}', linestyle='--',color=colors[i] )
            plt.plot(effPriorLogOdds, dcf_values,    label=f'       DCF : {i+1}',linestyle='-', color=colors[i])
            
        plt.ylim([0, 1.1])
        plt.xlim([-4, 4])

        plt.ylabel('minDCF/DCF Value')
        plt.xlabel('logOdds')
        plt.title(model)
        
        plt.legend(fontsize=10)
        # plt.grid(True)
        plt.savefig('plot/plot_gmm/error_plot/BayesErrorPlot_'+model+'.pdf')
        #plt.show()
        print()


# ======================= Main Program ==========================

if __name__ == '__main__':

    D, L = load(data_path)
    m = 6

    # DTR and LTR are model training data and labels
    # DVAL and LVAL are validation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)


    #eval_NumGaussianComponent()
    print()
    eval_models()
    print()
    eval_application(DTR,LTR)







