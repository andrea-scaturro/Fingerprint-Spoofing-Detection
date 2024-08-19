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
from src.bayes_decision import *




class Application:

    def __init__(self, pi, Cfn, Cfp):
        self.pi = pi
        self.Cfn = Cfn
        self.Cfp = Cfp


#======================= Main Program ==========================

if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load(data_path)
    m=6

    # DTR and LTR are model training data and labels 
    # DVAL and LVAL are validation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)


    ## LAB 7





    applications_data = [
    (0.5, 1.0, 1.0),
    (0.9, 1.0, 1.0),
    (0.1, 1.0, 1.0),
    (0.5, 1.0, 9.0),
    (0.5, 9.0, 1.0)
]


    

    applications = [Application(pi, Cfn, Cfp) for pi, Cfn, Cfp in applications_data]
    
    
    t = np.zeros((len(applications)), dtype=float)

    for i in range(len(applications)):
        t[i] = compute_bayes_threshold(applications[i].pi, applications[i].Cfn, applications[i].Cfp)
    
    
    threshold_effetive_prior = np.zeros((5), dtype=float)
    effetive_prior = np.zeros((5), dtype=float)

    for i in range(len(applications)):

        effetive_prior[i] = (applications[i].pi* applications[i].Cfn /((((1-applications[i].pi)*applications[i].Cfp) + applications[i].pi*applications[i].Cfn)))
    
    
    for i in range(len(applications)):
        threshold_effetive_prior[i] = compute_bayes_threshold(effetive_prior[i], 1, 1)
    



    print("\nMVG \n")
    
    for i in range(len(applications)):
        res,_ ,_ = mvg(DTR, LTR, DVAL, LVAL,t[i],m)
        res2,_, _ = mvg(DTR, LTR, DVAL, LVAL,threshold_effetive_prior[i],m)
        print("For (pi, Cfn, Cfp) = (%4.2f, %4.2f, %4.2f): " % (applications[i].pi, applications[i].Cfn ,applications[i].Cfp), "Error Rate:  %.3f"  %((res) / int( DVAL.shape[1])*100) ,"%" )    
        print("Effective Prior: (%4.2f): " %(effetive_prior[i]), "Error Rate:  %.3f"  %((res2) / int( DVAL.shape[1])*100) ,"%\n" )    
    

    print("\nNaive Bayes \n")
    
    for i in range(len(applications)):
        res,_, _= mvg_nb(DTR, LTR, DVAL, LVAL,t[i],m)
        res2,_, _= mvg_nb(DTR, LTR, DVAL, LVAL,threshold_effetive_prior[i],m)
        print("For (pi, Cfn, Cfp) = (%4.2f, %4.2f, %4.2f): " % (applications[i].pi, applications[i].Cfn ,applications[i].Cfp), "Error Rate:  %.3f"  %((res) / int( DVAL.shape[1])*100) ,"%" )    
        print("Effective Prior: (%4.2f): " % (effetive_prior[i]), "Error Rate:  %.3f"  %((res2) / int( DVAL.shape[1])*100) ,"%\n" )    

    print("\nTied \n")
    
    for i in range(len(applications)):
        res,_, _= mvg_tied(DTR, LTR, DVAL, LVAL,t[i],m)
        res2,_, _= mvg_tied(DTR, LTR, DVAL, LVAL,threshold_effetive_prior[i],m)
        print("For (pi, Cfn, Cfp) = (%4.2f, %4.2f, %4.2f): " % (applications[i].pi, applications[i].Cfn ,applications[i].Cfp), "Error Rate:  %.3f"  %((res) / int( DVAL.shape[1])*100) ,"%" )
        print("Effective Prior: (%4.2f): " % (effetive_prior[i]), "Error Rate:  %.3f"  %((res2) / int( DVAL.shape[1])*100) ,"%\n" )    


    print("===========================================================================================")
    print("\nMVG \n")
    for i in range(3):
        res, predictions,_= mvg(DTR, LTR, DVAL, LVAL,t[i],m)
        res2, predictions2, _= mvg(DTR, LTR, DVAL, LVAL,threshold_effetive_prior[i],m)
        conf_1 = compute_confusion_matrix(predictions,LVAL,applications[i].pi )
        conf_2 = compute_confusion_matrix(predictions2,LVAL,effetive_prior[i] )
        
        format_confusion_matrices(
            conf_1, conf_2,
            f"Without Effective Prior:  (pi: {applications[i].pi:.3f})",
            f"With Effective Prior:  (pi: {effetive_prior[i]:.3f})"
        )

        print("With PCA:")

        for l in range(m):
            print("m = %d" %l)
            apply_PCA = PCA(DTR, l)
            DTR_pca = numpy.dot(apply_PCA.T, DTR)

            apply_PCA = PCA(DVAL, l)
            DVAL_pca = numpy.dot(apply_PCA.T, DVAL)


            res, predictions,llr_pca = mvg(DTR_pca, LTR, DVAL_pca, LVAL,t[i],l)
            res2, predictions2,_= mvg(DTR_pca, LTR, DVAL_pca, LVAL,threshold_effetive_prior[i],l)


            print("Prior:  (%4.2f): " % (applications[i].pi), "Error Rate:  %.3f"  %((res) / int( DVAL_pca.shape[1])*100) ,"%" )    
            print("Effective Prior: (%4.2f): " %(effetive_prior[i]), "Error Rate:  %.3f"  %((res2) / int( DVAL_pca.shape[1])*100) ,"%\n" )    
    

            conf_1 = compute_confusion_matrix(predictions,LVAL,applications[i].pi )
            conf_2 = compute_confusion_matrix(predictions2,LVAL,effetive_prior[i] )

            format_confusion_matrices(
                conf_1, conf_2,
                f"Without Effective Prior:  (pi: {applications[i].pi:.3f})",
                f"With Effective Prior:  (pi: {effetive_prior[i]:.3f})"
            )
            print("\nDCF: %f" %compute_actDCF(DVAL_pca,LVAL, llr_pca,applications[i].pi, applications[i].Cfn, applications[i].Cfp))
        
            min_dcf = compute_minDCF(DVAL_pca,LVAL, llr_pca,applications[i].pi, applications[i].Cfn, applications[i].Cfp)
            print("DCFu: %f\n" %min_dcf)
        #bayes_error_plot(DVAL, LVAL, llr, 'MVG')
        print("-------------------------------------------------------------------------------------------\n")


    print("===========================================================================================")
    print("\nNaive Bayes \n")
    for i in range(3):
        res, predictions, _= mvg_nb(DTR, LTR, DVAL, LVAL,t[i],m)
        res2, predictions2, _= mvg_nb(DTR, LTR, DVAL, LVAL,threshold_effetive_prior[i],m)
        conf_1 = compute_confusion_matrix(predictions,LVAL,applications[i].pi )
        conf_2 = compute_confusion_matrix(predictions2,LVAL,effetive_prior[i] )
        

        format_confusion_matrices(
            conf_1, conf_2,
            f"Without Effective Prior:  (pi: {applications[i].pi:.3f})",
            f"With Effective Prior:  (pi: {effetive_prior[i]:.3f})"
        )

        print("With PCA:")

        for l in range(m):
           

            apply_PCA = PCA(DTR, l)
            DTR_pca = numpy.dot(apply_PCA.T, DTR)

            apply_PCA = PCA(DVAL, l)
            DVAL_pca = numpy.dot(apply_PCA.T, DVAL)

            res, predictions, llr_pca= mvg_nb(DTR_pca, LTR, DVAL_pca, LVAL,t[i],l)
            res2, predictions2, _= mvg_nb(DTR_pca, LTR, DVAL_pca, LVAL,threshold_effetive_prior[i],l)


            print("m = %d" %l)
            print("Prior:  (%4.2f): " % (applications[i].pi), "Error Rate:  %.3f"  %((res) / int( DVAL_pca.shape[1])*100) ,"%" )    
            print("Effective Prior: (%4.2f): " %(effetive_prior[i]), "Error Rate:  %.3f"  %((res2) / int( DVAL_pca.shape[1])*100) ,"%\n" )    
    
            conf_1 = compute_confusion_matrix(predictions,LVAL,applications[i].pi )
            conf_2 = compute_confusion_matrix(predictions2,LVAL,effetive_prior[i] )

            format_confusion_matrices(
                conf_1, conf_2,
                f"Without Effective Prior:  (pi: {applications[i].pi:.3f})",
                f"With Effective Prior:  (pi: {effetive_prior[i]:.3f})"
            )
        
            print("\nDCF: %f" %compute_actDCF(DVAL_pca,LVAL, llr_pca,applications[i].pi, applications[i].Cfn, applications[i].Cfp))
            min_dcf = compute_minDCF(DVAL_pca,LVAL, llr_pca,applications[i].pi, applications[i].Cfn, applications[i].Cfp)
            print("DCFu: %f\n" %min_dcf)

        #bayes_error_plot(DVAL, LVAL, llr, 'Naive Bayes')
        print("-------------------------------------------------------------------------------------------\n")


    print("===========================================================================================")
    print("\nTied \n")
    for i in range(3):
        res, predictions,_= mvg_tied(DTR, LTR, DVAL, LVAL,t[i],m)
        res2, predictions2, _= mvg_tied(DTR, LTR, DVAL, LVAL,threshold_effetive_prior[i],m)
        
        conf_1 = compute_confusion_matrix(predictions,LVAL,applications[i].pi )
        conf_2 = compute_confusion_matrix(predictions2,LVAL,effetive_prior[i] )
        

        format_confusion_matrices(
            conf_1, conf_2,
            f"Without Effective Prior:  (pi: {applications[i].pi:.3f})",
            f"With Effective Prior:  (pi: {effetive_prior[i]:.3f})"
        )
        print("With PCA:")
        for l in range(m):
            print("m = %d" %l)
            apply_PCA = PCA(DTR, l)
            DTR_pca = numpy.dot(apply_PCA.T, DTR)

            apply_PCA = PCA(DVAL, l)
            DVAL_pca = numpy.dot(apply_PCA.T, DVAL)



            res, predictions, llr_pca= mvg_tied(DTR_pca, LTR, DVAL_pca, LVAL,t[i],l)
            res2, predictions2, _= mvg_tied(DTR_pca, LTR, DVAL_pca, LVAL,threshold_effetive_prior[i],l)


            print("Prior:  (%4.2f): " % (applications[i].pi), "Error Rate:  %.3f"  %((res) / int( DVAL_pca.shape[1])*100) ,"%" )    
            print("Effective Prior: (%4.2f): " %(effetive_prior[i]), "Error Rate:  %.3f"  %((res2) / int( DVAL_pca.shape[1])*100) ,"%\n" )    
    
            conf_1 = compute_confusion_matrix(predictions,LVAL,applications[i].pi )
            conf_2 = compute_confusion_matrix(predictions2,LVAL,effetive_prior[i] )

            format_confusion_matrices(
                conf_1, conf_2,
                f"Without Effective Prior:  (pi: {applications[i].pi:.3f})",
                f"With Effective Prior:  (pi: {effetive_prior[i]:.3f})"
            )
            print("\nDCF: %f" %compute_actDCF(DVAL_pca,LVAL, llr_pca,applications[i].pi, applications[i].Cfn, applications[i].Cfp))
            min_dcf = compute_minDCF(DVAL_pca,LVAL, llr_pca,applications[i].pi, applications[i].Cfn, applications[i].Cfp)
            print("DCFu: %f\n" %min_dcf)
        #bayes_error_plot(DVAL, LVAL, llr, 'Tied')
        print("-------------------------------------------------------------------------------------------\n")


        
        
_,_,llr_MVG = mvg(DTR,LTR,DVAL,LVAL,compute_bayes_threshold(0.1,1,1),6)
_,_,llr_NB = mvg_nb(DTR,LTR,DVAL,LVAL,compute_bayes_threshold(0.1,1,1),6)
_,_,llr_T = mvg_tied(DTR,LTR,DVAL,LVAL,compute_bayes_threshold(0.1,1,1),6)

comparing_recognizer_plot(DVAL,LVAL,llr_MVG,llr_NB,llr_T)


    