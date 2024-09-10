import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy

from .mvg  import *

def vrow(v):
    return v.reshape((1, v.size))


def vcol(v):
    return v.reshape((v.size, 1))  



def compute_DCF(DVAL, LVAL,llr,pi1, Cfn=1, Cfp=1):
    
    prior_HT = pi1
    prior_HF = 1 - pi1
    
    
    posterior_HT = prior_HT * np.exp(llr) / (prior_HT * np.exp(llr) + prior_HF)
    posterior_HF = 1 - posterior_HT
    
    
    cost_HT = Cfp * posterior_HF
    cost_HF = Cfn * posterior_HT
    
    
    decisions = np.where(cost_HT < cost_HF, 1, 0)

    
    conf_matrix = np.zeros((2, 2), dtype=int)

    
    for true_label, decision in zip(LVAL, decisions):
        if  decision == 0 and true_label == 0:
            conf_matrix[0, 0] += 1
        elif decision == 1 and true_label == 0 :
            conf_matrix[1, 0] += 1
        elif decision == 0 and true_label == 1:
            conf_matrix[0, 1] += 1
        elif decision == 1 and true_label == 1:
            conf_matrix[1, 1] += 1


    Pfn = conf_matrix[0,1] / (conf_matrix[0,1] + conf_matrix[1,1])
    Pfp = conf_matrix[1,0] / (conf_matrix[1,0] + conf_matrix[0,0] )

    DCF = prior_HT*Cfn*Pfn + prior_HF*Cfp*Pfp


    return DCF





def compute_actDCF(DVAL, LVAL,llr,pi1, Cfn=1, Cfp=1):
    
    
    labels = LVAL
   
    
    prior_HT = pi1
    prior_HF = 1 - pi1
    
    
    posterior_HT = prior_HT * np.exp(llr) / (prior_HT * np.exp(llr) + prior_HF)
    posterior_HF = 1 - posterior_HT
    
    
    cost_HT = Cfp * posterior_HF
    cost_HF = Cfn * posterior_HT
    
    
    decisions = np.where(cost_HT < cost_HF, 1, 0)

    
    conf_matrix = np.zeros((2, 2), dtype=int)

    
    for true_label, decision in zip(labels, decisions):
        if  decision == 0 and true_label == 0:
            conf_matrix[0, 0] += 1
        elif decision == 1 and true_label == 0 :
            conf_matrix[1, 0] += 1
        elif decision == 0 and true_label == 1:
            conf_matrix[0, 1] += 1
        elif decision == 1 and true_label == 1:
            conf_matrix[1, 1] += 1

    Pfn = conf_matrix[0,1] / (conf_matrix[0,1] + conf_matrix[1,1])
    Pfp = conf_matrix[1,0] / (conf_matrix[1,0] + conf_matrix[0,0] )

    DCF = prior_HT*Cfn*Pfn + prior_HF*Cfp*Pfp

    Bdummy = min(pi1*Cfn,(1-pi1)*Cfp )
    if Bdummy ==0 :
        Bdummy = 1 
    return DCF/Bdummy







def compute_minDCF(DVAL,LVAL,llr,prior1, Cfp=1, Cfn=1):  

    
    labels = LVAL
    prior0 = 1 - prior1

   
    llr_sorted = np.sort(llr)

   
    min_cost = float('inf')
    min_threshold = None
    for threshold in llr_sorted:
        predictions = np.zeros_like(llr, dtype=int) 
        predictions[llr > threshold] = 1
        predictions[llr <= threshold] = 0

        
        confusion_matrix = np.zeros((2, 2), dtype=int)
        for true_class, predicted_class in zip(labels, predictions):
            confusion_matrix[predicted_class, true_class] += 1  

        
        Pfn = confusion_matrix[0, 1] / (confusion_matrix[0, 1] + confusion_matrix[1, 1])  # false negative rate
        Pfp = confusion_matrix[1, 0] / (confusion_matrix[1, 0] + confusion_matrix[0, 0])  # false positive rate

        # Calcolo del DCF
        DCF = prior1 * Cfn * Pfn + prior0 * Cfp * Pfp
        
        
        if DCF < min_cost:
            min_cost = DCF
            min_threshold = threshold
    
    # Calcolo del DCF normalizzato
    Bdummy = min(prior1 * Cfn, prior0 * Cfp)
    
    if Bdummy ==0 :
        Bdummy = 1 
    DCF_normalized = min_cost / Bdummy
    return DCF_normalized








def bayes_error_plot(DVAL,LVAL,llr, title='',path=''):

    # Definire i log-odds efficaci
    effPriorLogOdds = np.linspace(-3, 3, 21)
    
    # Array per memorizzare i valori DCF e min DCF
    dcf_values = []
    min_dcf_values = []

    # Calcolo dei valori DCF e min DCF per ciascun prior efficace
    for p in effPriorLogOdds:
        pi = 1 / (1 + np.exp(-p))
        dcf = compute_actDCF(DVAL,LVAL,llr,pi, 1, 1)
        min_dcf = compute_minDCF(DVAL,LVAL,llr,pi,1,1)
        dcf_values.append(dcf)
        min_dcf_values.append(min_dcf)

    # Tracciare il grafico
    plt.plot(effPriorLogOdds, dcf_values, label='DCF', color='r')
    plt.plot(effPriorLogOdds, min_dcf_values, label='min DCF', color='b')
    plt.ylim([0, 0.8])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF Value')
    plt.title('Bayes Error Plot: '+title)
    plt.legend()
    #plt.grid(True)

    if path != '':

        plt.savefig(path)

    plt.show()





def bayes_error_plotDCF(DVAL,LVAL,llr1,llr2,llr3 ,llr4,title='',path=''):

    
    effPriorLogOdds = np.linspace(-4, 4, 21)
    
    
    dcf_values1 = []
    dcf_values3 = []
    dcf_values2 = []
    dcf_values4 = []

    # Calcolo dei valori DCF e min DCF per ciascun prior efficace
    for p in effPriorLogOdds:
        pi = 1 / (1 + np.exp(-p))
        dcf1 = compute_actDCF(DVAL,LVAL,llr1,pi, 1, 1)
        dcf2 = compute_actDCF(DVAL,LVAL,llr2,pi, 1, 1)
        dcf3 = compute_actDCF(DVAL,LVAL,llr3,pi, 1, 1)
        dcf4 = compute_actDCF(DVAL,LVAL,llr4,pi, 1, 1)

        dcf_values1.append(dcf1)
        dcf_values2.append(dcf2)
        dcf_values3.append(dcf3)
        dcf_values4.append(dcf4)


    # Tracciare il grafico
    plt.plot(effPriorLogOdds, dcf_values4, label='DCF All'    , color='gold')
    plt.plot(effPriorLogOdds, dcf_values1, label='DCF_GMM-SVM ', color='r')
    plt.plot(effPriorLogOdds, dcf_values2, label='DCF_GMM-Log', color='b')
    plt.plot(effPriorLogOdds, dcf_values3, label='DCF_SVM-Log', color='g')
    plt.ylim([0, 0.8])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF Value')
    plt.title('Bayes Error Plot: '+title)
    plt.legend()
    #plt.grid(True)

    if path != '':

        plt.savefig(path)

    plt.show()




def bayes_error_plot_2(effPriorLogOdds,dcf_values,min_dcf_values, title='',path=''):

    # cancello il grafico precedente
    plt.clf()

    #plt.xscale("log", base=10)
    
    plt.plot(effPriorLogOdds, dcf_values, label='DCF',linestyle='-', color='b')
    plt.plot(effPriorLogOdds, min_dcf_values, label='min DCF', linestyle='--',color='b')
   
    #plt.ylim([0, 1.1])
    plt.ylim([0, 0.8])
    plt.xlim([-3, 3])

    plt.xlabel('C')
    plt.ylabel('DCF Value')
    plt.title('Bayes Error Plot: '+title)
    plt.legend()
    #plt.grid(True)
    
    if path != '':

        plt.savefig(path)

    #plt.show()



def plot_dcf_and_min_dcf():  #funzione per il lab

    # Definire i log-odds efficaci
    effPriorLogOdds = np.linspace(-3, 3, 21)
    
    # Array per memorizzare i valori DCF e min DCF
    dcf_values = []
    min_dcf_values = []

    path= 'data_lab7/commedia_llr_infpar.npy'

    # Calcolo dei valori DCF e min DCF per ciascun prior efficace
    for p in effPriorLogOdds:
        pi = 1 / (1 + np.exp(-p))
        dcf = compute_actDCF(path,pi, 1, 1)
        min_dcf = compute_minDCF(path,pi,1,1)
        dcf_values.append(dcf)
        min_dcf_values.append(min_dcf)

    # Tracciare il grafico
    plt.plot(effPriorLogOdds, dcf_values, label='DCF', color='r')
    plt.plot(effPriorLogOdds, min_dcf_values, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF Value')
    plt.title('Grafico dell\'Errore di Bayes Normalizzato')
    plt.legend()
    #plt.grid(True)
    plt.show()



def comparing_recognizer():

    path= 'data_lab7/commedia_llr_infpar.npy'
    path_1= 'data_lab7/commedia_llr_infpar_eps1.npy'


    val1 = compute_actDCF(path, 0.5, 1, 1)
    val2 = compute_minDCF(path, 0.5, 1, 1)
    val3 = compute_actDCF(path_1, 0.5, 1, 1)
    val4 = compute_minDCF(path_1, 0.5, 1, 1)

    val5 = compute_actDCF(path, 0.8, 1, 1)
    val6 = compute_minDCF(path, 0.8, 1, 1)
    val7 = compute_actDCF(path_1, 0.8, 1, 1)
    val8 = compute_minDCF(path_1, 0.8, 1, 1)

    val9 = compute_actDCF(path, 0.5, 10, 1)
    val10 = compute_minDCF(path, 0.5, 10, 1)
    val11 = compute_actDCF(path_1, 0.5, 10, 1)
    val12 = compute_minDCF(path_1, 0.5, 10, 1)

    val13 = compute_actDCF(path, 0.8, 1, 10)
    val14 = compute_minDCF(path, 0.8, 1, 10)
    val15 = compute_actDCF(path_1, 0.8, 1, 10)
    val16 = compute_minDCF(path_1, 0.8, 1, 10)


    print("\n-------------------------------------------------------")

    print("  \u03C0,  Cfn,  Cfp  |  DCF    min DCF |  DCF    min DCF")

    print("-------------------------------------------------------")
    print(" (0.5)  1    1   | %.3f   %.3f   | %.3f    %.3f  " % (val1, val2, val3, val4))
    print("-------------------------------------------------------")
    print(" (0.8)  1    1   | %.3f   %.3f   | %.3f    %.3f  " % (val5, val6, val7, val8))
    print("-------------------------------------------------------")
    print(" (0.5)  10   1   | %.3f   %.3f   | %.3f    %.3f  " % (val9, val10, val11, val12))
    print("-------------------------------------------------------")
    print(" (0.8)  1    10  | %.3f   %.3f   | %.3f    %.3f  \n" % (val13, val14, val15, val16))
  



def comparing_recognizer_plot(DVAL,labels,llr,llr_1,llr_2):

    # Definire i log-odds efficaci
    effPriorLogOdds = np.linspace(-4, 4, 21)
    
    # Array per memorizzare i valori DCF e min DCF
    dcf_values = []
    min_dcf_values = []

    dcf_values_1 = []
    min_dcf_values_1 = []

    dcf_values_2 = []
    min_dcf_values_2 = []

  

    # Calcolo dei valori DCF e min DCF per ciascun prior efficace
    for p in effPriorLogOdds:
        pi = 1 / (1 + np.exp(-p))

        dcf = compute_actDCF(DVAL,labels,llr,pi, 1, 1)
        min_dcf = compute_minDCF(DVAL,labels,llr,pi, 1, 1)
        dcf_values.append(dcf)
        min_dcf_values.append(min_dcf)


        dcf_1 = compute_actDCF(DVAL,labels,llr_1,pi, 1, 1)
        min_dcf_1 = compute_minDCF(DVAL,labels,llr_1,pi, 1, 1)
        dcf_values_1.append(dcf_1)
        min_dcf_values_1.append(min_dcf_1)

        dcf_2 = compute_actDCF(DVAL,labels,llr_2,pi, 1, 1)
        min_dcf_2= compute_minDCF(DVAL,labels,llr_2,pi, 1, 1)
        dcf_values_2.append(dcf_2)
        min_dcf_values_2.append(min_dcf_2)

    # Tracciare il grafico
    plt.plot(effPriorLogOdds, dcf_values, label='DCF MVG',  linestyle='-',color='r',linewidth=2.0)
    plt.plot(effPriorLogOdds, min_dcf_values, label='min DCF MVG',linestyle='--', color='r',linewidth=2.0)

    plt.plot(effPriorLogOdds, dcf_values_1, label='DCF NB', linestyle='-',color='g',linewidth=2.0)
    plt.plot(effPriorLogOdds, min_dcf_values_1, label='min DCF NB', linestyle='--',color='g',linewidth=2.0)

    plt.plot(effPriorLogOdds, dcf_values_2, label='DCF T', linestyle='-',color='c',linewidth=2.0)
    plt.plot(effPriorLogOdds, min_dcf_values_2, label='min DCF T', linestyle='--',color='c',linewidth=2.0)


    plt.ylim([0, 1.1])
    plt.xlim([-4, 4])
    plt.xlabel('prior log-odds')
    plt.ylabel('DCF Value')
    plt.title('')
    plt.legend()
    #plt.grid(True)
    plt.show()






def compute_logPosterior(S_logLikelihood, v_prior):
    SJoint = S_logLikelihood + vcol(np.log(v_prior))
    SMarginal = vrow(scipy.special.logsumexp(SJoint, axis=0))
    SPost = SJoint - SMarginal
    return SPost


    
def multiclass_eval(ll, ll_labels, C, vPrior):
    
    log_posterior = compute_logPosterior(ll, vPrior)
    posterior = np.exp(log_posterior)
    C_final = np.dot(C, posterior) 
    predictions =np.argmin(C_final, axis=0)
    
    confusion_matrix = np.zeros((3, 3), dtype=int)

    for true_class, predicted_class in zip(ll_labels, predictions):
        confusion_matrix[predicted_class, true_class] += 1
        
    print("\nMatrice di Confusione:")
    print(confusion_matrix,"\n")
    
    DCFnorm=np.min(np.dot(C, vPrior)) 
    
    misclassification_ratios = confusion_matrix / np.sum(confusion_matrix, axis=0, keepdims=True)
    DCFu = np.sum(vPrior[:, np.newaxis] * misclassification_ratios.T * C)
    
    print("DCFu: %.3f" %DCFu)
    DCF = DCFu / DCFnorm
    print("DCF: %.3f" %DCF)



def compute_bayes_threshold(pi1,Cfn,Cfp):
    
    ratio = (pi1 * Cfn) / ((1 - pi1) * Cfp)
    return -np.log(ratio)


def compute_confusion_matrix(predictions,labels, prior1):
      
    confusion_matrix = numpy.zeros((2, 2), dtype=int)
    for true_class, predicted_class in zip(labels, predictions):
        confusion_matrix[int(predicted_class), int(true_class)] += 1  

    return confusion_matrix


def format_confusion_matrices(conf1, conf2, label1, label2):
    # Convert matrices to strings
    conf1_str = np.array2string(conf1, formatter={'int': lambda x: f"{x:4d}"})
    conf2_str = np.array2string(conf2, formatter={'int': lambda x: f"{x:4d}"})

    # Split the string representation into lines
    conf1_lines = conf1_str.split('\n')
    conf2_lines = conf2_str.split('\n')

    # Print headers
    print(f"{label1:<45}{label2}")
    print("")

    # Print each line side by side
    for line1, line2 in zip(conf1_lines, conf2_lines):
        print(f"{line1:<45}{line2}")
    print("")

