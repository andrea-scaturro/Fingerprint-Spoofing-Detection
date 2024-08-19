import numpy as np
import scipy.optimize

from dataset.dataset import *
from .dimreduction import *
from .classifier_pca_lda import *
from .plotter import *
from .gaussian_analysis import *
from .mvg import *


##sol1
def trainLogReg_sol1(DTR, LTR, l):
   
    def logreg_obj(v): # ...
        
        w, b = v[:-1], v[-1]

        S = (np.dot(vcol(w).T, DTR) + b).ravel()
        

        ZTR = 2 * LTR - 1
        logistic_loss = np.logaddexp(0, -ZTR * S).mean() + 0.5*l*np.linalg.norm(w)**2
        
        # Obiettivo totale
        J = logistic_loss
    
        return J
    res = scipy.optimize.fmin_l_bfgs_b(func = logreg_obj, x0 = numpy.zeros(DTR.shape[0]+1), approx_grad=True)
    xf = res
    return xf



## sol2
def logreg_obj_sol2(v, DTR, LTR, l):
    
    w, b = v[:-1], v[-1]

    
    S = (np.dot(vcol(w).T, DTR) + b).ravel()
    ZTR = 2 * LTR - 1
    
    
    logistic_loss = np.logaddexp(0, -ZTR * S)
    J = logistic_loss.mean()+ 0.5*l*np.linalg.norm(w)**2
    
    return J


def trainLogReg_sol2(DTR, LTR, l):
    # Inizializzare i parametri a zero (D+1 elementi)
    x0 = np.zeros(DTR.shape[0] + 1)
    
    
    result = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj_sol2, x0=x0, args=(DTR, LTR, l), approx_grad=True)
    
    xf = result
    return xf






## With gradient



##sol1
def trainLogReg_sol1_withgrad(DTR, LTR, l):
   
    def logreg_obj(v): 
        
        w, b = v[:-1], v[-1]
    
        
        S = (np.dot(vcol(w).T, DTR) + b).ravel()
    
        ZTR = 2 * LTR - 1
    

    
        logistic_loss = np.logaddexp(0, -ZTR * S).mean() + 0.5*l*np.linalg.norm(w)**2
        
    
        J = logistic_loss

        G = -ZTR / (1.0 + np.exp(ZTR * S))
        grad_w = np.dot(vrow(G), DTR.T).ravel() / DTR.shape[1] + l * w
        
        grad_b = G.mean()
        
        grad = np.append(grad_w, grad_b)
        return J, grad
    
    res = scipy.optimize.fmin_l_bfgs_b(func = logreg_obj, x0 = numpy.zeros(DTR.shape[0]+1), approx_grad=False)
    xf = res
    return xf




## sol2
def logreg_obj_sol2_withgrad(v, DTR, LTR, l):
   
    w, b = v[:-1], v[-1]
    
    
    S = (np.dot(vcol(w).T, DTR) + b).ravel()
    
    
    ZTR = 2 * LTR - 1
    
    
    logistic_loss = np.logaddexp(0, -ZTR * S)
    
    
    J = logistic_loss.mean() + 0.5 * l * np.linalg.norm(w)**2

    
    G = -ZTR / (1.0 + np.exp(ZTR * S))
    grad_w = ((np.dot(vrow(G), DTR.T).ravel()) / DTR.shape[1]) + (l * w)
    
    grad_b = G.mean()
    
    grad = np.append(grad_w, grad_b)
    return J, grad


def trainLogReg_sol2_withgrad(DTR, LTR, l):
    
    x0 = np.zeros(DTR.shape[0] + 1)
    
    
    result = scipy.optimize.fmin_l_bfgs_b(func=logreg_obj_sol2_withgrad, x0=x0, args=(DTR, LTR, l),approx_grad=False)
    xf = result
    return xf





def error_rate(DVAL,LVAL, w,b):

    S = vcol(w).T @ DVAL + b
    LP = (S > 0).astype(int)
    num_errors = numpy.sum(LVAL.astype(numpy.int32) != LP)
    return num_errors / float(len(LVAL))*100



def sllr_actual(DTR, LTR, DVAL,LVAL, w,b):

     S = vcol(w).T @ DVAL + b
     
     pi_emp = np.sum(LTR)/len(LTR)
     p = pi_emp/(1-pi_emp)

     sllr = S - np.log(p)
     return sllr


def sllr_min(DVAL,LVAL, w,b, pi):

     S = vcol(w).T @ DVAL + b
     p = pi/(1-pi)

     sllr = S - np.log(p)
     return sllr





##sol1 con pesi 
def trainLogReg_sol1_withgrad_weight(DTR, LTR, l, pi):
   
    def logreg_obj(v): 
        
        w, b = v[:-1], v[-1]
        S = (np.dot(vcol(w).T, DTR) + b).ravel()
    
        ZTR = 2 * LTR - 1
        eps = np.where(ZTR == 1, pi/len(LTR[LTR == 1]), (1 -  pi )/ len(LTR[LTR == 0]))
       
        
        logistic_loss = np.logaddexp(0, -ZTR * S)
        J = (eps * logistic_loss ).sum() + 0.5 * l * np.linalg.norm(w)**2

        G = -ZTR / (1.0 + np.exp(ZTR * S))
       
       
        grad_w = ((np.dot( vrow(eps*G) , DTR.T)))+ (l * w)
        
        # Gradiente rispetto a b
        grad_b = (eps* G).sum()
        grad = np.append(grad_w, grad_b)
        return J, grad
    
    res = scipy.optimize.fmin_l_bfgs_b(func = logreg_obj, x0 = numpy.zeros(DTR.shape[0]+1), approx_grad=False)
    
    xf = res
    return xf


def sllr_weight(DVAL,LVAL, w,b,pi):

     S =( vcol(w).T @ DVAL) + b
     
     p = pi/(1-pi)

     sllr = S - np.log(p)
     return sllr






## sol2 quadtratic


def quadratic(DTR):


    fi = np.zeros((DTR.shape[0]**2 + DTR.shape[0], DTR.shape[1]))
    
    for i in range(DTR.shape[1]):
        col = DTR[:,i]
        x_t = (vcol(col) @ vrow(col)).flatten()
        elem = np.concatenate((x_t, col))
        fi[:,i]= elem
    
    return fi


def computeA(DTR,LTR):
    for cls in [0,1]:        
            DCls = DTR[:, LTR==cls]
            mu = DCls.mean(1).reshape(DCls.shape[0], 1)
            
            C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
    return C