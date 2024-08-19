import numpy
import scipy
import scipy.special
import matplotlib.pyplot as plt

from dataset.dataset import *
from .gaussian_analysis import *
from .stats import * 


def logpdf_GMM(X, gmm):

    S = []
    for w, mu, C in gmm:
        logpdf_conditional = compute_ll(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
        
    S = numpy.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    return logdens

def smooth_covariance_matrix(C, psi): 

    U, s, Vh = numpy.linalg.svd(C)
    s[s<psi]=psi
    CUpd = U @ (vcol(s) * U.T)
    return CUpd




def train_GMM_EM_Iteration(X, gmm, type = 'full', psiEig = None): 

    
    S = []
    
    for w, mu, C in gmm:
        logpdf_conditional = compute_ll(X, mu, C)
        logpdf_joint = logpdf_conditional + numpy.log(w)
        S.append(logpdf_joint)
        
    S = numpy.vstack(S) # compute joint densities 
    logdens = scipy.special.logsumexp(S, axis=0) # compute marginal for samples f(x_i)

    gammaAllComponents = numpy.exp(S - logdens)

    
    gmmUpd = []
    for gIdx in range(len(gmm)): 

    
        gamma = gammaAllComponents[gIdx] 
        Z = gamma.sum()
        F = vcol((vrow(gamma) * X).sum(1)) 
        S = (vrow(gamma) * X) @ X.T
        muUpd = F/Z
        CUpd = S/Z - muUpd @ muUpd.T
        wUpd = Z / X.shape[1]
        if type == 'diagonal':
            CUpd  = CUpd * numpy.eye(X.shape[0]) 
        gmmUpd.append((wUpd, muUpd, CUpd))

    if type == 'tied':
        CTied = 0
        for w, mu, C in gmmUpd:
            CTied += w * C
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]

    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]
        
    return gmmUpd


def train_GMM_EM(X, gmm, type = 'full', psiEig = None, epsLLAverage = 1e-6):

    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    
 
    while (llDelta is None or llDelta > epsLLAverage):

        gmm = train_GMM_EM_Iteration(X, gmm, type = type, psiEig = psiEig)
        

        llUpd = logpdf_GMM(X, gmm).mean()
        llDelta = llUpd - llOld
        llOld = llUpd
        
    return gmm


    
def split_GMM_LBG(gmm, alpha = 0.1):

    gmmOut = []
   
    for (w, mu, C) in gmm:
        U, s, Vh = numpy.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))

    return gmmOut


# train LBG + EM
def train_GMM_LBG_EM(X, numComponents, type = 'full', psiEig = None, epsLLAverage = 1e-6, lbgAlpha = 0.1):

    mu, C = compute_mu_C(X)

    if type.lower() == 'diagonal':
        C = C * numpy.eye(X.shape[0]) 
    
    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C, psiEig))] 
    else:
        gmm = [(1.0, mu, C)] 
    
    while len(gmm) < numComponents:
        
        gmm = split_GMM_LBG(gmm, lbgAlpha)
        gmm = train_GMM_EM(X, gmm, type = type, psiEig = psiEig, epsLLAverage = epsLLAverage)
    return gmm

    