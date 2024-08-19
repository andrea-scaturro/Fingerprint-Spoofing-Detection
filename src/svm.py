import numpy
import scipy.special


def vcol(x):
    return x.reshape((x.size, 1))

def vrow(x):
    return x.reshape((1, x.size))





def train_dual_SVM_linear(DTR, LTR, C, K = 1):
    
    ZTR = LTR * 2.0 - 1.0 
    DTR_EXT = numpy.vstack([DTR, numpy.ones((1,DTR.shape[1])) * K])
    
    H = numpy.dot(DTR_EXT.T, DTR_EXT) * vcol(ZTR) * vrow(ZTR)

    
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR_EXT.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)
    
    
    def primalLoss(w_hat):
        S = (vrow(w_hat) @ DTR_EXT).ravel()
        return 0.5 * numpy.linalg.norm(w_hat)**2 + C * numpy.maximum(0, 1 - ZTR * S).sum()

    
    w_hat = (vrow(alphaStar) * vrow(ZTR) * DTR_EXT).sum(1)
    
    
    w, b = w_hat[0:DTR.shape[0]], w_hat[-1] * K 

    primalLoss, dualLoss = primalLoss(w_hat), -fOpt(alphaStar)[0]
    
    
    return w, b



def polyKernel(degree, c):
    
    def polyKernelFunc(D1, D2):
        return (numpy.dot(D1.T, D2) + c) ** degree

    return polyKernelFunc

def rbfKernel(gamma):

    def rbfKernelFunc(D1, D2):

        D1Norms = (D1**2).sum(0)
        D2Norms = (D2**2).sum(0)
        Z = vcol(D1Norms) + vrow(D2Norms) - 2 * numpy.dot(D1.T, D2)
        return numpy.exp(-gamma * Z)

    return rbfKernelFunc


def train_dual_SVM_kernel(DTR, LTR, C, kernelFunc, eps = 1.0):

    ZTR = LTR * 2.0 - 1.0 
    K = kernelFunc(DTR, DTR) + eps
    H = vcol(ZTR) * vrow(ZTR) * K

    
    def fOpt(alpha):
        Ha = H @ vcol(alpha)
        loss = 0.5 * (vrow(alpha) @ Ha).ravel() - alpha.sum()
        grad = Ha.ravel() - numpy.ones(alpha.size)
        return loss, grad

    alphaStar, _, _ = scipy.optimize.fmin_l_bfgs_b(fOpt, numpy.zeros(DTR.shape[1]), bounds = [(0, C) for i in LTR], factr=1.0)

    

    
    def fScore(DTE):
        
        K = kernelFunc(DTR, DTE) + eps
        H = vcol(alphaStar) * vcol(ZTR) * K
        return H.sum(0)

    return fScore 

