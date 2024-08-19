import matplotlib.pyplot as plt
import numpy


# compute ll


def logpdf_single(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu).T @ P @ (x-mu)).ravel()

# slow version
def logpdf_GAU_ND(X, mu, C):
    ll = [logpdf_single(X[:, i:i+1], mu, C) for i in range(X.shape[1])]
    return numpy.array(ll).ravel()


# fast version
def compute_ll(x, mu, C):
    P = numpy.linalg.inv(C)
    return -0.5*x.shape[0]*numpy.log(numpy.pi*2) - 0.5*numpy.linalg.slogdet(C)[1] - 0.5 * ((x-mu) * (P @ (x-mu))).sum(0)


