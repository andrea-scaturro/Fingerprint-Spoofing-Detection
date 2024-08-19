import numpy 
import matplotlib
import matplotlib.pyplot as plt
import scipy

from dataset.dataset import *
from .dimreduction import *
from .classifier_pca_lda import *
from .gaussian_analysis import *
from .stats import *



def plot_hist(D, L,title=''):

    D0 = D[:, L==0]
    D1 = D[:, L==1]


    hFea = {
        0: 'feature 0',
        1: 'feature 1',
        2: 'feature 2',
        3: 'feature 3',
        4: 'feature 4',
        5: 'feature 5'
        }


    for dIdx in range(D.shape[0]):
        plt.figure()
        plt.xlabel(hFea[dIdx])
        plt.hist(D0[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Fake')
        plt.hist(D1[dIdx, :], bins = 10, density = True, alpha = 0.4, label = 'Genuine')
    
        plt.title(title)
        plt.legend()
        plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
        #plt.savefig('plot/plot_'+title+'/hist_%d.pdf' % dIdx)
    plt.show()

def plot_scatter(D, L,title=''):
    
    D0 = D[:, L==0]
    D1 = D[:, L==1]
   

    hFea = {
        0: 'feature 0',
        1: 'feature 1',
        2: 'feature 2',
        3: 'feature 3',
        4: 'feature 4',
        5: 'feature 5'
        }

    for dIdx1 in range(D.shape[0]):
        for dIdx2 in range(6):
            if dIdx1 == dIdx2:
                continue
            plt.figure()
            plt.xlabel(hFea[dIdx1])
            plt.ylabel(hFea[dIdx2])
            plt.scatter(D0[dIdx1, :], D0[dIdx2, :], label = 'Fake')
            plt.scatter(D1[dIdx1, :], D1[dIdx2, :], label = 'Genuine')
         
        
            plt.legend()
            plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
            #plt.savefig('plot/scatter_%d_%d.pdf' % (dIdx1, dIdx2))
        plt.show()



def plot_hist_lab3(D, L,title=''):

    D0 = D[:, L==0]
    D1 = D[:, L==1]


    plt.figure()
    plt.hist(D0[0], bins = 5, density = True, alpha = 0.4, label = 'False')
    plt.hist(D1[0], bins = 5, density = True, alpha = 0.4, label = 'True')
    plt.title(title) 
    plt.legend()
    plt.tight_layout() # Use with non-default font size to keep axis label inside the figure
      #  plt.savefig('hist_%d.pdf' % dIdx)
    plt.show()



# plottiamo il dataset diviso per avere colori diversi nella legenda
def plot_scatter_lab3(D, L, title=''):    
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]    
   
    plt.scatter(D0[0], (-1)*D0[1], label='Setosa')    
    plt.scatter(D1[0], (-1)*D1[1], label='Versicolor')
    
    plt.legend()
    plt.tight_layout()  # Use with non-default font size to keep axis label inside the figure
    plt.title(title)
    #plt.savefig('hist_%s.pdf' % title)
    plt.show()








# For Gaussian Analysis

def fit_and_plot_gaussian(D, L):

    
    for label in numpy.unique(L):
        
        for feature_idx in range(D.shape[0]):
          
            # Selezioniamo gli esempi corrispondenti a quella classe
            D_class = D[feature_idx, L == label]
            D_class = numpy.reshape(D_class, (1, -1))
           
            m_ML, C_ML = compute_mu_C(D_class)

            plt.figure()
            plt.title("Feature: "+ str(feature_idx) +"  Label: "+ str(label))
            plt.hist(D_class.ravel(), bins=50, density=True)
            XPlot = numpy.linspace(D_class.min(), D_class.max(), 1000)
            plt.plot(XPlot.ravel(), numpy.exp(logpdf_GAU_ND(vrow(XPlot), m_ML, C_ML)))
            #plt.savefig('plot/plot_gau_density/hist_%d_%d.pdf' % (feature_idx, label))
            plt.show()


def fit_and_plot_gaussian2(D, L):


#plot for each feature
    for feature_idx in range(D.shape[0]):
        
        
        D_0 = D[feature_idx, L == 0]
        D_1 = D[feature_idx, L == 1]

        D_0 = numpy.reshape(D_0, (1, -1))
        D_1 = numpy.reshape(D_1, (1, -1))
        
        # Compute mean and variance
        m_ML_0, C_ML_0 = compute_mu_C(D_0)
        m_ML_1, C_ML_1 = compute_mu_C(D_1)
        
        # Ist class 0
        plt.figure()
        plt.title("Feature: " + str(feature_idx))
        plt.hist(D_0.ravel(), bins=50, density=True, alpha=0.5, label='Fake')
        
        # Gau class 0
        XPlot_0 = numpy.linspace(D_0.min(), D_0.max(), 1000)
        plt.plot(XPlot_0, numpy.exp(logpdf_GAU_ND(vrow(XPlot_0), m_ML_0, C_ML_0)), color='blue', linewidth=2, label='Gau Fake')
        
         # Ist class 1
        plt.hist(D_1.ravel(), bins=50, density=True, alpha=0.5, label='Genuine')
        
        # Gau class 1
        XPlot_1 = numpy.linspace(D_1.min(), D_1.max(), 1000)
        plt.plot(XPlot_1, numpy.exp(logpdf_GAU_ND(vrow(XPlot_1), m_ML_1, C_ML_1)), color='red', linewidth=2, label='Gau Genuine')
        
        plt.legend(fontsize='small')
        plt.savefig('plot/plot_gau_density/hist_%d.pdf' % feature_idx)
        plt.show()
