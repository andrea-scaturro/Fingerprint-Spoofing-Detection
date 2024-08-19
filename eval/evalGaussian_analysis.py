import numpy 
import matplotlib
import matplotlib.pyplot as plt
import scipy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'trainData.txt'))


from dataset.dataset import *
from src.dimreduction import *
from src.classifier_pca_lda import *
from src.plotter import *
from src.gaussian_analysis import *

#======================= Main Program ==========================

if __name__ == '__main__':

    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load(data_path)

    fit_and_plot_gaussian2(D,L)

 

