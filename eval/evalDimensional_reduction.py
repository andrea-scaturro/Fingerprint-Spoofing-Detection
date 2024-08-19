import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset', 'trainData.txt'))


from dataset.dataset import load, split_db_2to1
from src.dimreduction import *
from src.classifier_pca_lda import *
from src.plotter import *

#======================= Main Program ==========================

if __name__ == '__main__':
    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load(data_path)
    m = 5
    
    # PCA
    pca_stats(D, L, m)

    # LDA
    lda_stats(D, L, m)

    # DTR and LTR are model training data and labels 
    # DVAL and LVAL are validation data and labels
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)

    # LDA
    PVAL = lda_classifier(DTR, LTR, DVAL, LVAL, m)
    print("Error LDA: ", PVAL)

    # PCA
    PVAL = pca_classifier(DTR, LTR, DVAL, LVAL, m)
    print("Error PCA: ", PVAL)

    for i in range(1,7):
        print('')
        print('m: ', i)
        
        # PCA - LDA
        PVAL = pca_lda_classifier(DTR, LTR, DVAL, LVAL, i)
        
        num_errors, num_right = count_errors(LVAL, PVAL)
        print("Errors: ", num_errors)
        print("Accuracy: ", num_right)
        print('')
