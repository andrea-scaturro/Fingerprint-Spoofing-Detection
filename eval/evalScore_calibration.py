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
from src.score_calibration import *
from src.fusion import *



# ======================= Main Program ==========================

if __name__ == '__main__':

    D, L = load(data_path)
    m = 6

   
    (DTR, LTR), (DVAL, LVAL) = split_db_2to1(D, L)


    for model in ['logReg','gmm', 'svm']:
        
        #scoreCalibration_findPt(DTR,LTR,DVAL,LVAL, model) #find the best value for prior
        calibrateScores,labelsCalibrate=scoreCalibration(DTR,LTR,DVAL,LVAL, model)

        minDCF_cal = compute_minDCF( DVAL,labelsCalibrate,calibrateScores, 0.1, 1.0, 1.0)
        dcf_cal  =  compute_actDCF(DVAL,labelsCalibrate,calibrateScores, 0.1, 1.0, 1.0)

        print ('minDCF, calibrate: %.3f' % minDCF_cal) 
        print ('actDCF, calibrate: %.3f' % dcf_cal)
        

         

