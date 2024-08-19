# Fingerprint Spoofing Detection

<br>
This project focuses on solving a binary classification problem with the goal of detecting fingerprint forgeries. The objective is to distinguish between authentic and counterfeit fingerprint images.

----------------------------
<br> </br>

## Overview

The dataset used in this project consists of labeled samples categorized into two classes:
- **Authentic (True, label 1)**
- **Fake (False, label 0)**

The data is generated through a feature extractor, which synthesizes the high-level properties of fingerprint images, resulting in a six-dimensional feature space. The challenge lies in accurately classifying these samples to identify whether a fingerprint is genuine or counterfeit.

<br> </br>

## Dataset

- **Number of features:** 6
- **Classes:** 
  - Label 1: Authentic (True)
  - Label 0: Fake (False)


<br> </br>
## Structure



├── dataset/ # Folder containing the dataset  
├── src/ # Source code for data processing and model implementation  
├── eval/ # Scripts for model evaluation and analysis  
├── README.md # Project documentation  
└── requirements.txt # Required dependencies  



<br> </br>

## How to Run

1. **Clone the repository.** In your terminal, use Git to clone the repository to your local machine.
      

        git clone https://github.com/andrea-scaturro/Fingerprint-Spoofing-Detection.git

2. **Navigate to the project directory.** Once the repository is cloned, change your directory to the project folder.

3. **Install the required dependencies.** Use `pip` to install all the necessary Python libraries listed in `requirements.txt`.
      
        pip install -r requirements.txt

4. **Evaluate the model.**
    - To perform all analyses, run the files located in the eval directory.
    - To see the results on the final model, run the project.py file in the src/ directory.



<br></br>

## Results

The integration of the models led to a significant improvement in performance, evidenced by the actDCF and minDCF values. The fusion of GMM, SVM and logistic regression achieved the best performance with an actDCF of 0.06403 and a minDCF of 0.0609, demonstrating good model calibration and fit. In addition, the analysis revealed that the final model showed robustness across different applications, maintaining effective performance despite variations in error costs. These results confirm the effectiveness of the system in detecting fingerprint forgeries.


<br></br>

  ### *Training*

  |     Model                     |     minDCF     |     actDCF     |
  |-------------------------------|----------------|----------------|
  | GMM-SVM-Logistic Regression   | Riga 1 Col 2   | Riga 1 Col 3   |
  | GMM-SVM                       | Riga 2 Col 2   | Riga 2 Col 3   |
  | GMM-Logistic Regression       | Riga 3 Col 2   | Riga 3 Col 3   |
  | SVM-Logistic Regression       | Riga 3 Col 2   | Riga 3 Col 3   |

  <br>


  ### *Evaluation*

  |     Model                     |     minDCF     |     actDCF     |
  |-------------------------------|----------------|----------------|
  | GMM-SVM-Logistic Regression   | Riga 1 Col 2   | Riga 1 Col 3   |
  | GMM-SVM                       | Riga 2 Col 2   | Riga 2 Col 3   |
  | GMM-Logistic Regression       | Riga 3 Col 2   | Riga 3 Col 3   |
  | SVM-Logistic Regression       | Riga 3 Col 2   | Riga 3 Col 3   |
