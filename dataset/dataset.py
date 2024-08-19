import numpy 
import matplotlib
import matplotlib.pyplot as plt
import scipy

# mcol(v): Questa funzione prende un vettore v e restituisce una matrice con una colonna. 
# È utile per cambiare la forma di un vettore in una matrice colonna.


def mcol(v):
    return v.reshape((v.size, 1))   


def mrow(v):
   
    return v.reshape((1, v.size))


def vcol(v):
    return v.reshape((v.size, 1))   


def vrow(v):
   
    return v.reshape((1, v.size))



#load(fname): Questa funzione carica il dataset Iris da un file CSV specificato da fname.
#  Viene utilizzato un dizionario hLabels per mappare i nomi delle classi 
# ("Iris-setosa", "Iris-versicolor", "Iris-virginica") ai loro rispettivi 
# identificatori numerici (0, 1, 2). Il dataset viene quindi caricato in due array 
# numpy: uno per le feature (DList) e uno per le etichette di classe (labelsList).


def load(fname):
    DList = []
    labelsList = []

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]   # prende tutti tranne l'ultimo
                
     #Questa riga crea un array numpy contenente le feature di un singolo esempio nel dataset Iris.
                attrs = mcol(numpy.array([float(i) for i in attrs]))

                label = line.split(',')[-1].strip()

    # DList.append(attrs) e labelsList.append(label): Queste righe aggiungono le feature e 
    # l'etichetta dell'esempio corrente alle liste DList e labelsList, rispettivamente.
    #DList contiene le feature di tutti gli esempi nel dataset.
    #labelsList contiene le etichette di classe corrispondenti agli esempi nel dataset.
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    #numpy.hstack(DList):la funzione numpy.hstack() per concatenare
    # orizzontalmente (lungo l'asse delle colonne) gli array contenuti nella lista DList. 
   # quindi restituirà una matrice con colonne l'elemento e righe i valori di tutte le feature

    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)


#load2(): Questa funzione carica il dataset Iris utilizzando la libreria scikit-learn. 
# Restituisce una tupla contenente le feature e le etichette di classe.

def load2():

    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']




def split_db_2to1(D, L, seed=0):

    nTrain = int(D.shape[1]*2.0/3.0) 
    numpy.random.seed(seed)
    idx = numpy.random.permutation(D.shape[1]) 
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DVAL = D[:, idxTest]
    LTR = L[idxTrain]
    LVAL = L[idxTest]
    return (DTR, LTR), (DVAL, LVAL)



