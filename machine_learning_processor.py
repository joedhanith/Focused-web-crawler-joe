import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier



class MachineLearningClassifier(object):
    
    @staticmethod
    def nb_classifier(x, y):
        x= np.array(x).reshape(-1,1)
        y = np.array(y)
        nb = GaussianNB()
        nb.fit(x, y)
        return nb

    @staticmethod
    def mlp_classifier(x, y):
        x= np.array(x).reshape(-1,1)
        y = np.array(y)
        print("X :",x)
        print("Y : ", y)
        ml = MLPClassifier(activation='identity', solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
        ml.fit(x, y)
        return ml

    