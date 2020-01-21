import pandas as pd
import numpy as np
import random

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn import metrics
from mnist import MNIST




# Função principal que carrega a base de dados, separa em amostras de treinamento e teste
def main():

    # Carrega dataset
    mndata = MNIST('/home/natalia/Unb/FSI/mnist_base')

    training_images, training_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    training_images = pd.DataFrame(training_images)
    training_images = training_images.sample(10000)
    training_labels = np.array(training_labels)
    training_labels = training_labels[training_images.index]

    test_images = pd.DataFrame(test_images)
    test_images = test_images.sample(1000)
    test_labels = np.array(test_labels)
    test_labels = test_labels[test_images.index]

    # Para o svc faz a abordagem one-vs-rest, one-vs-one e Error-Correcting Output-Codes
    print("Usando o classificador svc:")

    ovr_clf_svc = OneVsRestClassifier(svm.SVC()).fit(training_images, training_labels)
    ovr_svc_prediction = ovr_clf_svc.predict(test_images)
    print("one_against_rest ", metrics.accuracy_score(test_labels,ovr_svc_prediction))



    




if __name__ == "__main__":
    main()