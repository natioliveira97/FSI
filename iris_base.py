import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn import metrics


# Função principal que carrega a base de dados, separa em amostras de treinamento e teste
def main():

    # Carrega dataset
    data = pd.read_csv("iris.data")

    training_data = data.sample(100)
    test_data = data.drop(training_data.index)

    # Divide dados de treinamento em atributos e classes
    training_atributes = training_data[['sepal.length', 'sepal.width', 'pedal.length', 'pedal.width']].values
    training_classes = training_data[['species']].transpose().values
    training_classes = np.array(training_classes)[0]

    # Divide dados de teste em atributos e classes
    test_atributes = test_data[['sepal.length', 'sepal.width', 'pedal.length', 'pedal.width']].values
    test_classes = test_data[['species']].transpose().values
    test_classes = np.array(test_classes)[0]

    
    # Para o svc faz a abordagem one-vs-rest, one-vs-one e Error-Correcting Output-Codes
    ovr_clf_svc = OneVsRestClassifier(svm.SVC()).fit(training_atributes, training_classes)
    ovr_svc_prediction = ovr_clf_svc.predict(test_atributes)
    print("one_against_rest ",metrics.accuracy_score(test_classes,ovr_svc_prediction))

    ovo_clf_svc = OneVsOneClassifier(svm.SVC()).fit(training_atributes, training_classes)
    ovo_svc_prediction = ovo_clf_svc.predict(test_atributes)
    print("one_against_one ",metrics.accuracy_score(test_classes,ovo_svc_prediction))

    eoc_clf_svc = OutputCodeClassifier(svm.SVC(), code_size = 1.0).fit(training_atributes, training_classes)
    eoc_svc_prediction = eoc_clf_svc.predict(test_atributes)
    print("error correcting code ",metrics.accuracy_score(test_classes,eoc_svc_prediction))


if __name__ == "__main__":
    main()

