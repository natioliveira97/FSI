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
    data = pd.read_csv("car.data") 

    n_classes = 4

    training_data = data.sample(1200)
    test_data = data.drop(training_data.index)

    # Pré-processamento: transforma atributos categóricos em varios atributos binários

    # data_atributes = pd.get_dummies(data[['buying', 'maint', 'doors', 'persons','lug_boot','safety']])
    # print(data_atributes.head())

    

    # Divide dados de treinamento em atributos e classes
    training_atributes = training_data[['buying', 'maint', 'doors', 'persons','lug_boot','safety']]
    training_atributes = pd.get_dummies(training_atributes).values
    training_classes = training_data[['car_type']].transpose().values
    training_classes = np.array(training_classes)[0]

    # Divide dados de teste em atributos e classes
    test_atributes = test_data[['buying', 'maint', 'doors', 'persons','lug_boot','safety']]
    test_atributes = pd.get_dummies(test_atributes).values
    test_classes = test_data[['car_type']].transpose().values
    test_classes = np.array(test_classes)[0]

    # Para o svc faz a abordagem one-vs-rest, one-vs-one e Error-Correcting Output-Codes
    ovr_clf_svc = OneVsRestClassifier(svm.SVC()).fit(training_atributes, training_classes)
    ovr_svc_prediction = ovr_clf_svc.predict(test_atributes)
    print("one_against_rest ",metrics.accuracy_score(test_classes,ovr_svc_prediction))

    ovo_clf_svc = OneVsOneClassifier(svm.SVC()).fit(training_atributes, training_classes)
    ovo_svc_prediction = ovo_clf_svc.predict(test_atributes)
    print("one_against_one ",metrics.accuracy_score(test_classes,ovo_svc_prediction))

    eoc_clf_svc = OutputCodeClassifier(svm.SVC(), code_size =((2 ** (n_classes-1) -1)/n_classes)).fit(training_atributes, training_classes)
    eoc_svc_prediction = eoc_clf_svc.predict(test_atributes)
    print("error correcting code ",metrics.accuracy_score(test_classes,eoc_svc_prediction))




if __name__ == "__main__":
    main()