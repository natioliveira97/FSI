import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics


# Função principal que carrega a base de dados, separa em amostras de treinamento e teste
def main():

    # Carrega dataset
    data = pd.read_csv("car.data") 

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

    # Cria uma instancia de classificador svc (utiliza one_against_one)
    ovo_svc_classifier = svm.SVC(decision_function_shape='ovo')

    # Treina o modelo com os dados de treinamento
    ovo_svc_classifier.fit(training_atributes,training_classes)

    # Prediz os dados de teste
    prediction = ovo_svc_classifier.predict(test_atributes)
    print("one_against_one ",metrics.accuracy_score(test_classes,prediction))


    # Cria uma instancia de classificador svc (utiliza one_against_rest)
    ovr_svc_classifier = svm.SVC(decision_function_shape='ovr')

    # Treina o modelo com os dados de treinamento
    ovr_svc_classifier.fit(training_atributes,training_classes)

    # Prediz os dados de teste
    prediction = ovr_svc_classifier.predict(test_atributes)
    print("one_against_rest ",metrics.accuracy_score(test_classes,prediction))

if __name__ == "__main__":
    main()