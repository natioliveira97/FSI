import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
import sys
import pickle
import time
import matplotlib.pyplot as plt

n_classes = 4
n_samples = 1728
n_traning_samples = int(0.7*n_samples)
atribute_names = ['buying', 'maint', 'doors', 'persons','lug_boot','safety']
class_names = ['unacc', 'acc', 'good', 'vgood']

def plot_confusion_matrix(y_true, y_pred, classes, title):
    acc = accuracy_score(y_true, y_pred)
    title = title + " (Acurácia: " + str("{:10.4f}".format(acc)) + ")"

    cm = confusion_matrix(y_true, y_pred, classes, normalize='true')
    cm_df = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize=(5.5,4))
    sns.heatmap(cm_df, annot=True, cmap="YlGnBu")
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def ovr_training():

    # Carrega dataset
    data = pd.read_csv("car.data") 
    training_data = data.sample(n_traning_samples)
    pickle.dump(training_data.index, open("ovr_training_index.sav", 'wb'))

    # Divide dados de treinamento em atributos e classes
    training_atributes = training_data[atribute_names]
    training_atributes = pd.get_dummies(training_atributes).values
    training_classes = training_data[['car_type']].transpose().values
    training_classes = np.array(training_classes)[0]

    # Treina classificador SVC
    t0 = time.time()
    ovr_clf_svc = OneVsRestClassifier(svm.SVC()).fit(training_atributes, training_classes)
    t1 = time.time()
    print("O treinamento One vs Rest com SVC demorou", t1-t0, " segundos")
    pickle.dump(ovr_clf_svc, open("ovr_clf_svc.sav", 'wb'))

    # Treina classificador Perceptron
    t0 = time.time()
    ovr_clf_per = OneVsRestClassifier(Perceptron(tol=1e-3, random_state=0)).fit(training_atributes, training_classes)
    t1 = time.time()
    print("O treinamento One vs Rest com Perceptron demorou", t1-t0, " segundos")
    pickle.dump(ovr_clf_per, open("ovr_clf_per.sav", 'wb'))

def ovo_training():
    # Carrega dataset
    data = pd.read_csv("car.data") 
    training_data = data.sample(n_traning_samples)
    pickle.dump(training_data.index, open("ovo_training_index.sav", 'wb'))

    # Divide dados de treinamento em atributos e classes
    training_atributes = training_data[atribute_names]
    training_atributes = pd.get_dummies(training_atributes).values
    training_classes = training_data[['car_type']].transpose().values
    training_classes = np.array(training_classes)[0]

    # Treina classificador SVC
    t0 = time.time()
    ovo_clf_svc = OneVsOneClassifier(svm.SVC()).fit(training_atributes, training_classes)
    t1 = time.time()
    print("O treinamento One vs One com SVC demorou", t1-t0, " segundos")
    pickle.dump(ovo_clf_svc, open("ovo_clf_svc.sav", 'wb'))

    # Treina classificador Perceptron
    t0 = time.time()
    ovo_clf_per = OneVsOneClassifier(Perceptron(tol=1e-3, random_state=0)).fit(training_atributes, training_classes)
    t1 = time.time()
    print("O treinamento One vs One com Perceptron demorou", t1-t0, " segundos")
    pickle.dump(ovo_clf_per, open("ovo_clf_per.sav", 'wb'))

def eoc_training():
    # Carrega dataset
    data = pd.read_csv("car.data") 
    training_data = data.sample(n_traning_samples)
    pickle.dump(training_data.index, open("eoc_training_index.sav", 'wb'))

    # Divide dados de treinamento em atributos e classes
    training_atributes = training_data[atribute_names]
    training_atributes = pd.get_dummies(training_atributes).values
    training_classes = training_data[['car_type']].transpose().values
    training_classes = np.array(training_classes)[0]

    # Treina classificador SVC
    t0 = time.time()
    eoc_clf_svc = OutputCodeClassifier(svm.SVC(), code_size =((2 ** (n_classes-1) -1)/n_classes)).fit(training_atributes, training_classes)
    t1 = time.time()
    print("O treinamento Error Output Code com SVC demorou", t1-t0, " segundos")
    pickle.dump(eoc_clf_svc, open("eoc_clf_svc.sav", 'wb'))

    # Treina classificador Perceptron
    t0 = time.time()
    eoc_clf_per = OutputCodeClassifier(Perceptron(tol=1e-3, random_state=0),code_size =((2 ** (n_classes-1) -1)/n_classes)).fit(training_atributes, training_classes)
    t1 = time.time()
    print("O treinamento Error Output Code com Perceptron demorou", t1-t0, " segundos")
    pickle.dump(eoc_clf_per, open("eoc_clf_per.sav", 'wb'))

def ovr_test():
    data = pd.read_csv("car.data") 
    training_data_index = pickle.load(open("ovr_training_index.sav", 'rb'))

    test_data = data.drop(training_data_index)

    # Divide dados de teste em atributos e classes
    test_atributes = test_data[atribute_names]
    test_atributes = pd.get_dummies(test_atributes).values
    test_classes = test_data[['car_type']].transpose().values
    test_classes = np.array(test_classes)[0]

    # Predições para SVC
    ovr_clf_svc = pickle.load(open("ovr_clf_svc.sav", 'rb'))
    ovr_svc_prediction = ovr_clf_svc.predict(test_atributes)
    plot_confusion_matrix(test_classes, ovr_svc_prediction, classes=class_names,title='Confusion matrix for SVC')
    
    # Predições para Perceptron
    ovr_clf_per = pickle.load(open("ovr_clf_per.sav", 'rb'))
    ovr_per_prediction = ovr_clf_per.predict(test_atributes)
    plot_confusion_matrix(test_classes, ovr_per_prediction, classes=class_names,title='Confusion matrix for Perceptron')

def ovo_test():
    data = pd.read_csv("car.data") 
    training_data_index = pickle.load(open("ovo_training_index.sav", 'rb'))

    test_data = data.drop(training_data_index)

    # Divide dados de teste em atributos e classes
    test_atributes = test_data[atribute_names]
    test_atributes = pd.get_dummies(test_atributes).values
    test_classes = test_data[['car_type']].transpose().values
    test_classes = np.array(test_classes)[0]

    # Predições para SVC
    ovo_clf_svc = pickle.load(open("ovo_clf_svc.sav", 'rb'))
    ovo_svc_prediction = ovo_clf_svc.predict(test_atributes)
    plot_confusion_matrix(test_classes, ovo_svc_prediction, classes=class_names,title='Confusion matrix for SVC')

    # Predições para Perceptron
    ovo_clf_per = pickle.load(open("ovo_clf_per.sav", 'rb'))
    ovo_per_prediction = ovo_clf_per.predict(test_atributes)
    plot_confusion_matrix(test_classes, ovo_per_prediction, classes=class_names,title='Confusion matrix for Perceptron')

def eoc_test():
    data = pd.read_csv("car.data") 
    training_data_index = pickle.load(open("eoc_training_index.sav", 'rb'))

    test_data = data.drop(training_data_index)

    # Divide dados de teste em atributos e classes
    test_atributes = test_data[atribute_names]
    test_atributes = pd.get_dummies(test_atributes).values
    test_classes = test_data[['car_type']].transpose().values
    test_classes = np.array(test_classes)[0]

    # Predições para SVC
    eoc_clf_svc = pickle.load(open("eoc_clf_svc.sav", 'rb'))
    eoc_svc_prediction = eoc_clf_svc.predict(test_atributes)
    plot_confusion_matrix(test_classes, eoc_svc_prediction, classes=class_names,title='Confusion matrix for SVC')

    # Predições para Perceptron 
    eoc_clf_per = pickle.load(open("ovo_clf_per.sav", 'rb'))
    eoc_per_prediction = eoc_clf_per.predict(test_atributes)
    plot_confusion_matrix(test_classes, eoc_per_prediction, classes=class_names,title='Confusion matrix for Perceptron')

# Menu
def main():

    if(len(sys.argv)  < 2):
        print("Selecione sua opção (--ovo, --ovr, --eoc)")       
    

    if(sys.argv[1] == '--ovo'):
        print("ONE vs ONE\n")
        ovo_training()
        ovo_test()
    elif(sys.argv[1] == '--ovr'):
        print("ONE vs REST\n")
        ovr_training()
        ovr_test()
    elif(sys.argv[1] == '--eoc'):
        print("ERROR OUTPUT CODE\n")
        eoc_training()
        eoc_test()
    else:
        print("Não há essa opção")

if __name__ == "__main__":
    main()