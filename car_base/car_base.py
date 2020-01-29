import pandas as pd
import numpy as np
import seaborn as sns
import sys
import pickle
import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix

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

    plot_confusion_matrix(test_classes, eoc_per_prediction, classes=class_names,title='Confusion matrix for Perceptron')

def training(clf,filename):
    # Carrega dataset
    data = pd.read_csv("car.data") 
    training_data = data.sample(n_traning_samples)
    pickle.dump(training_data.index, open((filename+"_index.sav"), 'wb'))

    # Divide dados de treinamento em atributos e classes
    training_atributes = training_data[atribute_names]
    training_atributes = pd.get_dummies(training_atributes).values
    training_labels = training_data[['car_type']].transpose().values
    training_labels = np.array(training_labels)[0]

    t0 = time.time()
    clf.fit(training_atributes, training_labels)
    t1 = time.time()
    print("O treinamento demorou", t1-t0, " segundos")
    pickle.dump(clf, open((filename+".sav"), 'wb'))

def test(filename):
    #Carrega dataset
    data = pd.read_csv("car.data") 
    training_data_index = pickle.load(open((filename+"_index.sav"), 'rb'))

    test_data = data.drop(training_data_index)

    # Divide dados de teste em atributos e classes
    test_atributes = test_data[atribute_names]
    test_atributes = pd.get_dummies(test_atributes).values
    test_labels = test_data[['car_type']].transpose().values
    test_labels = np.array(test_labels)[0]

    # Faz as predições
    clf = pickle.load(open("eoc_clf_svc.sav", 'rb'))
    t0 = time.time()
    prediction = clf.predict(test_atributes)
    t1 = time.time()
    print("O teste demorou", t1-t0, " segundos")
    plot_confusion_matrix(test_labels, prediction, classes=class_names,title='Confusion matrix ')

# Menu
def menu(mult_clf_mode, bin_clf_mode):

    # Define o classificador binário
    if(bin_clf_mode == "--svc"):
        bin_clf = svm.SVC(class_weight = 'balanced')
        filename = "svc"
    elif(bin_clf_mode == "--mlp"):
        bin_clf = MLPClassifier()
        filename = "mlp"
    else:
        print("Escolha o segundo argumento como --svc ou --mlp")
        exit()

    # Define o classificador multiclasse
    if(mult_clf_mode == "--ovr"):
        mult_clf = OneVsRestClassifier(bin_clf, n_jobs = -1)
        filename = "ovr_"+filename
    elif(mult_clf_mode == "--ovo"):
        mult_clf = OneVsOneClassifier(bin_clf, n_jobs = -1)
        filename = "ovo_"+filename
    elif(mult_clf_mode == "--eoc"):
        OutputCodeClassifier(bin_clf,code_size =((2 ** (n_classes-1) -1)/n_classes), n_jobs=-1)
        filename = "eoc_"+filename
    else:
        print("Escolha o primeiro argumento como --ovr ou --ovo ou --eoc")
        exit()

    training(mult_clf,filename)
    test(filename)

def main():
    if(len(sys.argv)  < 3):
        print("Selecione suas opções:")
        print("1. --ovo --ovr --eoc")
        print("2. --svc --per") 
        exit()

    menu(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()