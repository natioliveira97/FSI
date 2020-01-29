import pandas as pd
import numpy as np
import random
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
from mnist import MNIST

class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Função que plota o gráfico da matriz de confusão das classes preditas
def plot_confusion_matrix(y_true, y_pred, classes, title):
    # Calcula acurácia
    acc = accuracy_score(y_true, y_pred)
    title = title + " (Acurácia: " + str("{:10.4f}".format(acc)) + ")"

    # Constroi matriz de confusão
    cm = confusion_matrix(y_true, y_pred, classes, normalize='true')
    cm_df = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize=(5.5,4))
    sns.heatmap(cm_df, annot=True, cmap="YlGnBu")
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Carrega imagens de treinamento na memória
def load_training():

    # Carrega dataset
    print("Carregando dados de treinamento na memória...")
    t0 = time.time()
    mndata = MNIST('/home/natalia/Unb/FSI/mnist_base')
    training_images, training_labels = mndata.load_training()

    # Divide dados de treinamento em atributos e classes
    training_images = pd.DataFrame(training_images)
    training_labels = np.array(training_labels)
    training_images = training_images.sample(60000)
    training_labels = training_labels[training_images.index]

    t1 = time.time()
    print("Carregamento das imagens de treinamento demorou ", t1-t0, " segundos")

    return training_images, training_labels

# Carrega imagens de teste na memória
def load_test():

    # Carrega dataset
    print("Carregando dados de teste na memória...")
    t0 = time.time()
    mndata = MNIST('/home/natalia/Unb/FSI/mnist_base')
    test_images, test_labels = mndata.load_testing()

    # Divide dados de treinamento em atributos e classes
    test_images = pd.DataFrame(test_images)
    test_labels = np.array(test_labels)

    t1 = time.time()
    print("Carregamento das imagens de teste demorou ", t1-t0, " segundos")

    return test_images, test_labels

# Faz treinamento
def training(clf, filename):
    training_images,training_labels = load_training()

    # Treina classificador
    t0 = time.time()
    clf.fit(training_images, training_labels)
    t1 = time.time()
    print("O treinamento demorou", t1-t0, " segundos")
    pickle.dump(clf, open((filename+".sav"), 'wb'))

# Carrega o classificador especificado realiza a fase de teste
def test(filename):
    test_images, test_labels = load_test()

    clf = pickle.load(open((filename+".sav"), 'rb'))
    prediction = clf.predict(test_images)
    plot_confusion_matrix(test_labels, prediction, classes=class_names,title='Confusion matrix ')

# Menu
def menu(mode, mult_clf_mode, bin_clf_mode):

    # Define o classificador binário
    if(bin_clf_mode == "--svc"):
        bin_clf = svm.SVC(class_weight = 'balanced')
        filename = "svc"
    elif(bin_clf_mode == "--mlp"):
        bin_clf = MLPClassifier()
        filename = "mlp"
    else:
        print("Escolha o terceiro argumento como --svc ou --mlp")
        exit()

    # Define o classificador multiclasse
    if(mult_clf_mode == "--ovr"):
        mult_clf = OneVsRestClassifier(bin_clf, n_jobs = -1)
        filename = "ovr_"+filename
    elif(mult_clf_mode == "--ovo"):
        mult_clf = OneVsOneClassifier(bin_clf, n_jobs = -1)
        filename = "ovo_"+filename
    elif(mult_clf_mode == "--eoc"):
        mult_clf = OutputCodeClassifier(bin_clf,code_size =3.0, n_jobs=-1)
        filename = "eoc_"+filename
    else:
        print("Escolha o segundo argumento como --ovr ou --ovo ou --eoc")
        exit()

    if(mode == "--train"):
        training(mult_clf,filename)
    elif(mode == "--test"):
        test(filename)
    else:
        print("Escolha o primeiro argumento como --train ou --test")
        exit()
        

def main():
    if(len(sys.argv)  < 4):
        print("Selecione suas opções:")
        print("1. --train --test")
        print("2. --ovo --ovr --eoc")
        print("3. --svc --per") 
        exit()

    menu(sys.argv[1], sys.argv[2], sys.argv[3])


if __name__ == "__main__":
    main()