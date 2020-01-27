import pandas as pd
import numpy as np
import random
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from mnist import MNIST
import sys
import pickle
import time
import matplotlib.pyplot as plt

class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

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

def load_training():
    print("Carregando dados de treinamento na memória...")

    # Carrega dataset
    t0 = time.time()
    mndata = MNIST('/home/natalia/Unb/FSI/mnist_base')
    training_images, training_labels = mndata.load_training()

    # Divide dados de treinamento em atributos e classes
    training_images = pd.DataFrame(training_images)
    training_images = training_images.sample(60000)
    training_labels = np.array(training_labels)
    training_labels = training_labels[training_images.index]
    t1 = time.time()
    print("Carregamento das imagens de treinamento demorou ", t1-t0, " segundos")

    return training_images, training_labels

def load_test():
    print("Carregando dados de teste na memória...")

    # Carrega dataset
    t0 = time.time()
    mndata = MNIST('/home/natalia/Unb/FSI/mnist_base')
    test_images, test_labels = mndata.load_testing()

    # Divide dados de treinamento em atributos e classes
    test_images = pd.DataFrame(test_images)
    # test_images = test_images.sample(10000)
    test_labels = np.array(test_labels)
    #test_labels = test_labels[test_images.index]
    t1 = time.time()
    print("Carregamento das imagens de teste demorou ", t1-t0, " segundos")

    return test_images, test_labels

def ovr_svc_training():
    training_images,training_labels = load_training()

    # Treina classificador SVC
    t0 = time.time()
    ovr_clf_svc = OneVsRestClassifier(svm.SVC(class_weight = 'balanced')).fit(training_images, training_labels)
    t1 = time.time()
    print("O treinamento One vs Rest com SVC demorou", t1-t0, " segundos")
    pickle.dump(ovr_clf_svc, open("ovr_clf_svc.sav", 'wb'))

def ovr_per_training():
    training_images,training_labels = load_training()

    # Treina classificador Perceptron
    t0 = time.time()
    ovr_clf_per = OneVsRestClassifier(Perceptron()).fit(training_images, training_labels)
    t1 = time.time()
    print("O treinamento One vs Rest com Perceptron demorou", t1-t0, " segundos")
    pickle.dump(ovr_clf_per, open("ovr_clf_per.sav", 'wb'))

def ovo_svc_training():
    training_images,training_labels = load_training()

    # Treina classificador SVC
    t0 = time.time()
    ovo_clf_svc = OneVsOneClassifier(svm.SVC(class_weight = 'balanced')).fit(training_images, training_labels)
    t1 = time.time()
    print("O treinamento One vs One com SVC demorou", t1-t0, " segundos")
    pickle.dump(ovo_clf_svc, open("ovo_clf_svc.sav", 'wb'))

def ovo_per_training():
    training_images,training_labels = load_training()

    # Treina classificador Perceptron
    t0 = time.time()
    ovo_clf_per = OneVsOneClassifier(Perceptron(tol=1e-3, random_state=0)).fit(training_images, training_labels)
    t1 = time.time()
    print("O treinamento One vs One com Perceptron demorou", t1-t0, " segundos")
    pickle.dump(ovo_clf_per, open("ovo_clf_per.sav", 'wb'))

def eoc_svc_training():
    training_images,training_labels = load_training()

    # Treina classificador SVC
    t0 = time.time()
    eoc_clf_svc = OutputCodeClassifier(svm.SVC(class_weight = 'balanced'),code_size = 3.0).fit(training_images, training_labels)
    t1 = time.time()
    print("O treinamento Error Output Code com SVC demorou", t1-t0, " segundos")
    pickle.dump(eoc_clf_svc, open("eoc_clf_svc.sav", 'wb'))

def eoc_per_training():
    training_images,training_labels = load_training()

    # Treina classificador Perceptron
    t0 = time.time()
    eoc_clf_per = OutputCodeClassifier(Perceptron(tol=1e-3, random_state=0),code_size = 3.0).fit(training_images, training_labels)
    t1 = time.time()
    print("O treinamento Error Output Code com Perceptron demorou", t1-t0, " segundos")
    pickle.dump(eoc_clf_per, open("eoc_clf_per.sav", 'wb'))

def test(filename):
    test_images, test_labels = load_test()

    clf = pickle.load(open(filename, 'rb'))
    prediction = clf.predict(test_images)
    plot_confusion_matrix(test_labels, prediction, classes=class_names,title='Confusion matrix ')



# Função principal que carrega a base de dados, separa em amostras de treinamento e teste
def main():

    if(len(sys.argv)  < 4):
        print("Selecione suas opções:")
        print("1. --train --test")
        print("2. --ovo --ovr --eoc")
        print("3. --svc --per") 
        exit()

    if(sys.argv[1] == '--train'):
        if(sys.argv[2] == '--ovo'):
            if (sys.argv[3] == '--svc'):
                ovo_svc_training()
            elif (sys.argv[3] == '--per'):
                ovo_per_training()
            else:
                print("Escolha por último se deseja --scv ou --per")
                exit()
        elif(sys.argv[2] == '--ovr'):
            if (sys.argv[3] == '--svc'):
                ovr_svc_training()
            elif (sys.argv[3] == '--per'):
                ovr_per_training()
            else:
                print("Escolha por último se deseja --scv ou --per")
                exit()
        elif(sys.argv[2] == '--eoc'):
            if (sys.argv[3] == '--svc'):
                eoc_svc_training()
            elif (sys.argv[3] == '--per'):
                eoc_per_training()
            else:
                print("Escolha por último se deseja --scv ou --per")
                exit()
        else:
            print("Escolha em seguida se deseja --ovo ou --ovr ou --eoc")
    elif(sys.argv[1] == '--test'):
        if(sys.argv[2] == '--ovo'):
            if (sys.argv[3] == '--svc'):
                filename = "ovo_clf_svc.sav"
            elif (sys.argv[3] == '--per'):
                filename = "ovo_clf_per.sav"
            else:
                print("Escolha por último se deseja --scv ou --per")
                exit()
        elif(sys.argv[2] == '--ovr'):
            if (sys.argv[3] == '--svc'):
                filename = "ovr_clf_svc.sav"
            elif (sys.argv[3] == '--per'):
                filename = "ovr_clf_per.sav"
            else:
                print("Escolha por último se deseja --scv ou --per")
                exit()
        elif(sys.argv[2] == '--eoc'):
            if (sys.argv[3] == '--svc'):
                filename = "eoc_clf_svc.sav"
            elif (sys.argv[3] == '--per'):
                filename = "eoc_clf_per.sav"
            else:
                print("Escolha por último se deseja --scv ou --per")
                exit()
        else:
            print("Escolha em seguida se deseja --ovo ou --ovr ou --eoc")
            exit()
        test(filename)
        
    else:
        print("Escolha primeiro se deseja --train ou --test")



    
    



if __name__ == "__main__":
    main()