import sys 
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import os
import glob
import pandas as pd
import random
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OutputCodeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix
from scipy import stats

# Seleciona a classes da base (5,10,40,120)
import data_conf_5 as dc

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

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    
    images = np.array(images)
    return images

def extract_features(image): 
    # Transforma imagem em grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # Instancia um descritor orb
    orb = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE,edgeThreshold=10,nfeatures=10)
    kp = orb.detect(gray, None)
    kp, des = orb.compute(gray, kp) 
    des = np.array(des)

    return des

def training(clf, filename):
    print("Extraindo features das imagens ...")

    # Varre as pastas selecionadas no arquivo data_conf.py
    for i in range(len(dc.folder_path_train)):
        # Carrega imagens da pasta
        images = load_images_from_folder(dc.folder_path_train[i])

        # Para cada imagem, retira os descritores orb da imagem e coloca na lista de descritores
        for j in range(len(images)):
            des = extract_features(images[j])
            if(des.shape!=()):
                if(j == 0):
                    des_list = des
                else:
                    des_list = np.vstack((des_list, des))

        # Cria a matriz de atributos e o vetor de labels
        if(i == 0):
            atributes = des_list
            labels = np.full((1,des_list.shape[0]),dc.class_names[i])
        else:
            atributes = np.vstack((atributes, des_list))
            labels = np.append(labels, np.full((1,des_list.shape[0]),dc.class_names[i]))

    # Treina classificador
    print("Inicio do treinamento ...")
    t0 = time.time()
    clf.fit(atributes, labels)
    t1 = time.time()
    print("O treinamento demorou", t1-t0, " segundos")
    pickle.dump(clf, open((filename+".sav"), 'wb'))

def test(filename):

    # Carrega o classificador
    clf = pickle.load(open((filename+".sav"), 'rb'))

    t0 = time.time()
    for i in range(len(dc.folder_path_test)):
        images = load_images_from_folder(dc.folder_path_test[i])

        for j in range(len(images)):
            # Extrai as features da imagem
            des = extract_features(images[j])

            # Se existirem features, faz a predição do classificador das features, se não excolhe uma classe aleatoriamente
            if(des.shape != ()):
                prediction = clf.predict(des) # Vetor de classificação das features
                # A classe da imagem será a moda das classificações das features
                prediction = stats.mode(prediction).mode[0]
            else:
                prediction = np.random.choice(dc.class_names)

            if(i==0 and j==0):
                y_pred = prediction
                y_true = dc.class_names[i]
            else:
                y_pred = np.append(y_pred,prediction)
                y_true = np.append(y_true,dc.class_names[i])

    t1 = time.time()
    print("A fase de predição e teste durou", t1-t0, " segundos")
    plot_confusion_matrix(y_true, y_pred, classes=dc.class_names,title='Confusion matrix ')

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