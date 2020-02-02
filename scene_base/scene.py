from scipy.io import arff
import pandas as pd
import numpy as np
import sys
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import Perceptron 
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

def plot_confusion_matrix(y_true, y_pred, classes, title):
    # Calcula acurácia
    acc = accuracy_score(y_true, y_pred)

    plt.figure(figsize=(10,10))

    for i in range(6):
      # Constroi matriz de confusão
      cm = multilabel_confusion_matrix(y_true, y_pred)[i]
      cm = normalize(cm, axis=1, norm='l1')
      cm_df = pd.DataFrame(cm, index = classes, columns = classes)
      plt.subplot(3,3,i+1)     
      sns.heatmap(cm_df, annot=True, cmap="YlGnBu")
      plt.title(title[i])
      plt.ylabel('True label')
      plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


sys.path.append('.')
data = arff.loadarff('scene-train.arff')

df_train = pd.DataFrame(data[0])

converte = {
    'Beach':{b'0': 0, b'1': 1},
    'Sunset':{b'0': 0, b'1': 1},
    'FallFoliage':{b'0': 0, b'1': 1},
    'Field':{b'0': 0, b'1': 1},
    'Mountain':{b'0': 0, b'1': 1},
    'Urban':{b'0': 0, b'1': 1}
}
df_train.replace(converte, inplace=True)

x_train = df_train.drop(categories, axis=1)
y_train = df_train[categories].values

mltout_clf_svm = MultiOutputClassifier(svm.SVC(), n_jobs=-1).fit(x_train,y_train)
mltout_clf_per = MultiOutputClassifier(Perceptron(), n_jobs=-1).fit(x_train,y_train)

data_test = arff.loadarff('scene-test.arff')
df_test = pd.DataFrame(data_test[0])
df_test.replace(converte, inplace=True)
x_test = df_test.drop(categories, axis=1)
y_test = df_test[categories]

# Gera o array com as predições para cada exemplo
y_pred_svm = mltout_clf_svm.predict(x_test)

# Acurácia do método
print("Acurácia:",mltout_clf_svm.score(x_test,y_test))

# Matriz de confusão para cada rótulo
plot_confusion_matrix(y_test, y_pred_svm, classes=[0,1], title=categories)


# Gera o array com as predições para cada exemplo
y_pred_per = mltout_clf_per.predict(x_test)

# Acurácia do método
print("Acurácia: ",mltout_clf_per.score(x_test,y_test))

# Matriz de confusão para cada rótulo
plot_confusion_matrix(y_test, y_pred_svm, classes=[0,1], title=categories)

