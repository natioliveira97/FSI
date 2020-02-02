import re
from time import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.linear_model import Perceptron
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import matplotlib.gridspec as gridspec

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

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

def classification(clf,name):
  print('_' * 80)
  print("Training: ", name)
  t0 = time()

  # Realiza o fit do classificador
  trainning = MultiOutputClassifier(clf)
  trainning.fit(data_train, label_train)
                        
  train_time = time() - t0                  
  print("train time: %0.3fs" % train_time)

  t0 = time()

  #Realiza a predição
  pred = trainning.predict(data_test)

  test_time = time() - t0

  # Imprime acurácia da classificação
  print("test time:  %0.3fs" % test_time)
  print('Test accuracy is {}'.format(accuracy_score(label_test, pred)))
  
  # Matriz de confusão para cada rótulo
  
  plot_confusion_matrix(label_test, pred, classes=[0,1], title=categories)


data = pd.read_csv("train.csv")
df = pd.DataFrame(data)

df['comment_text'] = df['comment_text'].map(lambda com : clean_text(com))

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
comment = ['comment_text']

dataset_serie, _ = train_test_split(df, random_state=42, test_size=0.0001, shuffle=True)

dataset_serie_comment = dataset_serie.comment_text
dataset_serie_categories = dataset_serie[categories]

vectorizer = TfidfVectorizer(stop_words=stop_words)
comment_vec = vectorizer.fit_transform(dataset_serie_comment)

tam_data,j = comment_vec.shape
limite_train = round(tam_data*0.66)

data_train = comment_vec[0:limite_train,]
data_test = comment_vec[limite_train:,]
label_train = dataset_serie_categories.iloc[0:limite_train,]
label_test = dataset_serie_categories.iloc[limite_train:,]

for clf, name in ((LinearSVC(),'LinearSVC'),(LinearSVC(class_weight='balanced',max_iter=3000),'LinearSVC Balanced (com ajuste de pesos das classes)'), (Perceptron(),'Perceptron')):
  classification(clf,name)
