# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 23:51:11 2019

@author: atidem
"""
## write on spyder ide

from textblob import TextBlob
from sklearn import model_selection,preprocessing,linear_model,naive_bayes,metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn import decomposition,ensemble
import xgboost, numpy, textblob, string
from keras.preprocessing import text, sequence
from keras import layers,models,optimizers
import pandas as pd

data = pd.read_csv("train.tsv",sep="\t")
data.head()
data.info()

#%%
## Preprocessing
data["Sentiment"].replace(0,value="negatif",inplace=True)
data["Sentiment"].replace(1,value="negatif",inplace=True)
data["Sentiment"].replace(3,value="pozitif",inplace=True)
data["Sentiment"].replace(4,value="pozitif",inplace=True)

data = data[(data.Sentiment=="negatif")|(data.Sentiment=="pozitif")]
data.head()

data.groupby("Sentiment").count()

df = pd.DataFrame()
df["text"] = data["Phrase"]
df["label"] = data["Sentiment"]
df.head()

df["text"] = df["text"].apply(lambda x:" ".join(x.lower() for x in x.split()))
df["text"] = df["text"].str.replace("[^\w\s]","")
df["text"] = df["text"].str.replace("\d","")

import nltk
from nltk.corpus import stopwords

##clear stop words
sw = stopwords.words("english")
df["text"] = df["text"].apply(lambda x:" ".join(x for x in x.split() if x not in sw))

sil = pd.Series(" ".join(df["text"]).split()).value_counts()[-1000:]
df["text"] = df["text"].apply(lambda x:" ".join(x for x in x.split() if x not in sil))

from textblob import Word 
df["text"] = df["text"].apply(lambda x:" ".join([Word(word).lemmatize() for word in x.split()]))

## Train-Test split
train_x,test_x,train_y,test_y = model_selection.train_test_split(df["text"],df["label"])
encoder = preprocessing.LabelEncoder()
train_y =encoder.fit_transform(train_y)
test_y =encoder.fit_transform(test_y)


## Tf_idf create vector
tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)

x_train_tf_idf = tf_idf_word_vectorizer.transform(train_x)
x_test_tf_idf = tf_idf_word_vectorizer.transform(test_x)
tf_idf_word_vectorizer.get_feature_names()[0:5]

x_train_tf_idf.toarray()


## N-grams n=2
tf_idf_ngram_vectorizer = TfidfVectorizer(ngram_range=(2,2))
tf_idf_ngram_vectorizer.fit(train_x)

x_train_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(train_x)
x_test_tf_idf_ngram = tf_idf_ngram_vectorizer.transform(test_x)
tf_idf_ngram_vectorizer.get_feature_names()[0:20]

#%%
## ann

mlpc_params = {"alpha":[0.1,0.01,0.001,0.0001,0.02,0.05],
               "hidden_layer_sizes":[(10,10,2),(100,100,2),(100,2),(20,2)],
               "solver":["lbfgs","adam","sgd"],
               "activation":["relu","logistic"]}
from sklearn.neural_network import MLPClassifier
ann = MLPClassifier()
mlpc_cv = GridSearchCV(ann,mlpc_params,cv=5,n_jobs=-1,verbose=2)
mlpc_cv.fit(x_train_tf_idf,train_y)
mlpc_cv.best_params_


mlpc_tuned = MLPClassifier(activation='relu',alpha=0.1,hidden_layer_sizes=(100,100),solver='sgd')
mlpc_tuned.fit(x_train_tf_idf,train_y)
y_pred = mlpc_tuned.predict(x_test_tf_idf_ngram)
accuracy_score(test_y,y_pred)


#%%
## svm 
from sklearn.svm import SVC

svm_model = SVC(kernel="linear").fit(x_train_tf_idf,train_y)
svm_model
y_pred = svm_model.predict(x_test_tf_idf)
accuracy_score(test_y,y_pred)

## test sentence
metin =pd.Series("it is not succesfull it is awful ")
tf_idf_word_vectorizer = TfidfVectorizer()
tf_idf_word_vectorizer.fit(train_x)
metin = tf_idf_word_vectorizer.transform(metin)



svm_model.predict(metin)
















