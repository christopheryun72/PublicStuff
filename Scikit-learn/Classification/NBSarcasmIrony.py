#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pandas as pd
import spacy
import string
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
nlp = spacy.load('en')
stopwords = STOP_WORDS
parser = English()


# In[2]:


df1 = pd.read_csv('SarcasmIronyTrain.csv')
df2 = pd.read_csv('SarcasmIronyTest.csv')
df = pd.concat([df1, df2], axis = 0, join='outer')
print(df.describe())
print(df.info())
df.shape


# In[3]:


print(df['tweets'].isna().sum())
df.dropna(inplace=True)
print(df['tweets'].isna().sum())
df.head()


# In[4]:


def tokenizer(sentence):
    tokens = parser(sentence)
    tokensEdit = [token.lower_ if token.lemma_ == '-PRON-' else token.lemma_.lower().strip() for token in tokens]
    noPunct = [token for token in tokensEdit if token not in string.punctuation]
    noStop = [token for token in noPunct if token not in stopwords]
    return noStop


# In[5]:


class customCleaner(TransformerMixin):
    def transform(self, X, **transform_params):
        cleaned = [sentence.strip().lower() for sentence in X if type(sentence) != float]
        return cleaned
    def fit(self, X, y=None, **fit_params):
        return self
    def get_params(self, deep=True):
        return {}
    
count_vector = CountVectorizer(tokenizer=tokenizer, ngram_range=(1,1))
tfidf_vector = TfidfVectorizer(tokenizer=tokenizer)


# In[6]:


X = df.iloc[:, 0]
y = df.iloc[:, 1]
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.15)


# In[7]:


mnb = MultinomialNB()
pipmodel = Pipeline([('cleaner', customCleaner()), ('counter', count_vector), ('model', mnb)])
pipmodel.fit(xtrain, ytrain) 


# In[8]:


ypred = pipmodel.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)
cr = classification_report(ytest, ypred)
print(cr)
accuracy = accuracy_score(ytest, ypred)
print(accuracy)
cv = RepeatedKFold(n_splits=5, n_repeats=2)
cv_score = cross_val_score(pipmodel, X, y, cv=cv)
print(cv_score.mean())


# In[ ]:




