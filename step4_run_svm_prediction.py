#!/usr/bin/env python
# coding: utf-8

# In[27]:


import joblib

svm_model = joblib.load('model.joblib')


# In[ ]:


import pandas as pd

threads_S_result = pd.read_csv('sentiment_check_results.csv')
print(threads_S_result['review_description'])

x_train, x_test, y_train, y_test = train_test_split(threads_S_result['review_description'], threads_S_result['sentiment'], test_size=0.2 )
tfidf_vectorizer = TfidfVectorizer(max_features=4000)
x_train_tfidf = tfidf_vectorizer.fit_transform(x_train)
x_test_tfidf = tfidf_vectorizer.transform(x_test)


# In[44]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def tokenize(str):
    tfidf_vectorizer = TfidfVectorizer(max_features=4000)
    v = tfidf_vectorizer.fit_transform(str)
    print(v)
    return v


# In[50]:


tokens = tokenize('hi this is new repository')


# In[51]:


def predict(tokens):
    result = svm_model.predict(tokens)
    print(result)
    return result

svm_predictions = predict(tokens)

