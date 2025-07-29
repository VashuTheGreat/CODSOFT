#!/usr/bin/env python
# coding: utf-8

# In[50]:


import nltk
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer   
from sklearn.preprocessing import LabelEncoder


# In[4]:


df=pd.read_json('data.json')


# In[16]:


tags=[x['tag'] for x in df['intents']]


# In[17]:


tags


# In[30]:


df1=pd.read_csv('data1.csv', on_bad_lines='skip')


# In[31]:


df1


# In[32]:


text=df1['text'].copy()


# In[33]:


text


# In[34]:


def cleaner(text):
    ps = PorterStemmer()
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [ps.stem(word) for word in text if word.isalpha()]
    return ' '.join(text)


# In[38]:


text=text.apply(cleaner)


# In[39]:


text


# In[42]:


cv=CountVectorizer(ngram_range=(1,2))


# In[48]:


vec = cv.fit_transform(text)


# In[46]:


cv.vocabulary_


# In[49]:


vec.toarray()


# In[51]:


le=LabelEncoder()


# In[53]:


label=df1['label'].copy()

y=le.fit_transform(label)


# In[54]:


y


# In[55]:


from sklearn.ensemble import RandomForestClassifier


# In[56]:


modelR=RandomForestClassifier()


# In[58]:


modelR.fit(vec.toarray(), y)


# In[60]:


modelR.predict(vec.toarray()[0].reshape(1, -1))


# In[61]:


from sklearn.pipeline import make_pipeline


# In[62]:


pipelines = make_pipeline(
    cleaner, 
    cv,    
    modelR 
)


# In[63]:


import pickle

with open('model.pkl', 'wb') as file:
    pickle.dump(pipelines, file)


# In[64]:


le.classes_


# In[65]:


le.classes_[y[0]]


# In[66]:


with open('indexes.pkl', 'wb') as file:
    pickle.dump(le.classes_, file)


# In[ ]:




