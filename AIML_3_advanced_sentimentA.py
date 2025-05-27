#!/usr/bin/env python
# coding: utf-8

# Sentiment Analysis using NLP. The goal is to analyze sentiments (positive, negative) from text data using Natural Language Processing techniques in Python.This project is based on the tutorial by robikscube on Kaggle.

# In[ ]:


#import neccessary liberaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import nltk


# In[2]:


#loading dataset Reviews.csv from Amazon
df = pd.read_csv('C:\\Users\\muthu\\OneDrive\\Documents\\AIML INTERNSHIP\\3_advancedllmproject\\Reviews.csv')
print(df.shape)
df = df.head(500)
print(df.shape)


# In[3]:


df.head()


# In[4]:


#visualizing and preprocessing
ax = df['Score'].value_counts().sort_index() \
    .plot(kind='bar',
          title='Count of Reviews by Stars',
          figsize=(10, 5))
ax.set_xlabel('Review Stars')
plt.show()


# In[5]:


example = df['Text'][50]
print(example)


# In[6]:


#tokenizing
tokens = nltk.word_tokenize(example)
tokens[:10]


# In[7]:


#tagging 
tagged = nltk.pos_tag(tokens)
tagged[:10]


# In[8]:


# Text Vectorization
entities = nltk.chunk.ne_chunk(tagged)
entities.pprint()


# In[9]:


#model building
from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()


# In[10]:


sia.polarity_scores('I am so happy!')


# In[11]:


sia.polarity_scores('This is the worst thing ever.')


# In[12]:


sia.polarity_scores(example)


# In[13]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['Text']
    myid = row['Id']
    res[myid] = sia.polarity_scores(text)


# In[14]:


# Vader sentiment analyzer
vaders = pd.DataFrame(res).T
vaders = vaders.reset_index().rename(columns={'index': 'Id'})
vaders = vaders.merge(df, how='left')


# In[15]:


vaders.head()


# In[16]:


#exploring data
ax = sns.barplot(data=vaders, x='Score', y='compound')
ax.set_title('Compund Score by Amazon Star Review')
plt.show()


# In[17]:


#Visualizing for positive, negative and neutral
fig, axs = plt.subplots(1, 3, figsize=(12, 3))
sns.barplot(data=vaders, x='Score', y='pos', ax=axs[0])
sns.barplot(data=vaders, x='Score', y='neu', ax=axs[1])
sns.barplot(data=vaders, x='Score', y='neg', ax=axs[2])
axs[0].set_title('Positive')
axs[1].set_title('Neutral')
axs[2].set_title('Negative')
plt.tight_layout()
plt.show()


# In[18]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax


# In[19]:


# RoBERTa - sentiment scoring (transformer-based model from Hugging Face)
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[20]:


print(example)
sia.polarity_scores(example)


# In[21]:


encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}
print(scores_dict)


# In[22]:


def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg' : scores[0],
        'roberta_neu' : scores[1],
        'roberta_pos' : scores[2]
    }
    return scores_dict


# In[23]:


res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {}
        for key, value in vader_result.items():
            vader_result_rename[f"vader_{key}"] = value
        roberta_result = polarity_scores_roberta(text)
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')


# In[24]:


results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})
results_df = results_df.merge(df, how='left')


# In[25]:


results_df.columns


# In[26]:


# Model comparison using pair plot
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                  'roberta_neg', 'roberta_neu', 'roberta_pos'],
            hue='Score',
            palette='tab10')
plt.show()


# In[27]:


results_df.query('Score == 1') \
    .sort_values('roberta_pos', ascending=False)['Text'].values[0]


# In[28]:


results_df.query('Score == 1') \
    .sort_values('vader_pos', ascending=False)['Text'].values[0]


# In[29]:


results_df.query('Score == 5') \
    .sort_values('roberta_neg', ascending=False)['Text'].values[0]


# In[30]:


results_df.query('Score == 5') \
    .sort_values('vader_neg', ascending=False)['Text'].values[0]


# In[34]:


# DistilBERT pipeline - custom inputs prediction
from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english",  # Smaller model
                         device=-1)


# In[45]:


#evaluating with different inputs
sent_pipeline('every Us server is LOCKED')


# In[55]:


sent_pipeline('okay nothing special charge diamond member hilto')


# In[44]:


sent_pipeline('its frustating when the support team keeps saying, i have manually updated ur campaign, it will reflect in couple of hours with no avail. just say it will tske 24 - 48  hoours, so no one can plan accordingly, pathetic customer support')


# In[ ]:





# In[ ]:




