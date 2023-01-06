# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 16:03:32 2022

@author: Emrullah erdeve
"""
#İMPORT LİBRARY
import pandas as pd
import numpy as np
import neattext.functions as nfx
#%%  import data
df = pd.read_csv("train.csv")
df.drop(["id"],axis=1,inplace = True)

#CLEAN DATA
df['comment_text'].apply(nfx.extract_hashtags)
df['extracted_hashtags'] = df['comment_text'].apply(nfx.extract_hashtags)
df['clean_tweet'] = df['comment_text'].apply(nfx.remove_hashtags)
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: nfx.remove_userhandles(x))
df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_urls)
df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_multiple_spaces)
df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_puncts)
df[['comment_text','clean_tweet']]

# %% görev 1 ve 2
from textblob import TextBlob

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    if sentiment_polarity > 0.5:
        sentiment_label = 'TOXİC'
    else :
        sentiment_label = 'Not TOXİC'

    result = {'polarity':sentiment_polarity,
              'subjectivity':sentiment_subjectivity,
              'sentiment':sentiment_label}
    return result
def get_sentiment2(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity
    sentiment_subjectivity = blob.sentiment.subjectivity
    if sentiment_polarity < 0.333:
        sentiment_label = 'not TOXİC'
    elif sentiment_polarity < 0.66:
        sentiment_label = 'less TOXİC'
    else:
        sentiment_label = 'very toxic'
    result = {'polarity':sentiment_polarity,
              'subjectivity':sentiment_subjectivity,
              'sentiment':sentiment_label}
    return result
"""
Sınıflandırmanın neredeyse doğru olduğunu varsayıyorum çünkü datanın target ortalaması 0.1 iken
aynı oranda toxic sayısı bulunmaktadır.
"""

# %% görselleştirme ve görev 3 e hazırlık ve tokenaziton
df1=df
df['toxic_results'] = df['clean_tweet'].apply(get_sentiment)
df = df.join(pd.json_normalize(df['toxic_results']))
df1['toxic_results'] = df['clean_tweet'].apply(get_sentiment2)
df1 = df1.join(pd.json_normalize(df1['toxic_results']))
df2=df1
df['sentiment'].value_counts()
df1['sentiment'].value_counts()
df['sentiment'].value_counts().plot(kind='bar')
df1['sentiment'].value_counts().plot(kind='bar')
df2.drop(["toxic_results"],axis=1,inplace = True)
# %% Tokenization
positive_tweet = df1[df1['sentiment'] == 'not TOXİC']['clean_tweet']
negative_tweet = df1[df1['sentiment'] == 'very toxic']['clean_tweet']
lesstoxic_tweet = df1[df1['sentiment'] == 'less TOXİC']['clean_tweet']

positive_tweet_list = positive_tweet.apply(nfx.remove_stopwords).tolist()
negative_tweet_list = negative_tweet.apply(nfx.remove_stopwords).tolist()
lesstoxic_tweet_list = lesstoxic_tweet.apply(nfx.remove_stopwords).tolist()

pos_tokens = [token for line in positive_tweet_list  for token in line.split()]
neg_tokens = [token for line in negative_tweet_list  for token in line.split()]
lesstoxic_tokens = [token for line in lesstoxic_tweet_list  for token in line.split()]


df3=df2

