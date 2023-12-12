# Ukraine-Russia_War_Tweets_Analysis

# Importing all the Libraries
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv('Ukraine_Russia_War.csv',encoding='ISO-8859-1')
data.head()
data.info()
data.describe()
data.isnull().sum()

col=list(data.columns)
col

data['content']

data['lang'].value_counts()

data.lang.value_counts().sort_values().plot(kind='pie')

text=data['content'][100]
text

# Extracting Hashtags
import re
import nltk

ht=re.findall(r"#(\w+)",text)
ht

def hashtag_extract(text_corpus):
  hashtag=[]
  for text in text_corpus:
    ht=re.findall(r"#(\w+)",text)
    hashtag.append(ht)
  return hashtag

def hashtag_freq(hashtag):
  a=nltk.FreqDist(hashtag)
  d=pd.DataFrame({'Hashtag':list(a.keys()),
                 'Freq': list(a.values())}
                 )
  d=d.nlargest(columns="Freq",n=30)
  return d

hashtags=hashtag_extract(data['content'])
hashtags

hashtags=sum(hashtags,[])
hashtags

hash=hashtag_freq(hashtags)
hash

plt.figure(figsize=(26,7))
ax=sns.barplot(data=hash, x='Hashtag', y="Freq")
plt.xticks(rotation=90)
plt.show()

data['total_length']=data['content'].str.len()

data[['content','total_length']]

total_cart=data['total_length'].sum()
total_cart

avg_tweet_len=total_cart/len(data['content'])
avg_tweet_len

data['word_count']=data['content'].str.split().str.len()

data[['content','total_length','word_count']]

avg_word_count=data['word_count'].sum()/len(data['content'])
avg_word_count

# Sentimental Analysis
import string

from nltk.corpus import stopwords
nltk.download("stopwords")
stopword=stopwords.words('english')

stemmer=nltk.SnowballStemmer('english')

def clean(text):
  text=str(text).lower()
  text=re.sub('https?://\S+|www\.\S+','',text)
  text=re.sub('[%s]'%re.escape(string.punctuation),'',text)
  text=re.sub('\n|\t','',text)
  text=re.sub('\d','',text)
  text=[word for word in text.split(' ') if word not in stopword]
  text=" ".join(text)
  text=[stemmer.stem(word) for word in text.split(' ')]
  text=" ".join(text)
  return text

data['clear_tweet']=data['content'].apply(clean)

data[['clear_tweet','content']]

avg_len_after_clean=data['clear_tweet'].str.len().sum()/len(data['content'])
avg_len_after_clean

avg_word_count_after_clean=data['clear_tweet'].str.split().str.len().sum()/len(data['content'])
avg_word_count_after_clean

from textblob import TextBlob

text=data['clear_tweet'][1]
text

anlysis=TextBlob(data['clear_tweet'][14])
anlysis.sentiment.polarity

def sen_analysis(text):
  anlysis=TextBlob(text)
  if anlysis.sentiment.polarity > 0:
    return 1
  elif anlysis.sentiment.polarity ==0:
    return 0
  else:
    return -1

data['sentiment']=data['clear_tweet'].apply(lambda x:sen_analysis(x))

data[['clear_tweet','sentiment']]

data['retweetCount']

data['len_clean_text']=data['clear_tweet'].apply(len)
data['wordcount_clean_text']=data['clear_tweet'].apply(lambda x: len(str(x).split()))

dataset=data[['clear_tweet','sentiment', 'retweetCount', 'len_clean_text', 'wordcount_clean_text']]
dataset

sentiment=dataset['sentiment'].value_counts()
sentiment

sns.countplot(data=dataset, x='sentiment')

negative=dataset[dataset['sentiment']==-1]
neutral=dataset[dataset['sentiment']==0]
positive=dataset[dataset['sentiment']==1]

from wordcloud import WordCloud

text=' '.join(text for text in dataset['clear_tweet'])

len(text)

wordcloud=WordCloud(max_font_size=100,
                    max_words=50).generate(text)

plt.imshow(wordcloud, interpolation= 'bilinear')
plt.show()

positive

text=' '.join(text for text in positive['clear_tweet'])

len(text)

wordcloud=WordCloud(max_font_size=100,
                    max_words=50).generate(text)

plt.imshow(wordcloud, interpolation= 'bilinear')
plt.show()
