#topic modeling with LDA

import pandas as pd

data = pd.read_csv('src/data.csv')
print(data.head())



from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

data = data['content']
data.head()

en_stopwords = stopwords.words('english')
# instantiate the Lemmatizer to perform stemming/lemmatization
lmr = WordNetLemmatizer()

# tokenize the text
article_doc = []
full_text = ""
for t in word_tokenize(full_text):
    if t.isalpha():
        t = lmr.lemmatize(t.lower())
        if t not in en_stopwords:
            article_doc.append(t)

print(article_doc[:10])

def modelTopic(article):
  topic = str()

  return topic
#data['topic'] = data.apply(lambda row: modelTopic(row['content']),axis=1)