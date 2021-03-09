
# =============================================================================
# Import of Packages
# =============================================================================

import pandas
from pprint import pprint 

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

lem = WordNetLemmatizer()
stem = PorterStemmer()


# =============================================================================
# Import of Data
# =============================================================================

dataset = pandas.read_csv('papers2.txt', delimiter = '\t')
# print(dataset.head())


# =============================================================================
# Text Exploration
# =============================================================================

dataset['word_count'] = dataset['abstract'].apply(lambda x: len(str(x).split(" ")))
# pprint(dataset[['abstract','word_count']].head())

## Descriptive statistics of word counts
# pprint(dataset.word_count.describe())

## Identify common words
max_freq = pandas.Series(' '.join(dataset['abstract']).split()).value_counts()[:20]
# pprint(max_freq)

## Identify uncommon words
min_freq = pandas.Series(' '.join(dataset['abstract']).split()).value_counts()[-20:]
# pprint(min_freq)


# =============================================================================
# Text Pre-Processing
# =============================================================================

## Creating a list of stop words and adding custom stopwords
stop_words = set(stopwords.words("english"))
new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]
stop_words = stop_words.union(new_words)
# pprint(stop_words)


corpus = []

for i in range(len(dataset)):
    
    # Remove punctuations
    text = re.sub('[^a-zA-Z]', ' ', dataset['abstract'][i])
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove tags
    text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # Remove special characters and digits
    text = re.sub("(\\d|\\W)+"," ",text)
    
    # Convert to list from string
    text = text.split()
    
    # Stemming
    ps = PorterStemmer()
    
    # Lemmatisation
    text = [lem.lemmatize(word) for word in text if not word in  
            stop_words] 
    
    text = " ".join(text)    
    corpus.append(text)

# pprint(corpus[22])


# =============================================================================
# Visualization as WordCloud
# =============================================================================

wordcloud = WordCloud(
    background_color='white',
    stopwords=stop_words,
    max_words=100,
    max_font_size=50, 
    random_state=42
).generate(str(corpus))


#"""
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
# fig.savefig("word1.png", dpi=900)
#"""
