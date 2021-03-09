
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
from sklearn.feature_extraction.text import CountVectorizer

from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
import seaborn as sns

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
new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown", "et", "al", "cond", "mat", "per"]
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

"""
wordcloud = WordCloud(
    background_color='white',
    stopwords=stop_words,
    max_words=100,
    max_font_size=50, 
    random_state=42
).generate(str(corpus))

fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
# fig.savefig("word1.png", dpi=900)
#"""


# =============================================================================
# Text Preparation
# =============================================================================

# Tokenisation & Vectorisation 

# cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
# X=cv.fit_transform(corpus)
# pprint(list(cv.vocabulary_.keys())[:10])

# Most frequently occuring words
def get_top_n_words(corpus, n=None):
    # vec = CountVectorizer().fit(corpus)
    vec = CountVectorizer(ngram_range=(3,3), max_features=2000).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in      
                   vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], 
                       reverse=True)
    return words_freq[:n]

#Convert most freq words to dataframe for plotting bar plot
top_words = get_top_n_words(corpus, n=20)
top_df = pandas.DataFrame(top_words)
top_df.columns=["Word", "Freq"]

# Barplot of most freq words

sns.set(rc={'figure.figsize':(16,10)})
g = sns.barplot(x="Word", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=30)

plt.title("three-grams")
plt.tight_layout()
#plt.show()

fig = g.get_figure()
fig.savefig('three-grams.png') 
