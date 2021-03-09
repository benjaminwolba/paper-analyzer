
# =============================================================================
# Import of Packages
# =============================================================================

import pandas
from pprint import pprint 


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
pprint(dataset.word_count.describe())
