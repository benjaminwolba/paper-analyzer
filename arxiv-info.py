
# =============================================================================
# Import of Packages
# =============================================================================

import arxiv 
from pprint import pprint 
import pandas as pd


# =============================================================================
# Get Info from arXiv
# =============================================================================

# result = arxiv.query(
#   query="quantum",
#   id_list=[],
#   max_results=None,
#   start = 0,
#   sort_by="relevance",
#   sort_order="descending",
#   prune=True,
#   iterative=False,
#   max_chunk_results=100
# )

result = arxiv.query(
  query="au:sachdev_subir  AND cat:cond-mat.str-el",
  max_chunk_results=10,
  max_results=100,
  iterative=True
)

# result = arxiv.query(id_list=["2009.11432"])

datalist = []

for paper in result():
   # pprint(paper['title'])
   datalist.append([paper['id'],paper['updated'],paper['summary']])

df = pd.DataFrame(datalist, columns =['id', 'year', 'abstract'])

df.to_csv(r'papers2.txt', index=None, sep='\t', mode='a') 
