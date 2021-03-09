
# =============================================================================
# Import of Packages
# =============================================================================

import arxiv 
from pprint import pprint 


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

# result = arxiv.query(
#   query="quantum",
#   max_chunk_results=1,
#   max_results=1,
#   iterative=True
# )

result = arxiv.query(id_list=["2009.11432"])

for paper in result:
   pprint(paper)
