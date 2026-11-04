# pip install networkx


import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add web page links
G.add_edges_from([
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
    ("C", "A"),
    ("D", "C"),
    ("D", "A"),
    ("E", "C")
])

# Calculate PageRank
pagerank = nx.pagerank(
    G,
    alpha=0.85,
    max_iter=100
)

# Display PageRank scores
print("PageRank Scores:")


for page, score in pagerank.items():
    print(page, ":", round(score, 4))

# Sort pages by ranking
print("\nPages Ranked by Importance:")


ranked_pages = sorted(
    pagerank.items(),
    key=lambda x: x[1],
    reverse=True
)

for page, score in ranked_pages:
    print("Page", page, ":", round(score, 4))



# 5. PageRank Algorithm
# PageRank is an algorithm developed by Google 
# founders Larry Page and Sergey Brin to rank web pages 
# according to their importance. It works on the principle 
# that a web page receives importance from other pages linking to it. 
# In this practical, web pages are represented as nodes and hyperlinks 
# as directed edges. Python's NetworkX library is used to create the graph, 
# calculate PageRank scores and rank pages according to their importance.