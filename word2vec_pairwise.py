import fetch_model_name
import gensim.models
import plotly.graph_objects as go
import networkx as nx
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np


def reduce_dimensions(wv, word_list):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    #vectors = np.asarray(model.wv.vectors)
    #labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    vec = []
    lab = []
    for word in word_list:
        asd = wv.get_vector(word)
        vec.append(asd)
        lab.append(word)

    vectors = np.asarray(vec)
    labels = np.asarray(lab)  # fixed-width numpy strings
    print(vectors)
    print(labels)
    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0, perplexity=5)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels



model_name = fetch_model_name.fetch_model_name()

print(model_name)

new_model = gensim.models.Word2Vec.load(f"gensim-model-{model_name}")

#wv = api.load('word2vec-google-news-300')


wv = new_model.wv

word_list = ["fever", "chills", "sweats", "aches", "pains", "fatigue", "coughing", "breathing", "nausea", "vomiting", "diarrhoea"]
#word_list = ["sick", "sleepy", "uncomfortable", "dizzy", "nauseous", "unwell", "bedridden", "coughing", "fever", "hospitalized", "headache", "rashes"]


x_vals, y_vals, labels = reduce_dimensions(wv, word_list)

print(labels)

word_distances = {}

for i in range(0, len(word_list)):
    for j in range(i+1, len(word_list)):
        word = f"{word_list[i]} -> {word_list[j]}"
        diff = wv.similarity(word_list[i], word_list[j])
        word_distances[word] = diff

print(word_distances)

asd = list(word_distances.values())
# TODO:
# Add more words
# Spot the exact time and location for outbreak start
# Heatmap of features, similarity of similarity; create a vector of similarity for all symptoms, then compare the similarity of those vectors
# Self organized map

fig = go.Figure()

G = nx.Graph()

for i in range(0, len(labels)):
    G.add_node(labels[i], pos=(x_vals[i], y_vals[i]))

print(G.nodes().data())

node_x = []
node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)

#trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
fig.add_trace(
    go.Scatter(x=node_x, y=node_y, mode="markers+text", text=labels, textposition="top center")
)

fig.show()


'''
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')
'''

"""
fig.add_trace(
    go.Scatter(x=asd, y=asd, mode="markers")
)


fig.show()
"""
#print(word_distances)