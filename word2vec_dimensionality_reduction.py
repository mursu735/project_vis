import fetch_model_name
import random
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

word_list_filled = word_list.copy()

number_of_fillers = 0
max = len(wv.index_to_key) - 1
token_list = wv.index_to_key
indices_used = []

while number_of_fillers < 1000:
    index = random.randint(0, max)
    word = token_list[index]
    if word not in word_list and index not in indices_used:
        word_list_filled.append(word)
        indices_used.append(index)
        number_of_fillers += 1

x_vals, y_vals, labels = reduce_dimensions(wv, word_list_filled)

print(labels)

fig = go.Figure()

G = nx.Graph()

for i in range(0, len(labels)):
    G.add_node(labels[i], pos=(x_vals[i], y_vals[i]))

node_x = []
node_y = []
symptoms_node_x = []
symptoms_node_y = []
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)
    if node in word_list:
        symptoms_node_x.append(x)
        symptoms_node_y.append(y)

#trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
fig.add_trace(
    go.Scatter(x=node_x, y=node_y, mode="markers", text=labels, name="Filler words")
)

fig.add_trace(
    go.Scatter(x=symptoms_node_x, y=symptoms_node_y, mode="markers+text", text=labels, name="Symptoms", textposition="top center", marker=dict(color='Red'))
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