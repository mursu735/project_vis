import helpers
import random
import gensim.models
import gensim.downloader as api
import plotly.graph_objects as go
import networkx as nx
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import umap
import numpy as np


def reduce_dimensions(wv):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(wv.vectors)
    labels = np.asarray(wv.index_to_key)  # fixed-width numpy strings

    print(vectors)
    print(labels)
    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0, perplexity=5)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels



model_name = helpers.fetch_model_name()

print(model_name)



#wv = api.load('word2vec-google-news-300')

new_model = gensim.models.Word2Vec.load(f"{model_name}")
wv = new_model.wv

#wv = api.load("glove-twitter-300")  # load glove vectors

#word_list_filled.append(similar_symptoms)

x_vals, y_vals, labels = reduce_dimensions(wv)

fig = go.Figure()

G = nx.Graph()

for i in range(0, len(labels)):
    G.add_node(labels[i], pos=(x_vals[i], y_vals[i]))

node_x = []
node_y = []
circles = {}
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)
    circles[node] = [dict(
        type="circle",
        xref="x", yref="y",
        x0=x - 5, y0=y - 5,
        x1=x + 5, y1=y + 5,
        line=dict(color="DarkOrange"))]

#trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
fig.add_trace(
    go.Scatter(x=node_x, y=node_y, mode="markers", text=labels, name="words")
)

def highlight_group(group):
    result = []

    for tracer_ix, tracer in enumerate(fig["data"]):
        colors = ["red" if datapoint_group == group else "black" for datapoint_group in fig["data"][tracer_ix]["text"]]
        result.append(colors)

    return result

def highlight_circle(group):
    result = []

    for tracer_ix, tracer in enumerate(fig["data"]):
        print(fig["data"][tracer_ix]["text"])
        #print(group)
        colors = ["red" if datapoint_group == group else "black" for datapoint_group in fig["data"][tracer_ix]["text"]]
        result.append(colors)

    return result


fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "label": group,
                    "method": "update",
                    "args": [
                        {"marker.color": highlight_group(group)},
                        {"shapes": circles[group] }
                        
                    ],
                }
                for group in labels
            ]
        }
    ],
    margin={"l": 0, "r": 0, "t": 25, "b": 0},
    height=700
)

fig.show()

#fig.write_html("server/bar_chart_simple.html", auto_play=False, include_plotlyjs="cdn")


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