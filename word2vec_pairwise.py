import word2vec_helpers
import gensim.models
import gensim.downloader as api
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import gensim.downloader as api


model_name = word2vec_helpers.fetch_model_name()



#wv = api.load('word2vec-google-news-300')
#model = api.load("glove-twitter-25")  # load glove vectors
'''
data = np.genfromtxt("http://files.figshare.com/2133304/ExpRawData_E_TABM_84_A_AFFY_44.tab",
                     names=True,usecols=tuple(range(1,30)),dtype=float, delimiter="\t")
data_array = data.view((np.float, len(data.dtype.names)))
data_array = data_array.transpose()
labels = data.dtype.names
print(labels)
print(data_array)
'''

new_model = gensim.models.Word2Vec.load(f"gensim-model-{model_name}")
wv = new_model.wv
#wv = api.load("glove-twitter-100")  # load glove vectors

word_list = word2vec_helpers.get_word_list()

word_distances = np.zeros((len(word_list), len(word_list)))

for i in range(0, len(word_list)):
    for j in range(0, len(word_list)):
        diff = wv.similarity(word_list[i], word_list[j])
        word_distances[i, j] = diff

#print(word_distances)

np.savetxt("test_dist.txt", word_distances)

fig = px.imshow(word_distances,
                text_auto=True,
                labels=dict(x="Symptom", y="Symptom", color="Similarity"),
                x=word_list,
                y=word_list)

similarity_of_similarities = np.zeros((len(word_list), len(word_list)))

for i in range(0, len(word_list)):
    current = word_distances[:, i]
    for j in range(0, len(word_list)):
        comparison = word_distances[:, j]
        cosine = np.dot(current, comparison) / (np.linalg.norm(current) * np.linalg.norm(comparison))
        similarity_of_similarities[i, j] = cosine

fig.show()

# Hierarchical clustering
'''
model = AgglomerativeClustering(affinity='cosine', linkage="single", distance_threshold=0, n_clusters=None).fit(similarity_of_similarities)

#print(model.labels_)
counts = np.zeros(model.children_.shape[0])
n_samples = len(model.labels_)
for i, merge in enumerate(model.children_):
    current_count = 0
    for child_idx in merge:
        if child_idx < n_samples:
            current_count += 1  # leaf node
        else:
            current_count += counts[child_idx - n_samples]
    counts[i] = current_count

linkage_matrix = np.column_stack(
    [model.children_, model.distances_, counts]
).astype(float)

labels = []

for i in model.labels_:
    labels.append(word_list[i])

#print(labels)
print(len(linkage_matrix))
print(len(labels))
'''

# Plot the corresponding dendrogram
matrix = similarity_of_similarities
#matrix = word_distances

fig2 = ff.create_dendrogram(matrix, labels=word_list)
for i in range(len(fig2['data'])):
    fig2['data'][i]['yaxis'] = 'y2'

# Create Side Dendrogram
dendro_side = ff.create_dendrogram(matrix, orientation='right')
for i in range(len(dendro_side['data'])):
    dendro_side['data'][i]['xaxis'] = 'x2'

# Add Side Dendrogram Data to Figure
for data in dendro_side['data']:
    fig2.add_trace(data)

dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
dendro_leaves = list(map(int, dendro_leaves))

data_dist = pdist(matrix)
heat_data = squareform(data_dist)
heat_data = heat_data[dendro_leaves,:]
heat_data = heat_data[:,dendro_leaves]

print(heat_data)

heatmap = [
    go.Heatmap(
        x = dendro_leaves,
        y = dendro_leaves,
        z = heat_data,
        colorscale = 'RdBu'
    )
]

heatmap[0]['x'] = fig2['layout']['xaxis']['tickvals']
heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

# Add Heatmap Data to Figure
for data in heatmap:
    fig2.add_trace(data)

fig2['layout']['yaxis']['ticktext'] = fig2['layout']['xaxis']['ticktext']
fig2['layout']['yaxis']['tickvals'] = np.asarray(dendro_side['layout']['yaxis']['tickvals'])

# Edit Layout
fig2.update_layout({'width':800, 'height':800,
                         'showlegend':False, 'hovermode': 'closest',
                         })
# Edit xaxis
fig2.update_layout(xaxis={'domain': [.15, 1],
                                  'mirror': False,
                                  'showgrid': False,
                                  'showline': False,
                                  'zeroline': False,
                                  'ticks':""})
# Edit xaxis2
fig2.update_layout(xaxis2={'domain': [0, .15],
                                   'mirror': False,
                                   'showgrid': False,
                                   'showline': False,
                                   'zeroline': False,
                                   'showticklabels': False,
                                   'ticks':""})

# Edit yaxis
fig2.update_layout(yaxis={'domain': [0, .85],
                                  'mirror': False,
                                  'showgrid': False,
                                  'showline': False,
                                  'zeroline': False,
                                  'showticklabels': False,
                                  'ticks': ""
                        })
# Edit yaxis2
fig2.update_layout(yaxis2={'domain':[.825, .975],
                                   'mirror': False,
                                   'showgrid': False,
                                   'showline': False,
                                   'zeroline': False,
                                   'showticklabels': False,
                                   'ticks':""})

fig2.show()

#print(word_distances)