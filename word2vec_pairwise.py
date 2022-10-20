import fetch_model_name
import gensim.models
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import networkx as nx
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np
from sklearn.cluster import AgglomerativeClustering


model_name = fetch_model_name.fetch_model_name()


new_model = gensim.models.Word2Vec.load(f"gensim-model-{model_name}")

#wv = api.load('word2vec-google-news-300')


wv = new_model.wv

word_list = ["fever", "chills", "sweats", "aches", "pains", "fatigue", "coughing", "breathing", "nausea", "vomiting", "diarrhoea"]
#word_list = ["sick", "sleepy", "uncomfortable", "dizzy", "nauseous", "unwell", "bedridden", "coughing", "fever", "hospitalized", "headache", "rashes"]

word_distances = np.zeros((len(word_list), len(word_list)))

for i in range(0, len(word_list)):
    for j in range(0, len(word_list)):
        diff = wv.similarity(word_list[i], word_list[j])
        word_distances[i, j] = diff

print(word_distances)

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

model = AgglomerativeClustering(affinity='cosine', linkage="single", distance_threshold=0, n_clusters=None).fit(similarity_of_similarities)

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

# Plot the corresponding dendrogram
fig2 = ff.create_dendrogram(linkage_matrix)
fig2.update_layout(width=800, height=500)
fig2.show()

# TODO:
# Add more words (1000 random words)
# Spot the exact time (DONE) and location for outbreak start (Add animation for outbreak), 
# Add alpha to map
# Spot the first message (?)
# Check for more symptoms
# Heatmap of features, similarity of similarity; create a vector of similarity for all symptoms, then compare the similarity of those vectors (cosine)
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html?highlight=clustering
# Self organized map

#print(word_distances)