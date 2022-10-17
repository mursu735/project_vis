import fetch_model_name
import gensim.models
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
import numpy as np


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

fig.show()
# TODO:
# Add more words (1000 random words)
# Spot the exact time (DONE) and location for outbreak start (Add animation for outbreak), DONE
# Check for more symptoms
# Heatmap of features, similarity of similarity; create a vector of similarity for all symptoms, then compare the similarity of those vectors (cosine)
# Self organized map

#print(word_distances)