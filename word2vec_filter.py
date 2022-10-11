import logging
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import plotly.express as px
from skimage import io
import plotly.graph_objects as go
import fetch_model_name


import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

model_name = fetch_model_name.fetch_model_name()

print(model_name)

new_model = gensim.models.Word2Vec.load(f"gensim-model-{model_name}")

#wv = api.load('word2vec-google-news-300')

wv = new_model.wv

#wv.evaluate_word_pairs('MC_1_Materials_3-30-2011/Microblogs.csv')

print(wv.most_similar(positive=['sick'], topn=100))

#word_list = ["sick", "sleepy", "uncomfortable", "dizzy", "nauseous", "unwell", "bedridden", "coughing", "fever", "hospitalized", "headache", "rashes"]
#word_list = ["fever", "chills", "sweats", "aches", "pains", "fatigue", "coughing", "breathing", "nausea", "vomiting", "diarrhoea", "lymph node"]
word_list = ["fever", "chills", "sweats", "aches", "pains", "fatigue", "coughing", "breathing", "nausea", "vomiting", "diarrhoea"]

word_distances = {}

for i in range(0, len(word_list)):
    for j in range(i+1, len(word_list)):
        word = f"{word_list[i]} -> {word_list[j]}"
        diff = wv.similarity(word_list[i], word_list[j])
        word_distances[word] = diff

print(word_distances)
        
# Pairwise distance between symptoms
# Graph renderding, networkx

with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
    reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["Created_at", "text", "Location"])
    converted = reader.text.to_list()
    coords = []
    lines = []
    print(reader)
    for index, row in reader.iterrows():
        #print(row)
        #print(row["Created_at"])
        # "5/18/2011" in row["Created_at"] and
        if any(substring in row["text"] for substring in word_list):
            asd = row["text"]
            coords.append(row["Location"])
            lines.append(row["text"])

with open("filtered.txt", "w") as file:
    for line in lines:
        file.write(f"{line}\n")

with open("filtered_coords.txt", "w") as file:
    for line in coords:
        file.write(f"{line}\n")

print(f"Remaining number of messages: {len(lines)}")