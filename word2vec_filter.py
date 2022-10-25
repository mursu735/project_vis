import logging
import word2vec_helpers
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import plotly.express as px
from skimage import io
import plotly.graph_objects as go
import pandas as pd

def check_if_separator_is_used(separator):
    with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
        reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["Created_at", "text", "Location"])
        converted = reader.text.to_list()
        coords = []
        lines = []
        count = 0
        print(reader)
        for index, row in reader.iterrows():
            if separator in row["text"]:
                text = row["text"]
                print(f"Row {text} contains separator {separator}")
                count += 1
    if count == 0:
        print("Row did not contain separator")


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

model_name = word2vec_helpers.fetch_model_name()

print(model_name)

new_model = gensim.models.Word2Vec.load(f"gensim-model-{model_name}")

#wv = api.load('word2vec-google-news-300')

wv = new_model.wv

#wv.evaluate_word_pairs('MC_1_Materials_3-30-2011/Microblogs.csv')

print(wv.most_similar(positive=['sick'], topn=100))

word_list = word2vec_helpers.get_word_list()
#word_list = ["fever", "chills", "sweats", "aches", "pains", "fatigue", "coughing", "breathing", "nausea", "vomiting", "diarrhoea", "lymph node"]
#word_list = ["fever", "chills", "sweats", "aches", "pains", "fatigue", "coughing", "breathing", "nausea", "vomiting", "diarrhoea"]

word_distances = {}

for i in range(0, len(word_list)):
    for j in range(i+1, len(word_list)):
        word = f"{word_list[i]} -> {word_list[j]}"
        diff = wv.similarity(word_list[i], word_list[j])
        word_distances[word] = diff

print(word_distances)
        
# Pairwise distance between symptoms
# Graph renderding, networkx
'''
with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
    reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["Created_at", "text", "Location"])
    converted = reader.text.to_list()
    coords = {}
    lines = {}
    print(reader)
    for index, row in reader.iterrows():
        #print(row)
        #print(row["Created_at"])
        split = row["Created_at"].split(" ")
        time = split[0]
        # "5/18/2011" in row["Created_at"] and
        if any(substring in row["text"] for substring in word_list):
            if time not in lines:
                coords[time] = []
                lines[time] = []
            coords[time].append(row["Location"])
            lines[time].append(row["text"])
'''
separator = ':^:'

#check_if_separator_is_used(separator)


with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
    reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["Created_at", "text", "Location"])
    converted = reader.text.to_list()
    coords = []
    lines = []
    print(reader)
    for index, row in reader.iterrows():
        #print(row)
        #print(row["Created_at"])
        split = row["Created_at"].split(" ")
        time = split[0]
        # "5/18/2011" in row["Created_at"] and
        if any(substring in row["text"] for substring in word_list):
            text = row["Location"] + separator + row["text"]
            coords.append(text)
            lines.append(row["Created_at"])


with open("filtered2.txt", "w") as file:
    for line in lines:
        file.write(f"{line}\n")
'''
        for line in val:
            file.write(f"{line}\n")
'''

with open("filtered_coords.txt", "w") as file:
    for line in coords:
        file.write(f"{line}\n")
'''
        for line in val:
            file.write(f"{line}\n")
'''

print(f"Remaining number of messages: {len(lines)}")
