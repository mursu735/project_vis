import logging
import word2vec_helpers
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import plotly.express as px
from datetime import datetime
import pandas as pd
from PIL import Image

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


print(wv.most_similar(positive=['sick'], topn=100))

word_list = word2vec_helpers.get_word_list()

word_distances = {}

for i in range(0, len(word_list)):
    for j in range(i+1, len(word_list)):
        word = f"{word_list[i]} -> {word_list[j]}"
        diff = wv.similarity(word_list[i], word_list[j])
        word_distances[word] = diff

print(word_distances)
        


separator = ':^:'
edited_image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map_edited.png"
im = Image.open(edited_image_filename) # Can be many different formats.
pix = im.load()

start_time = datetime(2011, 5, 17, 0, 0)
end_time = datetime(2011, 5, 18, 0, 0)

with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
    reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["Created_at", "text", "Location"])
    converted = reader.text.to_list()
    coords = []
    lines = []
    print(reader)
    for index, row in reader.iterrows():
        split = row["Created_at"].split(" ")
        time = split[0]
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
