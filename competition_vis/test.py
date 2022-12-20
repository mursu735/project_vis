import logging
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
from datetime import datetime
import re
from PIL import Image
import plotly.graph_objects as go
import base64
import numpy as np

import word2vec_helpers

import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

'''
regex = re.compile("^5\/((1[7-9])|(2[0-9]))\/2011")

dates = []

with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
    reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["text", "Created_at"])
    for index, row in reader.iterrows():
        #print(line)
        # assume there's one document per line, tokens separated by whitespace
        if (re.match(regex, row["Created_at"])):
            try:
                time = datetime.strptime(row["Created_at"], '%m/%d/%Y %H:%M').replace(minute=0, hour=0).strftime('%m/%d/%Y %H:%M')
                if time not in dates:
                    dates.append(time)
            except ValueError:
                print(f"Invalid date: {row['Created_at']}")
dates.sort()

print(dates)
'''
##################################################

'''
im = Image.open('MC_1_Materials_3-30-2011/Vastopolis_Map_edited.png') # Can be many different formats.
pix = im.load()
print(im.size)  # Get the width and hight of the image for iterating over
print(f"Top left: {pix[2,2]}")  # Get the RGBA Value of the a pixel of an image
print(f"Top middle: {pix[2820,2]}")  # Get the RGBA Value of the a pixel of an image
print(f"Top middle: {pix[3600,2]}")  # Get the RGBA Value of the a pixel of an image

x_arr = []
y_arr = []

coord_arr = ["42.300 93.5600", "42.2200 93.4200", "42.3000 93.1930", "42.1620 93.1924", "42.1620 93.5600", "42.250 93.4250", "42.3000 93.5000", "42.3000 93.3580", "42.250 93.3580", "42.2200 93.3580", "42.2500 93.5000", "42.1620 93.420", "42.1800 93.2700", "42.2240 93.2700"]


for coords in coord_arr:
    x, y = word2vec_helpers.get_coords_in_pixels(coords)
    x_arr.append(x)
    y_arr.append(y)
    y_tl = word2vec_helpers.get_height() - y - 1
    color = pix[x, y_tl]
    print(f"Coordinate: {coords} / color: {color}")


fig2 = go.Figure(data=[
        go.Scatter(
            x=x_arr,
            y=y_arr,
            text=coord_arr,
            mode="markers",
            marker=dict(
            size=5,
            color="red"
            ))
        ]
    )
map_plot = base64.b64encode(open('MC_1_Materials_3-30-2011/Vastopolis_Map_edited.png', 'rb').read())

fig2.update_xaxes(range=[0, word2vec_helpers.get_width()])
fig2.update_yaxes(range=[0, word2vec_helpers.get_height()])


fig2.update_layout(
                title = "Animation of message locations for each hour, heuristics",
                images = [dict(
                    source='data:image/png;base64,{}'.format(map_plot.decode()),
                    xref="paper", yref="paper",
                    x=0, y=0,
                    sizex=1, sizey=1,
                    xanchor="left",
                    yanchor="bottom",
                    sizing="fill",
                    opacity=0.6,
                    layer="below")])

fig2.show()
'''
##################################################

'''

dataset = pd.read_csv("Binary_classification/Training_data/training_data_pre_ob.csv", sep=",", header=0, usecols=["ID", "Created_at", "Location", "text", "label"])
sickness = 0
other = 0

for index, row in dataset.iterrows():
    if int(row["label"]) == 1:
        sickness += 1
    else:
        other += 1

print(f"Sickness messages, pre-outbreak: {sickness}")
print(f"Other messages, pre-outbreak: {other}")


dataset2 = pd.read_csv("Binary_classification/Training_data/training_data_post_ob.csv", sep=",", header=0, usecols=["ID", "Created_at", "Location", "text", "label"])
sickness = 0
other = 0

for index, row in dataset2.iterrows():
    if int(row["label"]) == 1:
        sickness += 1
    else:
        other += 1

print(f"Sickness messages, post-outbreak: {sickness}")
print(f"Other messages, post-outbreak: {other}")

'''
##################################################
'''
counts = np.loadtxt("msgs_by_time_and_area.txt")
text_file = open("times.txt", "r")
lines = text_file.readlines()

count = 0
for i in lines:
    time = i.replace("\n", "")
    time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    lines[count] = time
    count += 1

#print(lines)
qwe = dict(zip(lines, counts))
print(qwe)
#qew = np.arange(0.0,5.0,1.0)

#for a in qew:
#    print(a)
'''
################################################
'''
reader = pd.read_csv("MC_1_Materials_3-30-2011/Population.csv", sep=",", header=0)
print(reader)

reader = reader.set_index("Zone_Name")
districts = ["Cornertown", "Northville", "Villa", "Westside", "Smogtown", "Plainville", "Downtown", "Uptown", "Riverside", "Southville", "Lakeside", "Eastside", "Suburbia"]

reader = reader.reindex(districts)

print(reader["Population_Density"].values)
'''
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

sentence = "David has a case of the chills wish they were feeling better"

'''
lmtzr = WordNetLemmatizer()
lemmatized = [[lmtzr.lemmatize(word) for word in word_tokenize(s)]
              for s in a]
print(lemmatized)
'''
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


lemmatizer = WordNetLemmatizer()
nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
#tuple of (token, wordnet_tag)
wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
lemmatized_sentence = []
for word, tag in wordnet_tagged:
    if tag is None:
        #if there is no available tag, append the token as is
        lemmatized_sentence.append(word)
    else:
        #else use the tag to lemmatize the token
        lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
print(lemmatized_sentence)

word_list = set(word2vec_helpers.get_word_list())
asd = set(lemmatized_sentence)
concat = ' '.join(lemmatized_sentence)
if word_list.intersection(asd):
    print("It works!")