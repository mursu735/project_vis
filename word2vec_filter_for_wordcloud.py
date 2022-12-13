import logging
import word2vec_helpers
import numpy as np
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import plotly.express as px
from datetime import datetime
import pandas as pd
from PIL import Image
import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet


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

def select_rows(row, pix, sizex, sizey):
    #print(row)
    try:
        time = datetime.strptime(row.Created_at, '%m/%d/%Y %H:%M').replace(minute=0)
        if time == target_time:
            pos = row.Location
            x, y = word2vec_helpers.get_coords_in_pixels(pos)
            x = min(x, sizex - 1)
            y = min(y, sizey - 1)
            # PIL has origin in top-left, convert bottom-left origin to this, then fetch pixel color
            y_tl = word2vec_helpers.get_height() - y - 1
            color = np.array(pix[x, y_tl])
            asd = target_area - color[:3] # Some pixels return [r, g, b, alpha], get rid of alpha
            sum = asd.sum()
            if sum == 0:
                return True
    except ValueError as e:
        text = row["text"]
        print(f"{e}, message: {text}")
        return False
    return False



# If the position and date is in the area of interest, save it, otherwise skip it
target_area = np.array([63, 72, 204])
# Explosion in Smogtown
#target_time = datetime(2011, 5, 17, 9, 0)
#edited_image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map_smogtown.png"
# Traffic accident near Interstate 610
target_time = datetime(2011, 5, 17, 11, 0)
edited_image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map_bridge.png"

im = Image.open(edited_image_filename) # Can be many different formats.
pix = im.load()
sizex, sizey = im.size

# Get IDs that were in the target area when the explosion happened
reader = pd.read_csv('MC_1_Materials_3-30-2011/Microblogs.csv', sep=",", header=0, usecols=["ID", "Created_at", "text", "Location"])
print(reader)
total = len(reader.index)
reader['target_area'] = reader.apply(select_rows, args=(pix, sizex, sizey), axis=1)
reader = reader[reader['target_area']]
#print(reader)

lines = []

for index, row in reader.iterrows():
    lines.append(row["text"])

with open("filtered_wordcloud.txt", "w", encoding="utf-8") as file:
    for line in lines:
        file.write(f"{line}\n")

print(f"Remaining number of messages: {len(lines)}")
