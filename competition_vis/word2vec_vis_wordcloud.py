import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from skimage import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import base64
from PIL import Image
from datetime import datetime
import word2vec_helpers
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

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
#title = "Explosion in Smogtown"

# Traffic accident near Interstate 610
#target_time = datetime(2011, 5, 17, 11, 0)
#edited_image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map_bridge.png"
#title = "Accident on Interstate 610"

# Traffic accident near Interstate 270
target_time = datetime(2011, 5, 17, 9, 0)
edited_image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map_bridge_other.png"
title = "Accident on Interstate 270"

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

text = " ".join(line for line in lines)

wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=1200, height=900).generate(text)

fig = px.imshow(wordcloud, title=title) # image show
fig.update_layout(title_x=0.5)

fig.show()
