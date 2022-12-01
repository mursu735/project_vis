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


# If the position and date is in the area of interest, save it, otherwise skip it
target_area = np.array([63, 72, 204])
start_time = datetime(2011, 5, 17, 0, 0)
end_time = datetime(2011, 5, 18, 0, 0)

separator = ':^:'
edited_image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map_edited_first_case_mod.png"
im = Image.open(edited_image_filename) # Can be many different formats.
pix = im.load()

messages = {}
total = 0

times = []

lines = []

with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
    reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["Created_at", "text", "Location"])
    print(reader)
    for index, row in reader.iterrows():
        try:
            time = datetime.strptime(row["Created_at"], '%m/%d/%Y %H:%M').replace(minute=0)
            if time >= start_time and time <= end_time:
                if time not in messages:
                    messages[time] = []
                    times.append(time)
                pos = row["Location"]
                x, y = word2vec_helpers.get_coords_in_pixels(pos)
                # PIL has origin in top-left, convert bottom-left origin to this, then fetch pixel color
                y_tl = word2vec_helpers.get_height() - y - 1
                color = np.array(pix[x, y_tl])
                asd = target_area - color[:3] # Some pixels return [r, g, b, alpha], get rid of alpha
                sum = asd.sum()
                if sum == 0:
                    messages[time].append(row["text"])
                    total += 1
                    text = row["Created_at"] + separator + row["Location"] + separator + row["text"]
                    lines.append(text)
        except ValueError as e:
            text = row["text"]
            print(f"{e}, message: {text}")

sorted_times = sorted(times)


with open("filtered_first_case.txt", "w") as file:
    for line in lines:
        file.write(f"{line}\n")

'''
with open("filtered_first_case_manual.txt", "w") as file:
    for time in sorted_times:
        file.write(f"{time}\n")
        for message in messages[time]:
            file.write(f"{message}\n")

'''
'''
with open("filtered_coords.txt", "w") as file:
    for line in coords:
        file.write(f"{line}\n")
        for line in val:
            file.write(f"{line}\n")
'''

print(f"Remaining number of messages: {len(lines)}")
