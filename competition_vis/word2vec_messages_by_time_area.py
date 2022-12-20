import word2vec_helpers
import pandas as pd
from datetime import datetime
import numpy as np
from PIL import Image
import plotly.graph_objects as go


#check_if_separator_is_used(separator)

count_map = {}
sorted_times = []

edited_image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map_edited.png"
image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map.png"
im = Image.open(edited_image_filename) # Can be many different formats.
pix = im.load()
sizex, sizey = im.size

colors = np.array(([255, 242, 0], [14, 206, 69], [255, 202, 24], [63, 72, 204], [184, 61, 186], [140, 255, 251], [236, 28, 36], [255, 13, 182], [136, 0, 27], [253, 236, 166], [88, 88, 88], [129, 39, 255], [255, 127, 39]))
districts = ["Cornertown", "Northville", "Villa", "Westside", "Smogtown", "Plainville", "Downtown", "Uptown", "Riverside", "Southville", "Lakeside", "Eastside", "Suburbia"]

areas = np.zeros(len(districts))

with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
    reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["Created_at", "Location"])
    for index, row in reader.iterrows():
        try:
            time = datetime.strptime(row["Created_at"], '%m/%d/%Y %H:%M').replace(minute=0)
            if time not in count_map:
                count_map[time] = np.zeros(len(districts))
                sorted_times.append(time)
            pos = row["Location"]
            x, y = word2vec_helpers.get_coords_in_pixels(pos)
            x = min(x, sizex - 1)
            y = min(y, sizey - 1)
            # PIL has origin in top-left, convert bottom-left origin to this, then fetch pixel color
            y_tl = word2vec_helpers.get_height() - y - 1
            pixel = np.array(pix[x, y_tl])
            asd = colors - pixel
            index = np.argmin(np.absolute(asd.sum(axis=1)))
            areas[index] += 1
            count_map[time][index] += 1
            #x_arr.append(x * scale_factor)
            #y_arr.append(y * scale_factor)
            #labels.append(f"Coordinates: {pos}, area: {districts[index]}")
        except ValueError as e:
            print(f"{e}")


sorted_times = sorted(sorted_times)

zipped = zip(districts, areas)
with open("times.txt", "w") as file:
    for time in sorted_times:
        file.write(f"{time}\n")

with open("msgs_by_time_and_area.txt", "w") as file:
    for time in sorted_times:
        string = ""
        for count in count_map[time]:
            string += str(count) + " "
        string = string.strip()
        string += "\n"
        file.write(string)

# TEMP
'''
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=x_arr, y=y_arr, text=labels, mode="markers")
)

im = Image.open(image_filename) # Can be many different formats.

fig.add_layout_image(
    dict(
        source=im,
        x=0,
        sizex=width * scale_factor,
        y=height * scale_factor,
        sizey=height * scale_factor,
        xref="x",
        yref="y",
        opacity=0.7,
        layer="below",
        sizing="stretch",
    )
)

fig.show()
'''