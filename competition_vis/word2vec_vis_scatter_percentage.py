import numpy as np
import plotly.graph_objects as go
import math
from datetime import datetime, time
from PIL import Image
import pandas as pd
import word2vec_helpers

north_end = 42.3017
west_start = 93.5673
north_start = 42.1609
west_end = 93.1923

width = 5216
height = 2653
image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map.png"
edited_image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map_edited.png"
population_csv = "MC_1_Materials_3-30-2011/Population.csv"
symptom1 = word2vec_helpers.get_disease_1_symptoms()
symptom2 = word2vec_helpers.get_disease_2_symptoms()
other_symptoms = []
tmp = word2vec_helpers.get_word_list()

im = Image.open(edited_image_filename) # Can be many different formats.
pix = im.load()
sizex, sizey = im.size

# Each row correponds to RGB color on edited map, used to group the symptoms
colors = word2vec_helpers.get_district_colors()
districts = ["Cornertown", "Northville", "Villa", "Westside", "Smogtown", "Plainville", "Downtown", "Uptown", "Riverside", "Southville", "Lakeside", "Eastside", "Suburbia"]
symbols = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'pentagon', 'hexagram', 'star-triangle-down', 'diamond-wide', 'hexagon', 'x-dot', 'diamond-wide-dot', 'y-up-open']

reader = pd.read_csv("MC_1_Materials_3-30-2011/Population.csv", sep=",", header=0)
reader = reader.set_index("Zone_Name")
# Have the same order as zone list above
reader = reader.reindex(districts)


row = 0
times = []
unique_times = []
separator = ":^:"
coords_map = {}
counts = {}

for element in tmp:
    if element not in symptom1 and element not in symptom2:
        other_symptoms.append(element)

#reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["Created_at", "text", "Location"])

with open("filtered2.txt") as file:
    for line in file.readlines():
        line = line.replace("\n", "")
        timestamp = datetime.strptime(line, '%m/%d/%Y %H:%M').replace(minute=0)
        times.append(timestamp)
        if timestamp not in counts:
            counts[timestamp] = 1
            unique_times.append(timestamp)
        else:
            counts[timestamp] += 1

counts = np.loadtxt("msgs_by_time_and_area.txt")
text_file = open("times.txt", "r")
lines = text_file.readlines()

count = 0
for i in lines:
    timestamp = i.replace("\n", "")
    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    lines[count] = timestamp
    count += 1

counts_by_time = dict(zip(lines, counts))

commute_start = time(9, 0, 0)
commute_end = time(17, 0, 0)

#Index tells the town part,
with open("filtered_coords.txt") as file:
    for line in file.readlines():
        line = line.replace("\n", "")
        timestamp = times[row]
        # Match coordinate and time, use the previously read times as guidance
        splitted = line.split(separator)
        coord = splitted[0]
        text = splitted[1]
        x, y = word2vec_helpers.get_coords_in_pixels(coord)
        x = min(x, sizex - 1)
        y = min(y, sizey - 1)
        if timestamp not in coords_map:
            coords_map[timestamp] = {"Symptom 1": np.zeros(13), "Symptom 2": np.zeros(13), "Other symptoms": np.zeros(13)}
        y_tl = word2vec_helpers.get_height() - y - 1
        pixel = pix[x, y_tl]
        asd = colors - pixel
        index = np.argmin(np.absolute(asd.sum(axis=1)))
        if any(symptom in text for symptom in symptom1):
            coords_map[timestamp]["Symptom 1"][index] += 1
        elif any(symptom in text for symptom in symptom2):
            coords_map[timestamp]["Symptom 2"][index] += 1
        else:
            coords_map[timestamp]["Other symptoms"][index] += 1
        row += 1

unique_times_sorted = sorted(unique_times)

start_time = datetime(2011, 5, 16, 0, 0)

unique_times_sorted = unique_times_sorted[unique_times_sorted.index(start_time):]

final_map = {}

for timestamp in unique_times_sorted:
    final_map[timestamp] = {
        "Symptom 1": coords_map[timestamp]["Symptom 1"] / counts_by_time[timestamp],
        "Symptom 2": coords_map[timestamp]["Symptom 2"] / counts_by_time[timestamp],
        "Other symptoms": coords_map[timestamp]["Other symptoms"] / counts_by_time[timestamp],
        "Population": reader["Daytime_Population"].values if commute_start <= timestamp.time() <= commute_end else reader["Population_Density"].values
    }


frames=[
    go.Frame(
        data=[
            go.Scatter(
                x=final_map[timestamp]["Population"],
                y=final_map[timestamp]["Symptom 1"],
                name=f"Symptom group 1: {symptom1}",
                mode="markers+text",
                text=districts,
                textposition="top center",
                marker_symbol=symbols
            ),
            go.Scatter(
                x=final_map[timestamp]["Population"],
                y=final_map[timestamp]["Symptom 2"],
                name=f"Symptom group 2: {symptom2}",
                mode="markers+text",
                text=districts,
                textposition="top center",
                marker_symbol=symbols
            ),
            go.Scatter(
                x=final_map[timestamp]["Population"],
                y=final_map[timestamp]["Other symptoms"],
                name=f"Other symptoms: {other_symptoms}",
                mode="markers+text",
                text=districts,
                textposition="top center",
                marker_symbol=symbols
            )
        ],
        name=timestamp.strftime('%m/%d/%Y %H:%M')
    )
    for timestamp in unique_times_sorted]


steps = []
for i in range(len(unique_times_sorted)):
    timestamp = unique_times_sorted[i]
    name = timestamp.strftime('%m/%d/%Y %H:%M')
    step = dict(
        method="animate",
        args=[
            [name],
            {"frame": {"duration": 1000, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 500, "easing": "cubic-in-out"}}
        ],
        label=name
    )
    steps.append(step)

sliders = [dict(
    pad={"t": 50},
    steps=steps
)]

# Defining figure
fig = go.Figure(
    data=[
        go.Scatter(
            x=final_map[timestamp]["Population"],
            y=final_map[unique_times_sorted[0]]["Symptom 1"],
            name=f"Symptom group 1: {symptom1}",
            mode="markers+text",
            text=districts,
            textposition="top center",
            marker_symbol=symbols
        ),
        go.Scatter(
            x=final_map[timestamp]["Population"],
            y=final_map[unique_times_sorted[0]]["Symptom 2"],
            name=f"Symptom group 2: {symptom2}",
            mode="markers+text",
            text=districts,
            textposition="top center",
            marker_symbol=symbols
        ),
        go.Scatter(
            x=final_map[timestamp]["Population"],
            y=final_map[unique_times_sorted[0]]["Other symptoms"],
            name=f"Other symptoms: {other_symptoms}",
            mode="markers+text",
            text=districts,
            textposition="top center",
            marker_symbol=symbols
        )
    ],
    layout=go.Layout( # Styling
        scene=dict(
        ),
        updatemenus=[
            dict(
                type='buttons',
                buttons=[
                    dict(
                        label='Play',
                        method='animate',
                        args=[None, {"frame": {"duration": 1000, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 500,
                                                                    "easing": "quadratic-in-out"}}]

                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0, "redraw": False},
                                        "mode": "immediate",
                                        "transition": {"duration": 0}}],
                )
                ]
            )
        ]
    ),
    frames=frames
)

for i in range(0, len(districts)):
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            name=districts[i],
            mode="markers",
            marker_symbol=symbols[i],
            marker_color="black"
        )
    )

fig.update_layout(
    sliders=sliders,
    yaxis_range=[0.0, 1.0],
    title="Percentage of relevant messages in relation to area population",
    xaxis_title="Area population",
    yaxis_title="Percentage of sickness-related messages"
)


'''yranges = {}

for time in coords_map:
    cur = int(max(np.amax(final_map[time]["Symptom 1"]), np.amax(coords_map[time]["Symptom 2"]), np.amax(coords_map[time]["Other symptoms"])))
    key = time.strftime('%m/%d/%Y %H:%M')
    cur = math.ceil(cur/10)*10
    yranges[key] = [0, cur]


for f in fig.frames:
    if f.name in yranges.keys():
        f.layout.update(yaxis_range = yranges[f.name])'''


fig.show()

#fig.write_html("server/bar_chart_simple.html", auto_play=False, include_plotlyjs="cdn")

# Cornertown = [255, 239, 54]
# Northville = [54, 206, 78]
# Villa = [254, 200, 54]
# Westside = [61, 79, 201]
# Smogtown = [180, 70, 184]
# Plainville = [148, 254, 250]
# Downtown = [236, 36, 45]
# Uptown = [249, 42, 181]
# Riverside = [133, 7, 30]
# Southville = [253, 235, 170]
# Lakeside = [88, 88, 88]
# Eastside = [125, 61, 251]
# Suburbia = [251, 127, 54]