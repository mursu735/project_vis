import numpy as np
import plotly.graph_objects as go
import math
from datetime import datetime
from PIL import Image
import word2vec_helpers

north_end = 42.3017
west_start = 93.5673
north_start = 42.1609
west_end = 93.1923

width = 5216
height = 2653
image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map.png"
edited_image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map_edited.png"
symptom1 = word2vec_helpers.get_disease_1_symptoms()
symptom2 = word2vec_helpers.get_disease_2_symptoms()

im = Image.open(edited_image_filename) # Can be many different formats.
pix = im.load()
sizex, sizey = im.size

# Each row correponds to RGB color on edited map, used to group the symptoms
colors = np.array(([255, 239, 54], [54, 206, 78], [254, 200, 54], [61, 79, 201], [180, 70, 184], [148, 254, 250], [236, 36, 45], [249, 42, 181], [133, 7, 30], [253, 235, 170], [88, 88, 88], [125, 61, 251], [251, 127, 54]))
districts = ["Cornertown", "Northville", "Villa", "Westside", "Smogtown", "Plainville", "Downtown", "Uptown", "Riverside", "Southville", "Lakeside", "Eastside", "Suburbia"]

row = 0
times = []
unique_times = []
separator = ":^:"
coords_map = {}
counts = {}

with open("filtered2.txt") as file:
    for line in file.readlines():
        line = line.replace("\n", "")
        time = datetime.strptime(line, '%m/%d/%Y %H:%M').replace(minute=0)
        times.append(time)
        if time not in counts:
            counts[time] = 1
            unique_times.append(time)
        else:
            counts[time] += 1

#Index tells the town part,
with open("filtered_coords.txt") as file:
    for line in file.readlines():
        line = line.replace("\n", "")
        time = times[row]
        # Match coordinate and time, use the previously read times as guidance
        splitted = line.split(separator)
        coord = splitted[0]
        text = splitted[1]
        x, y = word2vec_helpers.get_coords_in_pixels(coord)
        x = min(x, sizex - 1)
        y = min(y, sizey - 1)
        if time not in coords_map:
            coords_map[time] = {"Symptom 1": np.zeros(13), "Symptom 2": np.zeros(13), "Other symptoms": np.zeros(13)}
        y_tl = word2vec_helpers.get_height() - y - 1
        pixel = pix[x, y_tl]
        asd = colors - pixel
        index = np.argmin(np.absolute(asd.sum(axis=1)))
        if any(symptom in text for symptom in symptom1):
            coords_map[time]["Symptom 1"][index] += 1
        elif any(symptom in text for symptom in symptom2):
            coords_map[time]["Symptom 2"][index] += 1
        else:
            coords_map[time]["Other symptoms"][index] += 1
        row += 1


unique_times_sorted = sorted(unique_times)

start_time = datetime(2011, 5, 17, 0, 0)

unique_times_sorted = unique_times_sorted[unique_times_sorted.index(start_time):]

frames=[
    go.Frame(
        data=[
            go.Bar(
                x=districts,
                y=coords_map[time]["Symptom 1"],
                name=f"Symptom group 1: {symptom1}"
            ),
            go.Bar(
                x=districts,
                y=coords_map[time]["Symptom 2"],
                name=f"Symptom group 2: {symptom2}"
            ),
            go.Bar(
            x=districts,
            y=coords_map[time]["Other symptoms"],
            name="Other symptoms"
        )
        ],
        name=time.strftime('%m/%d/%Y %H:%M')
    )
    for time in unique_times_sorted]


steps = []
for i in range(len(unique_times_sorted)):
    time = unique_times_sorted[i]
    name = time.strftime('%m/%d/%Y %H:%M')
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
        go.Bar(
            x=districts,
            y=coords_map[unique_times_sorted[0]]["Symptom 1"],
            name=f"Symptom group 1: {symptom1}"
        ),
        go.Bar(
            x=districts,
            y=coords_map[unique_times_sorted[0]]["Symptom 2"],
            name=f"Symptom group 2: {symptom2}"
        ),
        go.Bar(
            x=districts,
            y=coords_map[unique_times_sorted[0]]["Other symptoms"],
            name="Other symptoms"
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
#fig.show()


fig.update_layout(barmode='group', xaxis_tickangle=-45, sliders=sliders)

yranges = {}

for time in coords_map:
    cur = int(max(np.amax(coords_map[time]["Symptom 1"]), np.amax(coords_map[time]["Symptom 2"]), np.amax(coords_map[time]["Other symptoms"])))
    key = time.strftime('%m/%d/%Y %H:%M')
    cur = math.ceil(cur/10)*10
    yranges[key] = [0, cur]


for f in fig.frames:
    if f.name in yranges.keys():
        f.layout.update(yaxis_range = yranges[f.name])


fig.show()

# TODO:  add frames to barchart


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