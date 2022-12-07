import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from skimage import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as py
import re
import base64
from PIL import Image
from datetime import datetime
import word2vec_helpers

north_end = 42.3017
west_start = 93.5673
north_start = 42.1609
west_end = 93.1923

width = 5216
height = 2653
scale_factor = 0.5
marker_size = 8
image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map.png"
separator = ":^:"
symptom1 = word2vec_helpers.get_disease_1_symptoms()
symptom2 = word2vec_helpers.get_disease_2_symptoms()

fig = go.Figure()

counter = -1
pattern = re.compile("^[4-5]\/[0-3]?[0-9]\/2011$")

counts = {}
times = []
unique_times = []

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

row = 0

#Dict, key is time, value is {x: [], y: [], text: []}, iterate through sorted list of times, access dict with them, use Frames
coords_map = {}

with open("filtered_coords.txt") as file:
    for line in file.readlines():
        line = line.replace("\n", "")
        time = times[row]
        # Match coordinate and time, use the previously read times as guidance
        splitted = line.split(separator)
        coord = splitted[0]
        text = splitted[1]
        x_interpolate, y_interpolate = word2vec_helpers.get_coords_in_pixels(coord)
        if time not in coords_map:
            coords_map[time] = {"Symptom1": {"x": [], "y": [], "text": []}, "Symptom2": {"x": [], "y": [], "text": []}, "Other": {"x": [], "y": [], "text": []}}
        if any(symptom in text for symptom in symptom1):
            coords_map[time]["Symptom1"]["x"].append(x_interpolate * scale_factor)
            coords_map[time]["Symptom1"]["y"].append(y_interpolate * scale_factor)
            coords_map[time]["Symptom1"]["text"].append(text)
            #coords_map[time]["label"].append("red")
        elif any(symptom in text for symptom in symptom2):
            coords_map[time]["Symptom2"]["x"].append(x_interpolate * scale_factor)
            coords_map[time]["Symptom2"]["y"].append(y_interpolate * scale_factor)
            coords_map[time]["Symptom2"]["text"].append(text)
            #coords_map[time]["label"].append("blue")
        else:
            coords_map[time]["Other"]["x"].append(x_interpolate * scale_factor)
            coords_map[time]["Other"]["y"].append(y_interpolate * scale_factor)
            coords_map[time]["Other"]["text"].append(text)
            #coords_map[time]["label"].append("black")
        row += 1

sorted_counts = sorted(counts.items())
# print(sorted)
unique_times_sorted = sorted(unique_times)

if len(coords_map[unique_times_sorted[0]]["Symptom1"]["x"]) == 0:
    coords_map[unique_times_sorted[0]]["Symptom1"]["x"].append(width + 20)
    coords_map[unique_times_sorted[0]]["Symptom1"]["y"].append(height + 20)
    coords_map[unique_times_sorted[0]]["Symptom1"]["text"].append("padding")

if len(coords_map[unique_times_sorted[0]]["Symptom2"]["x"]) == 0:
    coords_map[unique_times_sorted[0]]["Symptom2"]["x"].append(width + 20)
    coords_map[unique_times_sorted[0]]["Symptom2"]["y"].append(height + 20)
    coords_map[unique_times_sorted[0]]["Symptom2"]["text"].append("padding")

if len(coords_map[unique_times_sorted[0]]["Other"]["x"]) == 0:
    coords_map[unique_times_sorted[0]]["Other"]["x"].append(width + 20)
    coords_map[unique_times_sorted[0]]["Other"]["y"].append(height + 20)
    coords_map[unique_times_sorted[0]]["Other"]["text"].append("padding")

frames=[
    go.Frame(
        data=[
            go.Scatter(
                x=coords_map[time]["Symptom1"]["x"],
                y=coords_map[time]["Symptom1"]["y"],
                text=coords_map[time]["Symptom1"]["text"],
                name=f"Symptom group 1: {symptom1}",
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="red",
                    opacity=0.5
                )
            ),
            go.Scatter(
                x=coords_map[time]["Symptom2"]["x"],
                y=coords_map[time]["Symptom2"]["y"],
                text=coords_map[time]["Symptom2"]["text"],
                name=f"Symptom group 2: {symptom2}",
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="blue",
                    opacity=0.5
                )
            ),
            go.Scatter(
            x=coords_map[time]["Other"]["x"],
            y=coords_map[time]["Other"]["y"],
            text=coords_map[time]["Other"]["text"],
            name="Other symptoms",
            mode="markers",
            marker=dict(
                    size=marker_size,
                    color="black",
                    opacity=0.5
                )
        ),
        go.Scatter(
            x=[0, width * scale_factor],
            y=[0, height * scale_factor],
            mode="markers",
            marker_opacity=0
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
            {"frame": {"duration": 0, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 0, "easing": "cubic-in-out"}}
        ],
        label=name
    )
    steps.append(step)

sliders = [dict(
    pad={"t": 50},
    steps=steps
)]

fig = go.Figure(
    data=[
        go.Scatter(
                x=coords_map[unique_times_sorted[0]]["Symptom1"]["x"],
                y=coords_map[unique_times_sorted[0]]["Symptom1"]["y"],
                text=coords_map[unique_times_sorted[0]]["Symptom1"]["text"],
                name=f"Symptom group 1: {symptom1}",
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="red",
                    opacity=0.5
                )
            ),
            go.Scatter(
                x=coords_map[unique_times_sorted[0]]["Symptom2"]["x"],
                y=coords_map[unique_times_sorted[0]]["Symptom2"]["y"],
                text=coords_map[unique_times_sorted[0]]["Symptom2"]["text"],
                name=f"Symptom group 2: {symptom2}",
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="blue",
                    opacity=0.5
                )
            ),
            go.Scatter(
                x=coords_map[unique_times_sorted[0]]["Other"]["x"],
                y=coords_map[unique_times_sorted[0]]["Other"]["y"],
                text=coords_map[unique_times_sorted[0]]["Other"]["text"],
                name="Other symptoms",
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="black",
                    opacity=0.5
                )
            ),
            go.Scatter(
                x=[0, width * scale_factor],
                y=[0, height * scale_factor],
                mode="markers",
                name="",
                marker_opacity=0,
                marker=dict(
                    color="white"
                )
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
                        args=[None, {"frame": {"duration": 0, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 0,
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
        ],
        yaxis=dict(range=[0, height], autorange=False),
        xaxis=dict(range=[0, width], autorange=False)
    ),
    
    frames=frames
)

fig.update_layout(sliders=sliders)

# Configure axes
fig.update_xaxes(
    visible=True,
    range=[0, width * scale_factor]
)

fig.update_yaxes(
    visible=True,
    range=[0, height * scale_factor],
    # the scaleanchor attribute ensures that the aspect ratio stays constant
    scaleanchor="x"
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

fig.update_layout(
    width=width * scale_factor,
    height=height * scale_factor,
    margin={"l": 0, "r": 0, "t": 0, "b": 0},
)

fig.show()

fig.write_html("server/map_plot.html", auto_play=False, include_plotlyjs="cdn")
