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
edited_image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map_edited.png"
separator = ":^:"
symptom1 = word2vec_helpers.get_disease_1_symptoms()
symptom2 = word2vec_helpers.get_disease_2_symptoms()
other_symptoms = []
tmp = word2vec_helpers.get_word_list()

for element in tmp:
    if element not in symptom1 and element not in symptom2:
        other_symptoms.append(element)

fig = go.Figure()

#Dict, key is time, value is {x: [], y: [], text: []}, iterate through sorted list of times, access dict with them, use Frames
coords_map = {}
unique_times = []
unique_times_precise = []
interesting_messages = {}

#text_file = "filtered_smogtown_users.txt"
text_file = "filtered_first_case.txt"

with open(text_file) as file:
    for line in file.readlines():
        line = line.replace("\n", "")
        # Match coordinate and time, use the previously read times as guidance
        splitted = line.split(separator)
        group = splitted[0]
        time_precise = datetime.strptime(splitted[1], '%m/%d/%Y %H:%M')
        time = time_precise.replace(minute=0)
        coord = splitted[2]
        id = splitted[4]
        text = "ID: " + id + ";" + coord + " / " + splitted[3]

        x_interpolate, y_interpolate = word2vec_helpers.get_coords_in_pixels(coord)
        if time not in coords_map:
            coords_map[time] = {"Symptom1": {"x": [], "y": [], "text": []},
                                "Symptom2": {"x": [], "y": [], "text": []},
                                "Other": {"x": [], "y": [], "text": []},
                                "Filling": {"x": [], "y": [], "text": []}}
            unique_times.append(time)

        if group == "Symptom 1":
            coords_map[time]["Symptom1"]["x"].append(x_interpolate * scale_factor)
            coords_map[time]["Symptom1"]["y"].append(y_interpolate * scale_factor)
            coords_map[time]["Symptom1"]["text"].append(text)
            if time_precise not in interesting_messages:
                unique_times_precise.append(time_precise)
                interesting_messages[time_precise] = []
            asd = "Symptom 1: " + text
            interesting_messages[time_precise].append(asd)
            #coords_map[time]["label"].append("red")
        elif group == "Symptom 2":
            coords_map[time]["Symptom2"]["x"].append(x_interpolate * scale_factor)
            coords_map[time]["Symptom2"]["y"].append(y_interpolate * scale_factor)
            coords_map[time]["Symptom2"]["text"].append(text)
            if time_precise not in interesting_messages:
                unique_times_precise.append(time_precise)
                interesting_messages[time_precise] = []
            asd = "Symptom 2: " + text
            interesting_messages[time_precise].append(asd)
            #coords_map[time]["label"].append("blue")
        elif group == "Other":
            coords_map[time]["Other"]["x"].append(x_interpolate * scale_factor)
            coords_map[time]["Other"]["y"].append(y_interpolate * scale_factor)
            coords_map[time]["Other"]["text"].append(text)
            if time_precise not in interesting_messages:
                unique_times_precise.append(time_precise)
                interesting_messages[time_precise] = []
            asd = "Other: " + text
            interesting_messages[time_precise].append(asd)
            #coords_map[time]["label"].append("black")
        else:
            coords_map[time]["Filling"]["x"].append(x_interpolate * scale_factor)
            coords_map[time]["Filling"]["y"].append(y_interpolate * scale_factor)
            coords_map[time]["Filling"]["text"].append(text)
            #coords_map[time]["label"].append("black")

unique_times_sorted = sorted(unique_times)

unique_times_precise_sorted = sorted(unique_times_precise)

with open("filtered_first_case_interesting.txt", "w") as file:
    for time in unique_times_precise_sorted:
        file.write(f"{time}:\n")
        for message in interesting_messages[time]:
            file.write(f"{message}\n")

for time in unique_times_sorted:
    if len(coords_map[time]["Symptom1"]["x"]) == 0:
        coords_map[time]["Symptom1"]["x"].append(width + 20)
        coords_map[time]["Symptom1"]["y"].append(height + 20)
        coords_map[time]["Symptom1"]["text"].append("padding")

    if len(coords_map[time]["Symptom2"]["x"]) == 0:
        coords_map[time]["Symptom2"]["x"].append(width + 20)
        coords_map[time]["Symptom2"]["y"].append(height + 20)
        coords_map[time]["Symptom2"]["text"].append("padding")

    if len(coords_map[time]["Other"]["x"]) == 0:
        coords_map[time]["Other"]["x"].append(width + 20)
        coords_map[time]["Other"]["y"].append(height + 20)
        coords_map[time]["Other"]["text"].append("padding")

    if len(coords_map[time]["Filling"]["x"]) == 0:
        coords_map[time]["Filling"]["x"].append(width + 20)
        coords_map[time]["Filling"]["y"].append(height + 20)
        coords_map[time]["Filling"]["text"].append("padding")

frames=[
    go.Frame(
        data=[
            go.Scatter(
            x=coords_map[time]["Filling"]["x"],
            y=coords_map[time]["Filling"]["y"],
            text=coords_map[time]["Filling"]["text"],
            name="No symptoms",
            mode="markers",
            marker=dict(
                    size=marker_size,
                    color="black",
                    opacity=0.5
                )
            ),
            go.Scatter(
                x=coords_map[time]["Symptom1"]["x"],
                y=coords_map[time]["Symptom1"]["y"],
                text=coords_map[time]["Symptom1"]["text"],
                name=f"Symptom group 1: {symptom1}",
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="red",
                    opacity=1
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
                    opacity=1
                )
            ),
            go.Scatter(
            x=coords_map[time]["Other"]["x"],
            y=coords_map[time]["Other"]["y"],
            text=coords_map[time]["Other"]["text"],
            name=f"Other symptoms: {other_symptoms}",
            mode="markers",
            marker=dict(
                    size=marker_size,
                    color="yellow",
                    opacity=1
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
            x=coords_map[time]["Filling"]["x"],
            y=coords_map[time]["Filling"]["y"],
            text=coords_map[time]["Filling"]["text"],
            name="No symptoms",
            mode="markers",
            marker=dict(
                    size=marker_size,
                    color="black",
                    opacity=0.5
                )
            ),
        go.Scatter(
                x=coords_map[unique_times_sorted[0]]["Symptom1"]["x"],
                y=coords_map[unique_times_sorted[0]]["Symptom1"]["y"],
                text=coords_map[unique_times_sorted[0]]["Symptom1"]["text"],
                name=f"Symptom group 1: {symptom1}",
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="red",
                    opacity=1
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
                    opacity=1
                )
            ),
            go.Scatter(
                x=coords_map[unique_times_sorted[0]]["Other"]["x"],
                y=coords_map[unique_times_sorted[0]]["Other"]["y"],
                text=coords_map[unique_times_sorted[0]]["Other"]["text"],
                name=f"Other symptoms: {other_symptoms}",
                mode="markers",
                marker=dict(
                    size=marker_size,
                    color="yellow",
                    opacity=1
                )
            ),
            go.Scatter(
                x=[0, width * scale_factor],
                y=[0, height * scale_factor],
                mode="markers",
                marker_opacity=0
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

#fig.write_html("server/bar_chart_simple.html", auto_play=False, include_plotlyjs="cdn")
