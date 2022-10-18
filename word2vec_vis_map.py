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
from datetime import datetime
from PIL import Image

north_end = 42.3017
west_start = 93.5673
north_start = 42.1609
west_end = 93.1923

width = 5216
height = 2653
image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map.png"

fig = go.Figure()

x_arr = []
y_arr = []
dates = []
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

#Dict, key is time, value is {x: [], y: []}, iterate through sorted list of times, access dict with them, use Frames
coords_map = {}

with open("filtered_coords.txt") as file:
    for line in file.readlines():
        line = line.replace("\n", "")
        time = times[row]
        # Match coordinate and time, use the previously read times as guidance
        coords = line.split(" ")
        x = float(coords[0])
        y = float(coords[1])
        x_interpolate = ((x - north_start) / (north_end - north_start)) * width
        y_interpolate = ((y - west_start) / (west_end - west_start)) * height
        if time not in coords_map:
            coords_map[time] = {"x": [], "y": []}
        coords_map[time]["x"].append(x_interpolate)
        coords_map[time]["y"].append(y_interpolate)
        row += 1


sorted_counts = sorted(counts.items())
# print(sorted)
unique_times_sorted = sorted(unique_times)


fig_dict = {
    "data": [],
    "layout": {},
    "frames": []
}

fig_dict["layout"]["xaxis"] = {"range": [0, width], "autorange": False}
fig_dict["layout"]["yaxis"] = {"range": [0, height], "autorange": False}
fig_dict["layout"]["updatemenus"] = [
    {
        "buttons": [
            {
                "label": "play",
                "method": "animate",
                "args": [None]
            },
            {
                "label": "Pause",
                "method": "animate",
                "args": [
                    [None],
                    {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }
                ]
            }
        ]
    }
]

sliders_dict = {
    "active": 0,
    "yanchor": "top",
    "xanchor": "left",
    "currentvalue": {
        "font": {"size": 20},
        "prefix": "Date:",
        "visible": True,
        "xanchor": "right"
    },
    "transition": {"duration": 1000, "easing": "cubic-in-out"},
    "pad": {"b": 10, "t": 100},
    "len": 0.9,
    "x": 0.1,
    "y": 0,
    "steps": []
}

data_dict = {
    "x": coords_map[unique_times_sorted[0]]["x"],
    "y": coords_map[unique_times_sorted[0]]["y"],
    "mode": "markers",
    "marker": {
        "size": 5,
        "color": "red"
    }
}

fig_dict["data"].append(data_dict)

print(unique_times_sorted)

start_time = datetime(2011, 5, 18, 0, 0)

unique_times_sorted = unique_times_sorted[unique_times_sorted.index(start_time):]

print(unique_times_sorted)

for time in unique_times_sorted:
    name = time.strftime('%m/%d/%Y %H:%M')
    frame = {"data": [], "name": name}
    data_dict = {
        "x": coords_map[time]["x"],
        "y": coords_map[time]["y"],
        "mode": "markers",
        "marker": {
            "size": 5,
            "color": "red"
        }
    }
    frame["data"].append(data_dict)

    fig_dict["frames"].append(frame)
    slider_step = {"args": [
        [name],
        {"frame": {"duration": 1000, "redraw": False},
         "mode": "immediate",
         "transition": {"duration": 1000}}
    ],
        "label": name,
        "method": "animate"}
    sliders_dict["steps"].append(slider_step)

fig_dict["layout"]["sliders"] = [sliders_dict]

fig2 = go.Figure(fig_dict)

map_plot = base64.b64encode(open(image_filename, 'rb').read())

fig2.update_layout(
                title = "Animation of message locations for each hour",
                images = [dict(
                    source='data:image/png;base64,{}'.format(map_plot.decode()),
                    xref="paper", yref="paper",
                    x=0, y=0,
                    sizex=1, sizey=1,
                    xanchor="left",
                    yanchor="bottom",
                    sizing="fill",
                    layer="below")])

fig2.show()

