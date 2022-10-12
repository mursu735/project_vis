import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from skimage import io
import plotly.graph_objects as go
import re
import datetime

north_end = 42.3017
west_start = 93.5673
north_start = 42.1609
west_end = 93.1923

width = 5216
height = 2653

"""
#plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
im = plt.imread("MC_1_Materials_3-30-2011/Vastopolis_Map.png")
fig, ax = plt.subplots()
im = ax.imshow(im, extent=[0, 5216, 0, 2653])
x_arr = []
y_arr = []
with open("filtered_coords.txt") as file:
    for line in file.readlines():
        line = line.replace("\n", "")
        coords = line.split(" ")
        x = float(coords[0])
        y = float(coords[1])
        x_interpolate = ((x - north_start) / (north_end - north_start)) * 5216
        y_interpolate = ((y - west_start) / (west_end - west_start)) * 2653
        x_arr.append(x_interpolate)
        y_arr.append(y_interpolate)
        
ax.scatter(x_arr, y_arr, color='red', s=0.2)
plt.show()
"""

fig = go.Figure()

x_arr = []
y_arr = []
dates = []
counter = -1
pattern = re.compile("^[4-5]\/[0-3]?[0-9]\/2011$")
with open("filtered_coords.txt") as file:
    for line in file.readlines():
        line = line.replace("\n", "")
        #If we reach a new day, create a new entry to dict
        if pattern.match(line):
            dates.append(line)
            counter += 1
            x_arr.append([])
            y_arr.append([])
            continue
        coords = line.split(" ")
        x = float(coords[0])
        y = float(coords[1])
        x_interpolate = ((x - north_start) / (north_end - north_start)) * width
        y_interpolate = ((y - west_start) / (west_end - west_start)) * height
        x_arr[counter].append(x_interpolate)
        y_arr[counter].append(y_interpolate)

print(dates)

sorted = dates.sort(key=lambda date: datetime.datetime.strptime(date, "%#m/%#d/%Y"))

print(sorted)

#Sort times, get the original index, visualize each day separately

img = io.imread('MC_1_Materials_3-30-2011/Vastopolis_Map.png')
fig = px.imshow(img)
"""
fig.add_trace(
    go.Scatter(x=x_arr, y=y_arr, mode="markers")
)
"""

fig.show()