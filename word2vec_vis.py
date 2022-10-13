import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
from skimage import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as py
import re
from datetime import datetime
from PIL import Image

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
            # Add leading zero to day and month, makes sorting easier
            dates.append(datetime.strptime(line, '%m/%d/%Y').strftime('%m/%d/%Y'))
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
print(type(dates[0]))

sorted = sorted(dates, key=lambda date: datetime.strptime(date, "%m/%d/%Y"))

# Create a subplot for each day's comments
fig = make_subplots(rows=7, cols=3)

row = 0
col = 0

for date in sorted:
    index = dates.index(date)
    x = x_arr[index]
    y = y_arr[index]
    fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode="markers", 
                    name=date,
                    marker=dict(
                        size=2
                    )), 
                    row=row+1, col=col+1)
    # Move through columns (x) first, once we reach the final, move to another row
    col = (col + 1) % 3
    if col == 0:
        row = (row + 1) % 7

fig.show()


many_comments = ["05/18/2011", "05/19/2011", "05/20/2011"]
img = io.imread('MC_1_Materials_3-30-2011/Vastopolis_Map.png')

for date in many_comments:
    fig2 = px.imshow(img)
    index = dates.index(date)
    fig2.add_trace(
        go.Scatter(
            x=x_arr[index],
            y=y_arr[index],
            mode="markers",
            marker=dict(
                size=3
            )
        )
    )
    fig2.update_layout(title=f"Comments for date: {date}")
    
    fig2.show()

#fig.show()
'''
fig = px.imshow(img)

fig.add_trace(
    go.Scatter(
            x=x_arr[0],
            y=y_arr[0],
            mode="markers",
            marker=dict(
                size=2
            ))
)
fig.show()
'''