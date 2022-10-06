import numpy as np
from matplotlib import pyplot as plt

north_end = 42.3017
west_start = 93.5673
north_start = 42.1609
west_end = 93.1923
print(type(north_start))

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