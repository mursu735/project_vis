import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import plotly.express as px
from skimage import io
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import base64
from PIL import Image
from datetime import datetime
import word2vec_helpers
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

lines = []

text_file = "filtered_wordcloud.txt"

with open(text_file) as file:
    for line in file.readlines():
        line = line.replace("\n", "")
        lines.append(line)

text = " ".join(line for line in lines)

wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", width=1200, height=900).generate(text)

fig = px.imshow(wordcloud) # image show

fig.show()
