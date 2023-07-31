import pandas as pd
from wordcloud import (WordCloud, get_single_color_func)
import plotly.express as px
import numpy as np
import os
import re
import glob
from helpers import natural_sort, get_color, rgb_to_hex


class SimpleGroupedColorFunc(object):
    """Create a color function object which assigns EXACT colors
       to certain words based on the color to words mapping

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color
                              for (color, words) in color_to_words.items()
                              for word in words}

        self.default_color = default_color

    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)


class GroupedColorFunc(object):
    """Create a color function object which assigns DIFFERENT SHADES of
       specified colors to certain words based on the color to words mapping.

       Uses wordcloud.get_single_color_func

       Parameters
       ----------
       color_to_words : dict(str -> list(str))
         A dictionary that maps a color to the list of words.

       default_color : str
         Color that will be assigned to a word that's not a member
         of any value from color_to_words.
    """

    def __init__(self, color_to_words, default_color):
        self.color_func_to_words = [
            (get_single_color_func(color), set(words))
            for (color, words) in color_to_words.items()]

        self.default_color_func = get_single_color_func(default_color)

    def get_color_func(self, word):
        """Returns a single_color_func associated with the word"""
        try:
            color_func = next(
                color_func for (color_func, words) in self.color_func_to_words
                if word in words)
        except StopIteration:
            color_func = self.default_color_func

        return color_func

    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)

tfidf_files = natural_sort([f for f in glob.glob("./tf_idf/*.csv") if "CHAPTER" in f or "Epilogue" in f])

# Set word color based on tfidf
x = np.linspace(0, 1, 136)
c = get_color("rdbu", x)
asd = [col.lstrip('rbg(').rstrip(")").split(",") for col in c]
asd = [[int(float(j)) for j in i] for i in asd]
hex = [rgb_to_hex(col[0], col[1], col[2]) for col in asd]

word_colors = {}

for chapter in hex:
    word_colors[chapter] = []

#print(word_colors)

highest_tfidf_for_word = pd.read_csv("./tf_idf/result_total.csv", sep=",", header=0, usecols=["term", "tfidf", "chapter"])#.set_index("term").to_dict()

for index, row in highest_tfidf_for_word.iterrows():
    if not row["chapter"] == "filler_text":
        filename = "./tf_idf/result_" + row["chapter"] + ".csv"
        index = tfidf_files.index(filename)
        color = hex[index]
        word_colors[color].append(row["term"])

# Calculate wordclouds
for chapter in tfidf_files:
    tfidf = pd.read_csv(chapter, sep=",", header=0, usecols=["term", "tfidf"]).set_index("term").to_dict()
    wordcloud = WordCloud(background_color="white", width=1200, height=900, max_words=20).generate_from_frequencies(tfidf["tfidf"])
    # Create a color function with multiple tones
    grouped_color_func = GroupedColorFunc(word_colors, "grey")

    # Apply our color function
    wordcloud.recolor(color_func=grouped_color_func)

    fig = px.imshow(wordcloud)
    #fig.update_layout(title=f"Word cloud for {chapter}", title_x=0.5)
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(visible=False, showticklabels=False)
    if not os.path.exists("./wordcloud_images"):
        os.mkdir("wordcloud_images")
    filename = chapter.split("result_")[1].split(".")[0]
    print(filename)
    fig.write_image(f"wordcloud_images/{filename}.png")
