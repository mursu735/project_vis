import helpers
#import gensim.models
#import opennre
import pandas as pd
import math
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from PIL import Image
import numpy as np

def get_color(colorscale_name, loc):
    from _plotly_utils.basevalidators import ColorscaleValidator
    # first parameter: Name of the property being validated
    # second parameter: a string, doesn't really matter in our use case
    cv = ColorscaleValidator("colorscale", "")
    # colorscale will be a list of lists: [[loc1, "rgb1"], [loc2, "rgb2"], ...]
    colorscale = cv.validate_coerce(colorscale_name)

    if hasattr(loc, "__iter__"):
        return [get_continuous_color(colorscale, x) for x in loc]
    return get_continuous_color(colorscale, loc)


# Identical to Adam's answer
import plotly.colors
from PIL import ImageColor

def get_continuous_color(colorscale, intermed):
    """
    Plotly continuous colorscales assign colors to the range [0, 1]. This function computes the intermediate
    color for any value in that range.

    Plotly doesn't make the colorscales directly accessible in a common format.
    Some are ready to use:

        colorscale = plotly.colors.PLOTLY_SCALES["Greens"]

    Others are just swatches that need to be constructed into a colorscale:

        viridis_colors, scale = plotly.colors.convert_colors_to_same_type(plotly.colors.sequential.Viridis)
        colorscale = plotly.colors.make_colorscale(viridis_colors, scale=scale)

    :param colorscale: A plotly continuous colorscale defined with RGB string colors.
    :param intermed: value in the range [0, 1]
    :return: color in rgb string format
    :rtype: str
    """
    if len(colorscale) < 1:
        raise ValueError("colorscale must have at least one color")

    hex_to_rgb = lambda c: "rgb" + str(ImageColor.getcolor(c, "RGB"))

    if intermed <= 0 or len(colorscale) == 1:
        c = colorscale[0][1]
        return c if c[0] != "#" else hex_to_rgb(c)
    if intermed >= 1:
        c = colorscale[-1][1]
        return c if c[0] != "#" else hex_to_rgb(c)

    for cutoff, color in colorscale:
        if intermed > cutoff:
            low_cutoff, low_color = cutoff, color
        else:
            high_cutoff, high_color = cutoff, color
            break

    if (low_color[0] == "#") or (high_color[0] == "#"):
        # some color scale names (such as cividis) returns:
        # [[loc1, "hex1"], [loc2, "hex2"], ...]
        low_color = hex_to_rgb(low_color)
        high_color = hex_to_rgb(high_color)

    return plotly.colors.find_intermediate_color(
        lowcolor=low_color,
        highcolor=high_color,
        intermed=((intermed - low_cutoff) / (high_cutoff - low_cutoff)),
        colortype="rgb",
    )

'''
model_name = helpers.fetch_doc2vec_model_name()

new_model = gensim.models.Doc2Vec.load(f"{model_name}")

#print(new_model.dv['Chapter 1/paragraph 1'])

print(new_model.dv.get_vector('Chapter 1/paragraph 1'))

tags = []

with open("doc2vec_tags.txt") as file:
    tags = [line.replace('\n','') for line in file.readlines()]

print(tags)'''
'''
model = opennre.get_model('wiki80_bert_softmax')
text = "But I had not proceeded far, when I began to bethink me that the Captain with whom I was to sail yet remained unseen by me; though, indeed, in many cases, a whale-ship will be completely fitted out, and receive all her crew on board, ere the captain makes himself visible by arriving to take command; for sometimes these voyages are so prolonged, and the shore intervals at home so exceedingly brief, that if the captain have a family, or any absorbing concernment of that sort, he does not trouble himself much about his ship in port, but leaves her to the owners till all is ready for sea. However, it is always as well to have a look at him before irrevocably committing yourself into his hands. Turning back I accosted Captain Peleg, inquiring where Captain Ahab was to be found. "
target2 = "Ahab"
target1 = "Peleg"


start_h = text.index(target1)
end_h = start_h + len(target1)

start_t = text.index(target2)
end_t = start_t + len(target2)

print(start_h, end_h)

print(start_t, end_t)

asd = model.infer({'text': text, 
    'h': {'pos': (start_h, end_h)}, 't': {'pos': (start_t, end_t)}})
print(asd)
'''
#asd = pd.read_json("test.json", convert_dates=False)
#print(asd.keys())

asd = pd.read_csv("./tf_idf/result_CHAPTER 1.csv", sep=",", header=0, usecols=["term", "tfidf"]).set_index("term").to_dict()

#print(asd["tfidf"])

#wordcloud = WordCloud(background_color="white", width=1200, height=900).generate_from_frequencies(asd["tfidf"])
im = Image.open("banner.png") # Can be many different formats.
pix = im.load()

#fig = px.imshow(pix) # image show
'''fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[0],
    y=[0],
    mode="markers",
    marker=dict(opacity=0)))

fig.add_layout_image(
        dict(
            source=im,
            xref="x",
            yref="y",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing="fill",
            layer="above")
)

fig.update_xaxes(range=[0, 1], visible=False, showticklabels=False)
fig.update_yaxes(range=[0, 1], visible=False, showticklabels=False)
fig.update_layout()'''
'''
chapter_list = ["CHAPTER 2", "CHAPTER 3"]#, "CHAPTER 4", "CHAPTER 5", "CHAPTER 6"]
wordcloud_list = []

no_cols = 4
rows = math.ceil(len(chapter_list) / 2)
fig = make_subplots(
        rows=rows,
        cols=no_cols,
        vertical_spacing=0.02,
        subplot_titles=[f"Wordcloud for {chapter}" for chapter in chapter_list])

row = 1
col = 0
for chapter in chapter_list:
    number = chapter.split(" ")
    if len(number) > 1:
        chapter = f"CHAPTER {number[1]}"
    tfidf = pd.read_csv(f"./tf_idf/result_{chapter}.csv", sep=",", header=0, usecols=["term", "tfidf"]).set_index("term").to_dict()
    wordcloud = WordCloud(background_color="white", width=800, height=600).generate_from_frequencies(tfidf["tfidf"])
    fig_img = px.imshow(wordcloud)
    fig.add_trace(fig_img.data[0], row=row, col=col+1)
    col = (col + 1) % 4
    if col == 0:
        row += 1


#fig = px.imshow(wordcloud_list[0], facet_col=0, facet_col_wrap=5)
#fig.update_layout(title=f"Word cloud for {chapter}")

fig.show()
'''

def rgb_to_hex(red, green, blue):
    """Return color as #rrggbb for the given color values."""
    return '#%02x%02x%02x' % (red, green, blue)

from plotly.express.colors import sample_colorscale, hex_to_rgb, label_rgb

x = np.linspace(0, 1, 136)
#c = sample_colorscale('rdbu', list(x))
c = get_color("rdbu", x)
asd = [col.lstrip('rbg(').rstrip(")").split(",") for col in c]
asd = [[int(float(j)) for j in i] for i in asd]
#print(asd)
asd = [rgb_to_hex(col[0], col[1], col[2]) for col in asd]
print(asd)
jkl = [label_rgb(hex_to_rgb(hex)) for hex in asd]
print(jkl)
fig = go.Figure(data=[
    go.Bar(name='Average Cars Per Area', x=x, y=x, marker_color=jkl)
])

fig.show()