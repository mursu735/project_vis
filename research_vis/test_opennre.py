import helpers
#import gensim.models
#import opennre
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from PIL import Image

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
fig = go.Figure()
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
fig.update_layout()

fig.show()


