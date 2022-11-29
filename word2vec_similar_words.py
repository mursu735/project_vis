import logging
import word2vec_helpers
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import plotly.express as px
from skimage import io
import plotly.graph_objects as go
import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

model_name = word2vec_helpers.fetch_model_name()

print(model_name)

new_model = gensim.models.Word2Vec.load(model_name)
wv = new_model.wv

#wv = api.load('glove-twitter-100')

word = "fever"

#wv.evaluate_word_pairs('MC_1_Materials_3-30-2011/Microblogs.csv')

words = wv.most_similar(positive=[f"{word}"], topn=100)

with open(f"similar_own_model_{word}.txt", "w") as f:
    for asd in words:
        f.write(f"{asd}\n")

# sick -> fever
# fever -> headache, pneumonia, sweats, fatigue, flu, chills, heartburn, nausea, cramps, cold, cough, aching, breath, diarrhea, insomnia, unwell, vomitting