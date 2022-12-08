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
import nltk
from nltk.stem import WordNetLemmatizer


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

model_name = word2vec_helpers.fetch_model_name()

print(model_name)

new_model = gensim.models.Word2Vec.load(model_name)
wv = new_model.wv

#wv = api.load('glove-twitter-100')

'''word = "dead"

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

sentence = "aching aches wed couple"
word_list = nltk.word_tokenize(sentence)
lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in word_list])
print(lemmatized_output)

print("corpora :", lemmatizer.lemmatize("corpora"))

# a denotes adjective in "pos"
print("better :", lemmatizer.lemmatize("better", pos ="a"))
'''

vocab = wv.index_to_key
count = 0
total = len(vocab)
word_list = {}
#print(vocab)


with open("similar_words/word_list.txt", "w") as file:
    for word in vocab:
        print(f"{count}/{total}", end='\r')
        file.write(f"{word}\n")
        word_list[word] = wv.most_similar(positive=[f"{word}"], topn=100)
        count += 1

print("writing similar vectors")
count = 0
with open("similar_words/similar_words.txt", "w") as file:
    for k, v in word_list.items():
        print(f"{count}/{total}", end='\r')
        file.write(f"{k}: {v}\n")
        count += 1

#wv.evaluate_word_pairs('MC_1_Materials_3-30-2011/Microblogs.csv')
'''
words = wv.most_similar(positive=[f"{word}"], topn=100)

with open(f"similar_own_model_{word}.txt", "w") as f:
    for asd in words:
        print(asd)
        #f.write(f"{asd}\n")
'''
# sick -> fever
# fever -> headache, pneumonia, sweats, fatigue, flu, chills, heartburn, nausea, cramps, cold, cough, aching, breath, diarrhea, insomnia, unwell, vomitting