import logging
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import tempfile

import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        #corpus_path = datapath('lee_background.cor')
        with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
            reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["text"])
            converted = reader.text.to_list()
            print(type(converted))
            for line in converted:
                #print(line)
                # assume there's one document per line, tokens separated by whitespace
                yield utils.simple_preprocess(line)
"""
wv = api.load('word2vec-google-news-300')

for index, word in enumerate(wv.index_to_key):
    if index == 10:
        break
    print(f"word #{index}/{len(wv.index_to_key)} is {word}")

pairs = [
('car', 'minivan'),   # a minivan is a kind of car
('car', 'bicycle'),   # still a wheeled vehicle
('car', 'airplane'),  # ok, no wheels, but still a vehicle
('car', 'cereal'),    # ... and so on
('car', 'communism'),
]
for w1, w2 in pairs:
    print('%r\t%r\t%.2f' % (w1, w2, wv.similarity(w1, w2)))
"""

sentences = MyCorpus()
#sentences.dropna(inplace=True)
model = gensim.models.Word2Vec(sentences=sentences)

wv = model.wv

with open("words.txt", "w") as file:
    for index, word in enumerate(wv.index_to_key):
        #if index == 10:
        #    break
        print(f"word #{index}/{len(wv.index_to_key)} is {word}")
        file.write(f"word #{index}/{len(wv.index_to_key)} is {word}\n")

with tempfile.NamedTemporaryFile(prefix='gensim-model-', delete=False) as tmp:
    temporary_filepath = tmp.name
    split = temporary_filepath.split('/')
    temporary_filepath = split[-1]
    model.save(temporary_filepath)
    with open("model_name.txt", "w") as file:
        file.write(temporary_filepath)
    #
    # The model is now safely stored in the filepath.
    # You can copy it to other machines, share it with others, etc.