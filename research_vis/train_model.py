import logging
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import tempfile
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import remove_stopwords

import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

class MyCorpus:
    """An iterator that yields sentences (lists of str)."""

    def __iter__(self):
        #corpus_path = datapath('lee_background.cor')
        with open('Moby_Dick.txt') as file:
            for line in file:
                yield(simple_preprocess(line))

sentences = MyCorpus()
#sentences.dropna(inplace=True)
model = gensim.models.Word2Vec(sentences=sentences)


wv = model.wv

with open("words.txt", "w") as file:
    for index, word in enumerate(wv.index_to_key):
        #if index == 10:
        #    break
        #print(f"word #{index}/{len(wv.index_to_key)} is {word}")
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
