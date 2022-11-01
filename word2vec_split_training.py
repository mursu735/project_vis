import logging
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import tempfile
from datetime import datetime
import re

import pandas as pd

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)


class MyCorpus(object):
    """An iterator that yields sentences (lists of str)."""
    def __init__(self, regexp):
        self.regexp = regexp

    def __iter__(self):
        #corpus_path = datapath('lee_background.cor')
        with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
            reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["text", "Created_at"])
            for index, row in reader.iterrows():
            #print(line)
            # assume there's one document per line, tokens separated by whitespace
                if (re.match(self.regexp, row["Created_at"])):
                    yield utils.simple_preprocess(row['text'])

pre_outbreak = re.compile("^[4-5]\/((3[0-1]?)|([0-9]?)|(1[0-6]))\/2011")
post_outbreak = re.compile("^5\/((1[7-9])|(2[0-9]))\/2011")


sentences = MyCorpus(pre_outbreak)
#sentences.dropna(inplace=True)
model = gensim.models.Word2Vec(sentences=sentences)

wv = model.wv

'''
for index, word in enumerate(wv.index_to_key):
    #if index == 10:
    #    break
    print(f"word #{index}/{len(wv.index_to_key)} is {word}")
'''

with tempfile.NamedTemporaryFile(prefix='gensim-model-pre-outbreak-', delete=False) as tmp:
    temporary_filepath = tmp.name
    split = temporary_filepath.split('/')
    temporary_filepath = split[-1]
    model.save(temporary_filepath)
    with open("model_name_pre_ob.txt", "w") as file:
        file.write(temporary_filepath)
    #
    # The model is now safely stored in the filepath.
    # You can copy it to other machines, share it with others, etc.

outbreak_sentences = MyCorpus(post_outbreak)

model = gensim.models.Word2Vec(sentences=outbreak_sentences)

wv = model.wv

with tempfile.NamedTemporaryFile(prefix='gensim-model-post-outbreak-', delete=False) as tmp:
    temporary_filepath = tmp.name
    split = temporary_filepath.split('/')
    temporary_filepath = split[-1]
    model.save(temporary_filepath)
    with open("model_name_post_ob.txt", "w") as file:
        file.write(temporary_filepath)