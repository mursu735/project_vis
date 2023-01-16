import os
import glob
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

def read_corpus(names, tokens_only=False):
    for file in names:
        with open(file, encoding="utf-8") as f:
            for i, line in enumerate(f):
                tokens = gensim.utils.simple_preprocess(line)
                if tokens_only:
                    yield tokens
                else:
                    # For training data, add tags
                    split = file.split("CHAPTER")
                    chapter = ""
                    if len(split) == 1:
                        chapter = "Epilogue"
                    else:
                        number = split[1].strip().split(".")[0]
                        chapter = "Chapter " + number
                    tag = chapter + "/paragraph " + str(i+1)
                    yield gensim.models.doc2vec.TaggedDocument(tokens, [tag])

directory_path = "./Chapters/"
if not os.path.exists(directory_path):
    print("Chapters do not exists, unable to train model")


text_files = glob.glob(f"{directory_path}/*.txt")
text_files = [file for file in text_files if "CHAPTER" in file or "Epilogue" in file]

train_corpus = list(read_corpus(text_files))

tags = [file[1][0] for file in train_corpus]

model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=10, epochs=40)

model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)


with open("doc2vec_tags.txt", "w") as file:
    for tag in tags:
        file.write(f"{tag}\n")

with tempfile.NamedTemporaryFile(prefix='doc2vec-model-', delete=False) as tmp:
    temporary_filepath = tmp.name
    split = temporary_filepath.split('/')
    # Save in local directory for Windows
    if len(split) == 1:
        split = temporary_filepath.split('\\')
    temporary_filepath = split[-1]
    model.save(temporary_filepath)
    with open("model_name_doc2vec.txt", "w") as file:
        file.write(temporary_filepath)
    #
    # The model is now safely stored in the filepath.
    # You can copy it to other machines, share it with others, etc.