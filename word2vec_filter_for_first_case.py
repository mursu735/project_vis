import logging
import word2vec_helpers
import numpy as np
import gensim.downloader as api
from gensim.test.utils import datapath
from gensim import utils
import gensim.models
import plotly.express as px
from datetime import datetime
import pandas as pd
from PIL import Image
import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

from pandarallel import pandarallel
import multiprocessing
import time

def check_if_separator_is_used(separator):
    with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
        reader = pd.read_csv(csvfile, sep=",", header=0, usecols=["Created_at", "text", "Location"])
        converted = reader.text.to_list()
        coords = []
        lines = []
        count = 0
        print(reader)
        for index, row in reader.iterrows():
            if separator in row["text"]:
                text = row["text"]
                print(f"Row {text} contains separator {separator}")
                count += 1
    if count == 0:
        print("Row did not contain separator")

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def select_rows(row, pix, sizex, sizey, start_time, end_time, target_area):
    #print(row)
    try:
        time = datetime.strptime(row.Created_at, '%m/%d/%Y %H:%M').replace(minute=0)
        if time >= start_time and time <= end_time:
            pos = row.Location
            x, y = word2vec_helpers.get_coords_in_pixels(pos)
            x = min(x, sizex - 1)
            y = min(y, sizey - 1)
            # PIL has origin in top-left, convert bottom-left origin to this, then fetch pixel color
            y_tl = word2vec_helpers.get_height() - y - 1
            color = np.array(pix[x, y_tl])
            asd = target_area - color[:3] # Some pixels return [r, g, b, alpha], get rid of alpha
            sum = asd.sum()
            #if sum == 0:
            return True
    except ValueError as e:
        text = row["text"]
        print(f"{e}, message: {text}")
        return False
    return False

def edit_lines(row, lemmatizer, symptom1, symptom2, other_symptoms, blacklist):
    import nltk
    from word2vec_helpers import nltk_tag_to_wordnet_tag

    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(row["text"]))
    separator = ':^:'
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    # "5/18/2011" in row["Created_at"] and
    concat = set(lemmatized_sentence)
    if concat.intersection(blacklist):
        prefix = "None"
    else:    
        if concat.intersection(symptom1):
            prefix = "Symptom 1"
        elif concat.intersection(symptom2):
            prefix = "Symptom 2"
        elif concat.intersection(other_symptoms):
            prefix = "Other"
        else:
            prefix = "None"
    #if sum == 0:
    return prefix + separator + row["Created_at"] + separator + row["Location"] + separator + row["text"] + separator + str(row["ID"])

def do_stuff():
    # If the position and date is in the area of interest, save it, otherwise skip it
    target_area = np.array([63, 72, 204])
    start_time = datetime(2011, 5, 16, 0, 0)
    end_time = datetime(2011, 5, 20, 0, 0)

    separator = ':^:'
    edited_image_filename = "MC_1_Materials_3-30-2011/Vastopolis_Map_edited_first_case_mod.png"
    im = Image.open(edited_image_filename) # Can be many different formats.
    pix = im.load()
    sizex, sizey = im.size

    symptom1 = set(word2vec_helpers.get_disease_1_symptoms())
    symptom2 = set(word2vec_helpers.get_disease_2_symptoms())
    blacklist = word2vec_helpers.get_blacklist()
    other_symptoms = []
    tmp = word2vec_helpers.get_word_list()

    for element in tmp:
        if element not in symptom1 and element not in symptom2:
            other_symptoms.append(element)

    print(other_symptoms)
    other_symptoms = set(other_symptoms)

    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    lemmatizer = WordNetLemmatizer()

    #with open('MC_1_Materials_3-30-2011/Microblogs.csv') as csvfile:
    reader = pd.read_csv('MC_1_Materials_3-30-2011/Microblogs.csv', sep=",", header=0, usecols=["ID", "Created_at", "text", "Location"])

    begin_time = time.time()
    pandarallel.initialize(progress_bar=True)


    reader['target_area'] = reader.apply(select_rows, args=(pix, sizex, sizey, start_time, end_time, target_area), axis=1)
    reader = reader[reader['target_area']]


    reader["text"] = reader.parallel_apply(edit_lines, args=(lemmatizer, symptom1, symptom2, other_symptoms, blacklist), axis=1)

    lines = reader["text"].tolist()
    '''
    total = len(reader.index)
    count = 0
    for index, row in reader.iterrows():
        try:
            print(f"{count}/{total}", end='\r')
            count += 1
            prefix = ""
            nltk_tagged = nltk.pos_tag(nltk.word_tokenize(row["text"]))
            #tuple of (token, wordnet_tag)
            wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
            lemmatized_sentence = []
            for word, tag in wordnet_tagged:
                if tag is None:
                    #if there is no available tag, append the token as is
                    lemmatized_sentence.append(word)
                else:
                    #else use the tag to lemmatize the token
                    lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
            concat = set(lemmatized_sentence)
            if concat.intersection(blacklist):
                prefix = "None"
            else:    
                if concat.intersection(symptom1):
                    prefix = "Symptom 1"
                elif concat.intersection(symptom2):
                    prefix = "Symptom 2"
                elif concat.intersection(causes):
                    prefix = "Causes"
                elif concat.intersection(other_symptoms):
                    prefix = "Other"
                else:
                    prefix = "None"
            #if sum == 0:
            #total += 1
            text = prefix + separator + row["Created_at"] + separator + row["Location"] + separator + row["text"] + separator + str(row["ID"])
            lines.append(text)
        except ValueError as e:
            text = row["text"]
            print(f"{e}, message: {text}")
    '''

    print(f"Time elapsed: {time.time() - begin_time}")
    
    with open("filtered_first_case.txt", "w", encoding="utf-8") as file:
        for line in lines:
            file.write(f"{line}\n")

    '''
    with open("filtered_coords.txt", "w") as file:
        for line in coords:
            file.write(f"{line}\n")
            for line in val:
                file.write(f"{line}\n")
    '''

    print(f"Remaining number of messages: {len(lines)}")

if __name__ == '__main__':
    do_stuff()