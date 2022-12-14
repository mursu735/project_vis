import os
import re
import numpy as np
import pandas as pd
from nltk.corpus import wordnet


north_end = 42.3017
west_start = 93.5673
north_start = 42.1609
west_end = 93.1923

def fetch_model_name():
    with open("model_name.txt") as file:
        return file.read()

def fetch_model_name_pre_ob():
    with open("model_name_pre_ob.txt") as file:
        return file.read()

def fetch_model_name_post_ob():
    with open("model_name_post_ob.txt") as file:
        return file.read()

def get_pre_ob_regex():
    return re.compile("^[4-5]\/((3[0-1]?)|([0-9]?)|(1[0-6]))\/2011")

def get_post_ob_regex():
    return re.compile("^5\/((1[7-9])|(2[0-9]))\/2011")

def get_word_list():
    #return ["fever", "chills", "sweats", "aches", "pains", "fatigue", "coughing", "breathing", "nausea", "vomit", "diarrhea"]
    # return ["sick", "sleepy", "uncomfortable", "dizzy", "nauseous", "unwell", "bedridden", "coughing", "fever", "hospitalized", "headache", "rashes"]
    # return ["fever", "headache", "pneumonia", "sweats", "fatigue", "flu", "chills", "heartburn", "nausea", "cramps", "cold", "cough", "aching", "breath", "diarrhea", "insomnia", "unwell", "vomit"]
    return ["fever", "headache", "pneumonia", "sweats", "fatigue", "flu", "chills", "heartburn",
            "nausea", "cramps", "cold", "cough", "aching", "breathe", "diarrhea", "insomnia", "unwell",
            "vomit", "sick", "ache", "bomb", "explosion", "threat", "spill", "fire"]

def get_disease_1_symptoms():
    return ["flu", "sweats", "chill", "pneumonia", "fatigue", "headache", "cold", "fever"]

def get_disease_2_symptoms():
    return ["diarrhea", "nausea", "heartburn", "cramps"]
    #return ["diarrhea", "nausea", "heartburn", "cramps", "aching", "ache", "breath", "sick", "vomit", "insomnia", "unwell", "cough"]

def get_causes():
    return ["bomb", "explosion", "threat", "spill", "fire"]

def get_blacklist():
    return ["fried", "oil"]

def get_ambiguous_words():
    return ["sick", "chills", "cold"]

def get_width():
    return 5216

def get_height():
    return 2653

def get_district_colors():
    return np.array(([255, 242, 0], [14, 206, 69], [255, 202, 24], [63, 72, 204], [184, 61, 186], [140, 255, 251], [236, 28, 36], [255, 13, 182], [136, 0, 27], [253, 236, 166], [88, 88, 88], [129, 39, 255], [255, 127, 39]))

def get_image_name():
    return "MC_1_Materials_3-30-2011/Vastopolis_Map.png"

def get_coords_in_pixels(coord):
    coords = coord.split(" ")
    width = get_width()
    height = get_height()
    n = float(coords[0])
    w = float(coords[1])
    y_interpolate = (((n - north_start) / (north_end - north_start))) * height
    x_interpolate = (((w - west_start) / (west_end - west_start))) * width
    return x_interpolate, y_interpolate

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

def check_if_separator_is_used(separator):
    reader = pd.read_csv('MC_1_Materials_3-30-2011/Microblogs.csv', sep=",", header=0, usecols=["Created_at", "text", "Location"])
    count = 0
    print(reader)
    for index, row in reader.iterrows():
        if separator in row["text"]:
            text = row["text"]
            print(f"Row {text} contains separator {separator}")
            count += 1
    if count == 0:
        print("Row did not contain separator")

def determine_message_location(image, message):
    # Each row correponds to RGB color on edited map, used to group the symptoms
    colors = get_district_colors()
    districts = ["Cornertown", "Northville", "Villa", "Westside", "Smogtown", "Plainville", "Downtown", "Uptown", "Riverside", "Southville", "Lakeside", "Eastside", "Suburbia"]