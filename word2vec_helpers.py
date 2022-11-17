import os
import re

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
    return ["fever", "headache", "pneumonia", "sweats", "fatigue", "flu", "chills", "heartburn", "nausea", "cramps", "cold", "cough", "aching", "breath", "diarrhea", "insomnia", "unwell", "vomit"]

def get_disease_1_symptoms():
    return ["flu", "sweats", "chills", "pneumonia", "fatigue", "headache", "cold", "fever"]

def get_disease_2_symptoms():
    return ["diarrhea", "nausea", "heartburn", "cramps", "vomit"]

def get_width():
    return 5216

def get_height():
    return 2653

def get_image_name():
    return "MC_1_Materials_3-30-2011/Vastopolis_Map.png"

def get_coords_in_pixels(coord):
    coords = coord.split(" ")
    width = get_width()
    height = get_height()
    x = float(coords[0])
    y = float(coords[1])
    x_interpolate = ((x - north_start) / (north_end - north_start)) * width
    y_interpolate = ((y - west_start) / (west_end - west_start)) * height
    return x_interpolate, y_interpolate