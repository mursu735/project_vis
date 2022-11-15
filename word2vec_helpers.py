import os
import re

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
    return ["fever", "headache", "pneumonia", "bedridden", "sweats", "fatigue", "flu", "chills", "heartburn", "nausea", "cramps", "cold", "cough", "aching", "breath", "diarrhea", "insomnia", "unwell", "vomit"]

def get_disease_1_symptoms():
    return ["flu", "sweats", "chills", "pneumonia", "fatigue", "headache"]

def get_disease_2_symptoms():
 return []
