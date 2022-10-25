import os

def fetch_model_name():
    with open("model_name.txt") as file:
        return file.read()

def get_word_list():
    #return ["fever", "chills", "sweats", "aches", "pains", "fatigue", "coughing", "breathing", "nausea", "vomiting", "diarrhoea"]
    return ["sick", "sleepy", "uncomfortable", "dizzy", "nauseous", "unwell", "bedridden", "coughing", "fever", "hospitalized", "headache", "rashes"]
    # ["fever", "headache", "pneumonia", "sweats", "fatigue", "flu", "chills", "heartburn", "nausea", "cramps", "cold", "cough", "aching", "breath", "diarrhea", "insomnia", "unwell", "vomitting"]
