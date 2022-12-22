import os

def fetch_model_name():
    with open("model_name.txt") as file:
        return file.read()

def get_start_of_text():
    return "!!!!!!!!!!!!START HERE!!!!!!!!!!!!!!!!!!\n"

def get_end_of_text():
    return "!!!!!!!!!!!!END HERE!!!!!!!!!!!!!!!!!!"
