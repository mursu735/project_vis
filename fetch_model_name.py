import os

def fetch_model_name():
    with open("model_name.txt") as file:
        return file.read()