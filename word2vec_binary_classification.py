from datetime import datetime
import word2vec_helpers
import word2vec_helpers_graph
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB

'''
categories = ['alt.atheism', 'soc.religion.christian',
               'comp.graphics', 'sci.med']

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',
     categories=categories, shuffle=True, random_state=42)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

print(type(X_train_tfidf))

clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

print(type(twenty_train.target))
'''

def train_model_and_classify(training_file, classification_file, label):
    dataset = pd.read_csv(training_file, sep=",", header=0, usecols=["ID", "Created_at", "Location", "text", "label"])

    X = dataset['text'].to_numpy()
    y = dataset['label'].to_numpy()

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    print("Training classifier")

    clf = MultinomialNB().fit(X_train_tfidf, y)

    reader = pd.read_csv(classification_file, sep=",", header=0, usecols=["ID", "Created_at", "Location", "text"])
    docs_new = reader["text"].to_numpy()

    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    print("Creating predictions")

    predicted = clf.predict(X_new_tfidf)

    coords_map = {}

    unique_times = []

    sickness_messages = 0
    other_messages = 0
    symptom1 = word2vec_helpers.get_disease_1_symptoms()
    symptom2 = word2vec_helpers.get_disease_2_symptoms()

    for text, category, date, location in zip(docs_new, predicted, reader["Created_at"], reader["Location"]):
        if category == 1:
            sickness_messages += 1
            try:
                time = datetime.strptime(date, '%m/%d/%Y %H:%M').replace(minute=0)
                if time not in coords_map:
                    coords_map[time] = {"x": [], "y": [], "text": [], "label": []}
                    unique_times.append(time)
                x, y = word2vec_helpers.get_coords_in_pixels(location)
                coords_map[time]["x"].append(x)
                coords_map[time]["y"].append(y)
                coords_map[time]["text"].append(text)
                if any(symptom in text for symptom in symptom1):
                    coords_map[time]["label"].append("red")
                elif any(symptom in text for symptom in symptom2):
                    coords_map[time]["label"].append("green")
                else:
                    coords_map[time]["label"].append("black")
            except ValueError:
                print(f"Invalid date: {date}")
        else:
            other_messages += 1

    print(f"Number of messages classified as sickness-related: {sickness_messages}")
    print(f"Number of messages classified as non-sickness-related: {other_messages}")

    unique_times_sorted = sorted(unique_times)

    word2vec_helpers_graph.plot_timelapse_graph(coords_map, unique_times_sorted, label)


train_model_and_classify("Binary_classification/Training_data/training_data_pre_ob.csv", "Binary_classification/Training_data/other_data_pre_ob.csv", "Animation of sickness-related messages pre-outbreak, binary classifier")
train_model_and_classify("Binary_classification/Training_data/training_data_post_ob.csv", "Binary_classification/Training_data/other_data_post_ob.csv", "Animation of sickness-related messages post-outbreak, binary classifier")