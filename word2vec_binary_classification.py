import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB

categories = ['alt.atheism', 'soc.religion.christian',
               'comp.graphics', 'sci.med']

print("asd")

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',
     categories=categories, shuffle=True, random_state=42)

print("asd")

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

'''
dataset = pd.read_csv("Binary_classification/Training_data/training_data_pre_ob.csv", sep=",", header=0, usecols=["ID", "Created_at", "Location", "text", "label"])

X = dataset['text']
y = dataset['label']

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X)
X_train_counts.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = GaussianNB().fit(X_train_tfidf, y)
'''
