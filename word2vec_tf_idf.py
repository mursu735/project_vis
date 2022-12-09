import word2vec_helpers
import numpy as np
import plotly.express as px
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from pathlib import Path  
import glob

def convert_to_date(date):
    split = date.split(" ")
    return split[0].replace("/", "-")

'''
reader = pd.read_csv('MC_1_Materials_3-30-2011/Microblogs.csv', sep=",", header=0, usecols=["Created_at", "text", "Location"])

reader['Created_at'] = reader['Created_at'].apply(convert_to_date)

df = reader.groupby("Created_at")['text'].apply(list)

for row in df.keys():
    with open(f"tf_idf/{row}.txt", "w", encoding="utf-8") as file:
        text = ' '.join(df[row])
        file.write(text)
'''

directory_path = "./tf_idf/"
text_files = glob.glob(f"{directory_path}/*.txt")
#print(text_files)
text_titles = [Path(text).stem for text in text_files]

#print(text_titles)

tfidf_vectorizer = TfidfVectorizer(input='filename', stop_words='english')

tfidf_vector = tfidf_vectorizer.fit_transform(text_files)

tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=text_titles, columns=tfidf_vectorizer.get_feature_names())

#tfidf_df.loc['00_Document Frequency'] = (tfidf_df > 0).sum()
tfidf_df = tfidf_df.sort_index().round(decimals=2)

tfidf_df = tfidf_df.stack().reset_index()

tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'level_0': 'document','level_1': 'term', 'level_2': 'term'})

tfidf_df = tfidf_df.sort_values(by=['document','tfidf'], ascending=[True,False]).groupby(['document']).head(100)

dates = ['5-16-2011', '5-17-2011', '5-18-2011']
print("Writing files")
for date in dates:
    terms = tfidf_df[tfidf_df['document'].str.contains(date)]
    terms.to_csv(f"tf_idf/result_{date}.csv")


#print(tfidf_df)