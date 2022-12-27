import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import helpers
import re
import os
import glob
import nltk
from nltk import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from pathlib import Path



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

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()


tfidf_vectorizer = TfidfVectorizer(input='filename', stop_words='english')

def separate_text_to_chapters():
    start = helpers.get_start_of_text()
    text = []
    text_file = "Moby_Dick.txt"


    with open(text_file, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        line_no = 0
        #print(lines)
        for line in lines:
            if line != start:
                line_no += 1
            else:
                break
        text = lines[line_no+1:]

    chapter = ""
    current = ""

    chapter_map = {}
    #print(text)
    # If endline, add text line to chapter dict
    for line in text:
        # Chapter changes
        if (re.match("(^CHAPTER)|(^Epilogue)", line)):
            line = line.replace("\n", "")
            number = line.split(".")
            chapter = number[0]
            chapter_map[chapter] = []
        else:
            if line == "\n":
                if current != "":
                    chapter_map[chapter].append(current)
                current = ""
            else:
                current += line.replace("\n", " ")

    for chapter in chapter_map:
        with open(f"Chapters/{chapter}.txt", 'w', encoding="utf-8") as file:
            for paragraph in chapter_map[chapter]:
                file.write(paragraph)
                file.write("\n")



directory_path = "./Chapters/"
if not os.path.exists(directory_path):
    os.mkdir(directory_path)

separate_text_to_chapters()

text_files = glob.glob(f"{directory_path}/*.txt")
print(text_files)
text_titles = [Path(text).stem for text in text_files]
#nltk_tagged = nltk.pos_tag(nltk.word_tokenize(text))
#print(nltk_tagged)

tfidf_vector = tfidf_vectorizer.fit_transform(text_files)

tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index=text_titles, columns=tfidf_vectorizer.get_feature_names())

tfidf_df = tfidf_df.sort_index().round(decimals=2)

tfidf_df = tfidf_df.stack().reset_index()

tfidf_df = tfidf_df.rename(columns={0:'tfidf', 'level_0': 'chapter','level_1': 'term', 'level_2': 'term'})

tfidf_df = tfidf_df.loc[tfidf_df["tfidf"] > 0.0 ]

tfidf_df = tfidf_df.sort_values(by=['chapter','tfidf'], ascending=[True,False])#.groupby(['chapter']).head(100)

chapters = tfidf_df['chapter'].unique()

for chapter in text_titles:
    if not os.path.exists("./tf_idf"):
        os.mkdir("./tf_idf")
    terms = tfidf_df[tfidf_df['chapter'] == chapter]
    terms.to_csv(f"tf_idf/result_{chapter}.csv")
