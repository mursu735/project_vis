import glob
import re
import pandas as pd

def select_rows(row, df):
    cur_tfidf = row["tfidf"]
    term = row["term"]
    # Consider only words that have only alphabets
    if not term.isalpha():
        return False
    # Term is not in the list, so add it automatically
    if term not in df["term"].values:
        return True
    # Keep the higher tf-idf
    old_tfidf = df.loc[df['term'] == term]["tfidf"].values[0]
    if cur_tfidf > old_tfidf:
        return True
    return False

directory_path = "./tf_idf/"
tfidf_files = glob.glob(f"{directory_path}/*.csv")
result = pd.DataFrame(columns=["term", "chapter", "tfidf"])

for chapter in tfidf_files:
    reader = pd.read_csv(chapter, sep=",", header=0, usecols=["term", "chapter", "tfidf"])
    print(f"Currently in {chapter}", end='\r')
    reader['keep'] = reader.apply(select_rows, args=(result, ), axis=1)
    reader = reader[reader["keep"]]
    reader = reader.drop(columns=["keep"])
    result = result.set_index('term')
    reader = reader.set_index('term')
    # In case of conflicts, keep the higher tf-idf, lower values were already dropped earlier and values are not rewritten
    result = reader.combine_first(result)
    result.reset_index(inplace=True)


#asd = result.loc[result['term'] == "set"]["tfidf"]
#print(asd)
result = result.sort_values(by=['tfidf'], ascending=[False])
print(result)
result.to_csv("tf_idf/result_total.csv")
    