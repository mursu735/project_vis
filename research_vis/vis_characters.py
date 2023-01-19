import glob
import pandas as pd
import plotly.express as px
import re


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

characters = ["Ishmael", "Ahab", "Moby Dick/White Whale", "Starbuck", "Stubb", "Flask", "Queequeg", "Tashtego", "Daggoo", "Fedallah", "Pip", "Bulkington", "Dough-Boy", "Fleece", "Perth", "Carpenter", "Manxman", "Mapple", "Elijah", "Bildad", "Pereg", "Pequod"]
directory_path = "./Chapters/"
text_files = glob.glob(f"{directory_path}/*.txt")
text_files = [file for file in text_files if "CHAPTER" in file or "Epilogue" in file]
res = {}

for chapter in text_files:
    text = open(chapter).read()
    current = {}
    for character in characters:
        names = character.split("/")
        total = 0
        for name in names:
            total += text.lower().count(name.lower())
        current[names[0]] = total
    key = chapter.split("/")
    if len(key) == 1:
        key = chapter.split("\\")
    key = key[-1].split(".")[0].title()
    res[key] = current

df = pd.DataFrame(data=res)

df = df.reindex(natural_sort(df.columns), axis=1).transpose()

fig = px.bar(df, color_discrete_sequence=px.colors.qualitative.Alphabet)

fig.update_layout(
    title="Plot Title",
    xaxis_title="Chapter",
    yaxis_title="Number of occurrences",
    legend_title="Character"
)

fig.show()

#print(df)