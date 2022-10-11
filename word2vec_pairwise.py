import fetch_model_name
import gensim.models
import plotly.graph_objects as go


model_name = fetch_model_name.fetch_model_name()

print(model_name)

new_model = gensim.models.Word2Vec.load(f"gensim-model-{model_name}")

#wv = api.load('word2vec-google-news-300')

wv = new_model.wv

word_list = ["fever", "chills", "sweats", "aches", "pains", "fatigue", "coughing", "breathing", "nausea", "vomiting", "diarrhoea"]

word_distances = {}

for i in range(0, len(word_list)):
    for j in range(i+1, len(word_list)):
        word = f"{word_list[i]} -> {word_list[j]}"
        diff = wv.similarity(word_list[i], word_list[j])
        word_distances[word] = diff


asd = list(word_distances.values())

fig = go.Figure()

fig.add_trace(
    go.Scatter(x=asd, y=asd, mode="markers")
)


fig.show()

#print(word_distances)