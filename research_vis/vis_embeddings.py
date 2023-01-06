import helpers
import json
import glob
import gensim.models
import gensim.downloader as api
import plotly.graph_objects as go
import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from umap import UMAP
import numpy as np
import pandas as pd
import time

def get_tfidf_chapter(labels):
    result_chapter = []
    result_tfidf = []
    tfidf_df = pd.read_csv("./tf_idf/result_total.csv", sep=",", header=0, usecols=["term", "chapter", "tfidf"])
    max_tfidf = tfidf_df.iloc[0]["tfidf"]
    print(f"Max: {max_tfidf}")
    for word in labels:
        chapter = tfidf_df.loc[tfidf_df['term'] == word]
        # Should not happen
        if chapter.empty:
            result_chapter.append(137)
            result_tfidf.append(0.0)
        else:
            result_tfidf.append(chapter["tfidf"].values[0] / max_tfidf)
            chapter = chapter["chapter"].values[0]
            if chapter == "Epilogue":
                result_chapter.append(136)
            else:
                split = chapter.split(" ")
                result_chapter.append(int(split[1]))
    return result_chapter, result_tfidf

def perform_dbscan(data):
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    scaled_p = scaler.transform(data)
    clustering = DBSCAN(eps=0.5, min_samples=5).fit(scaled_p)
    labels = clustering.labels_
    print(len(labels))
    return labels

def reduce_dimensions(wv, word_list):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vec = []
    lab = []
    for word in word_list:
        asd = wv.get_vector(word)
        vec.append(asd)
        lab.append(word)

    vectors = np.asarray(vec)
    labels = np.asarray(lab)  # fixed-width numpy strings

    # reduce using t-SNE
    red = UMAP(n_components=num_dimensions, random_state=0, init='random', n_neighbors=15, min_dist=0.1)
    #red = TSNE(n_components=num_dimensions, random_state=0, perplexity=5)
    vectors = red.fit_transform(vectors)
    print("DBSCAN for UMAP:")
    dbscan = perform_dbscan(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    chapter_tuple = get_tfidf_chapter(labels)
    data = {"x": x_vals, "y": y_vals, "labels": labels, "cluster": dbscan, "chapter": chapter_tuple[0], "opacity": chapter_tuple[1]}
    df = pd.DataFrame(data=data)
    return df

def reduce_pca(wv, word_list):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vec = []
    lab = []
    for word in word_list:
        asd = wv.get_vector(word)
        vec.append(asd)
        lab.append(word)

    vectors = np.asarray(vec)
    labels = np.asarray(lab)  # fixed-width numpy strings
    # reduce using PCA
    pca = PCA(n_components=num_dimensions)
    vectors = pca.fit_transform(vectors)
    #print(vectors)
    print("DBSCAN for PCA")
    dbscan = perform_dbscan(vectors)
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    chapter_tuple = get_tfidf_chapter(labels)
    data = {"x": x_vals, "y": y_vals, "labels": labels, "cluster": dbscan, "chapter": chapter_tuple[0], "opacity": chapter_tuple[1]}
    df = pd.DataFrame(data=data)
    return df

def get_plot(df_umap, df_pca):
    fig = make_subplots(rows=1, cols=2,
                    vertical_spacing=0.02,
                    subplot_titles=("Embeddings reduced with tSNE", "Embeddings reduced with PCA"))

    trace1 = go.Scattergl(
        x=df_umap["x"],
        y=df_umap["y"],
        mode="markers",
        text=df_umap["labels"],
        marker=dict(
            color=df_umap["chapter"],
            colorbar=dict(
                title="Colorbar"
            ),
            colorscale="Viridis",
            opacity=df_umap["opacity"]
        ),
        name="words")

    trace2 = go.Scattergl(
        x=df_pca["x"],
        y=df_pca["y"],
        mode="markers",
        marker=dict(
            color=df_umap["chapter"],
            colorbar=dict(
                title="Colorbar"
            ),
            colorscale="Viridis",
            opacity=df_umap["opacity"]
        ),
        text=df_pca["labels"],
        name="words")

    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    return fig



model_name = helpers.fetch_model_name()

#print(model_name)

new_model = gensim.models.Word2Vec.load(f"{model_name}")
#wv = api.load('word2vec-google-news-300')
#labels = np.asarray(wv.index_to_key[:500])

trained_model = new_model.wv
model_labels = np.asarray(trained_model.index_to_key)

tfidf_df = pd.read_csv("./tf_idf/result_total.csv", sep=",", header=0, usecols=["term", "chapter", "tfidf"])
filtered = tfidf_df[tfidf_df["term"].isin(model_labels)]
filtered = filtered[filtered["chapter"] != "filler_text"]

labels = filtered[filtered["tfidf"] >= 0.05]["term"].values

#labels = tfidf_df["term"].values

print(f"Final labels: {labels}")
print(len(labels))

#wv = api.load('word2vec-google-news-300')
wv = new_model.wv

print(len(labels))

#colorscale = get_tfidf_chapter(labels)

        
# Go through all of the words of tf-idf and get the highest value, map it to the chapter where it is highest, filter words where the highest is less than 0.05

#wv = api.load("word2vec-google-news-300")  # load glove vectors

#word_list_filled.append(similar_symptoms)

start_time = time.time()

df_umap = reduce_dimensions(wv, labels)

print(f"Time taken to create UMAP mapping: {time.time() - start_time}")
start_time = time.time()

df_pca = reduce_pca(wv, labels)

print(f"Time taken to create PCA mapping: {time.time() - start_time}")


#print(len(x_vals))
fig = get_plot(df_umap, df_pca)
'''
fig = make_subplots(rows=1, cols=2,
                    vertical_spacing=0.02,
                    subplot_titles=("Embeddings reduced with tSNE", "Embeddings reduced with PCA"))
'''
'''
G = nx.Graph()

for i in range(0, len(labels)):
    G.add_node(labels[i], pos=(x_vals[i], y_vals[i]))

node_x = []
node_y = []
circles = {}
for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_x.append(x)
    node_y.append(y)
    circles[node] = [dict(
        type="circle",
        xref="x", yref="y",
        x0=x - 5, y0=y - 5,
        x1=x + 5, y1=y + 5,
        line=dict(color="DarkOrange"))]
'''
circles_pca = {}
circles = {}

start_time = time.time()
length = len(df_umap)

for i in range(0, length):
    umap_cur = df_umap.iloc[i]
    pca_cur = df_pca.iloc[i]
    '''
    x_pca = x_vals_pca[i]
    y_pca = y_vals_pca[i]
    x = x_vals[i]
    y = y_vals[i]
    '''
    label_umap = umap_cur["labels"]
    label_pca = pca_cur["labels"]
    if label_umap not in circles:
        circles[label_umap] = []
    if label_pca not in circles:
        circles[label_pca] = []

    x_umap = umap_cur["x"]
    y_umap = umap_cur["y"]
    x_pca = pca_cur["x"]
    y_pca = pca_cur["y"]
    circles[label_umap].append(dict(
        type="circle",
        xref="x1", yref="y1",
        x0=x_umap - 5, y0=y_umap - 5,
        x1=x_umap + 5, y1=y_umap + 5,
        line=dict(color="DarkOrange")))
    circles[label_pca].append(dict(
        type="circle",
        xref="x2", yref="y2",
        x0=x_pca - 0.15, y0=y_pca - 0.15,
        x1=x_pca + 0.15, y1=y_pca + 0.15,
        line=dict(color="DarkOrange")))

print(f"Time taken to calculate circles: {time.time() - start_time}")
start_time = time.time()

def highlight_group(group):
    result = []
    for tracer_ix, tracer in enumerate(fig["data"]):
        colors = ["red" if datapoint_group == group else "black" for datapoint_group in fig["data"][tracer_ix]["text"]]
        result.append(colors)
    return result

'''
buttons = [
    {
        "label": group,
        "method": "update",
        "args": [
            {"marker.color": highlight_group(group)},
            {
                "shapes": circles[group]
            }
        ],
    }
    for group in labels
]

fig.update_layout(
    updatemenus=[
        {
            "buttons": buttons,
        }
    ],
    margin={"l": 0, "r": 0, "t": 25, "b": 0},
    height=700
)
'''
print(f"Time taken to create graph: {time.time() - start_time}")
start_time = time.time()

def run_server(fig):
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for your data.
        '''),

        dcc.Graph(
            id='example-graph',
            figure=fig
        ),

        html.Button('Reset graph', n_clicks=0, id='reset-graph'),

        html.Div(children=["Selected word", html.Pre(id='selected-word')]),

        dbc.Row(
            [
                dbc.Col(html.Div([dcc.Markdown("""
                        **Click Data**

                        Words in the same chapter.
                    """),
                    html.Pre(id='click-data')])),
                    
                dbc.Col(html.Div([dcc.Markdown("""
                    **Click Data**

                    Words in the same cluster. 
                """),
                html.Pre(id='cluster-data')]))
            ]
        ),
    ])
    
    @app.callback(
        Output('example-graph', 'figure'),
        Output('click-data', 'children'),
        Output('cluster-data', 'children'),
        Output('selected-word', 'children'),
        Input('example-graph', 'clickData'),
        Input('reset-graph', 'n_clicks'))
    def display_click_data(clickData, n_clicks):
        ctx = dash.callback_context
        clicked_element = ctx.triggered[0]["prop_id"].split(".")[0]
        if clicked_element == "reset-graph":
            fig = get_plot(df_umap, df_pca)
            return fig, [], [], "None"
        if isinstance(clickData, dict):
            fig = make_subplots(rows=1, cols=2,
                    vertical_spacing=0.02,
                    subplot_titles=("Embeddings reduced with tSNE", "Embeddings reduced with PCA"))
            text = clickData["points"][0]["text"]
            highlight_umap = df_umap.loc[df_umap['labels'] == text]
            highlight_pca = df_pca.loc[df_pca['labels'] == text]
            trace1 = go.Scattergl(x=df_umap["x"], y=df_umap["y"], mode="markers", text=labels, marker=dict(color="black"), name="words")
            trace1_hl = go.Scattergl(x=highlight_umap["x"], y=highlight_umap["y"], mode="markers", text=highlight_umap["labels"], marker=dict(color="red"), name="Highlighted word")
            trace2 = go.Scattergl(x=df_pca["x"], y=df_pca["y"], mode="markers", text=df_pca["labels"], marker=dict(color="black"), name="words")
            trace2_hl = go.Scattergl(x=highlight_pca["x"], y=highlight_pca["y"], mode="markers", text=highlight_pca["labels"], marker=dict(color="red"), name="Highlighted word")

            fig.append_trace(trace1,1,1)
            fig.append_trace(trace1_hl,1,1)
            fig.append_trace(trace2,1,2)
            fig.append_trace(trace2_hl,1,2)
            # get marker color and circle
            #fig.update_traces(marker=dict(color=updated))
            for shape in circles[text]:
                fig.add_shape(shape)
            '''
            fig.update_layout(
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "label": group,
                                "method": "update",
                                "args": [
                                    {"marker.color": highlight_group(group)},
                                    {
                                        "shapes": circles[group]
                                    }
                                    
                                ],
                            }
                            for group in labels
                        ]
                    }
                ],
                margin={"l": 0, "r": 0, "t": 25, "b": 0},
                height=700
            )
            '''
            chapter = df_umap[df_umap["labels"] == text]["chapter"].values[0]
            #print(chapter)
            word_list = df_umap[df_umap["chapter"] == chapter]["labels"]
            #print(word_list)
            word_cluster = highlight_umap["cluster"].values[0]
            words_in_cluster = df_umap[df_umap["cluster"] == word_cluster]["labels"]
            return fig, "\n".join(words_in_cluster), "\n".join(word_list), text

        return dash.no_update, [], [], "None"
        #return json.dumps(clickData, indent=2)

    if __name__ == '__main__':
        app.run_server(debug=True)


run_server(fig)

print(f"Time taken to start server: {time.time() - start_time}")
#fig.show()

#fig.write_html("server/bar_chart_simple.html", auto_play=False, include_plotlyjs="cdn")


'''
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')
'''

"""
fig.add_trace(
    go.Scatter(x=asd, y=asd, mode="markers")
)


fig.show()
"""
#print(word_distances)