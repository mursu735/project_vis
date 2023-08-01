import helpers
import json
import glob
import gensim.models
import gensim.downloader as api
import plotly.graph_objects as go
import plotly.express as px
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
from wordcloud import WordCloud
from umap import UMAP
from PIL import Image
import numpy as np
import pandas as pd
import time
import re

cur_chapter = "None"
descriptive_chapters = [32, 33, 35, 42, 45, 55, 56, 57, 62, 63, 68, 74, 75, 76, 77, 79, 80, 82, 83, 85, 86, 88, 89, 90, 92, 101, 102, 103, 104, 105]

def perform_dbscan(data, eps, min_samples):
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    scaled_p = scaler.transform(data)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scaled_p)
    labels = clustering.labels_
    print(len(labels))
    return labels

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_subplot_template():
    return make_subplots(
        rows=1,
        cols=2,
        vertical_spacing=0.02,
        specs=[[{'type': 'surface'}, {'type': 'xy'}]],
        subplot_titles=("Chapter embeddings reduced with UMAP", "Chapter embeddings reduced with PCA"))

def get_wordcloud(chapter):
    res = "EPILOGUE"
    number = chapter.split(" ")
    if len(number) > 1:
        res = f"CHAPTER {number[1]}"
    tfidf = pd.read_csv(f"./tf_idf/result_{res}.csv", sep=",", header=0, usecols=["term", "tfidf"]).set_index("term").to_dict()
    wordcloud = WordCloud(background_color="white", width=1200, height=900, max_words=20).generate_from_frequencies(tfidf["tfidf"])
    fig = px.imshow(wordcloud)
    title = "Word cloud for "
    if res == "EPILOGUE":
        title += "Epilogue"
    else:
        title += f"Chapter {number[1]}"
    fig.update_layout(title=title, title_x=0.5)
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(visible=False, showticklabels=False)
    return fig
    #print(chapter)

def get_base_wordcloud_fig():
    wordcloud_fig = go.Figure()
    wordcloud_fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode="markers",
        marker=dict(opacity=0)))
    im = Image.open("banner.png") # Can be many different formats.
    wordcloud_fig.add_layout_image(
            dict(
                source=im,
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                sizing="fill",
                layer="above")
    )

    wordcloud_fig.update_xaxes(range=[0, 1], visible=False, showticklabels=False)
    wordcloud_fig.update_yaxes(range=[0, 0.5], visible=False, showticklabels=False)
    return wordcloud_fig

def get_base_wordcloud_fig2():
    wordcloud_fig = go.Figure()
    wordcloud_fig.add_trace(go.Scatter(
        x=[0],
        y=[0],
        mode="markers",
        marker=dict(opacity=0)))
    im = Image.open("test_banner.png") # Can be many different formats.
    wordcloud_fig.add_layout_image(
            dict(
                source=im,
                xref="x",
                yref="y",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                sizing="stretch",
                layer="above")
    )

    wordcloud_fig.update_xaxes(range=[0, 1], visible=False, showticklabels=False)
    wordcloud_fig.update_yaxes(range=[0, 1], visible=False, showticklabels=False)
    return wordcloud_fig

def reduce_dimensions(dv, tag_list, n_neighbors, min_dist, eps, min_samples):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)
    print(tag_list)
    # extract the words & their vectors, as numpy arrays
    vec = []
    lab = []
    chapter = []
    descriptive = []
    for tag in tag_list:
        split2 = tag.split(" ")
        number = 0
        if len(split2) == 1:
            number = 136
        else:
            number = int(split2[1])
        if number not in descriptive_chapters:
            asd = dv.get_vector(tag)
            vec.append(asd)
            lab.append(tag)
            if number in descriptive_chapters:
                descriptive.append(1)
            else:
                descriptive.append(0)
            chapter.append(number)

    vectors = np.asarray(vec)
    labels = np.asarray(lab)  # fixed-width numpy strings

    # reduce using t-SNE
    red = UMAP(output_metric='haversine', n_components=num_dimensions, random_state=0, init='random', n_neighbors=n_neighbors, min_dist=min_dist)
    #red = TSNE(n_components=num_dimensions, random_state=0, perplexity=5)
    dbscan = perform_dbscan(vectors, eps, min_samples)
    vectors = red.fit(vectors) #red.fit_transform(vectors)
    #print("DBSCAN for UMAP:")
    #dbscan = perform_dbscan(vectors)
    print(vectors)
    x_vals = np.sin(vectors.embedding_[:, 0]) * np.cos(vectors.embedding_[:, 1]) #[v[0] for v in vectors]
    y_vals = np.sin(vectors.embedding_[:, 0]) * np.sin(vectors.embedding_[:, 1]) #[v[1] for v in vectors]
    z_vals = np.cos(vectors.embedding_[:, 0]) #[v[2] for v in vectors]
    data = {"x": x_vals, "y": y_vals, "z": z_vals, "labels": labels, "cluster": dbscan, "chapter": chapter, "descriptive": descriptive}
    df = pd.DataFrame(data=data)
    return df

def reduce_pca(dv, tag_list, eps, min_samples):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vec = []
    lab = []
    for tag in tag_list:
        asd = dv.get_vector(tag)
        vec.append(asd)
        lab.append(tag)

    vectors = np.asarray(vec)
    labels = np.asarray(lab)  # fixed-width numpy strings
    # reduce using PCA
    pca = PCA(n_components=num_dimensions)
    vectors = pca.fit_transform(vectors)
    #print(vectors)
    print("DBSCAN for PCA")
    dbscan = perform_dbscan(vectors, eps, min_samples)
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    data = {"x": x_vals, "y": y_vals, "labels": labels, "cluster": dbscan}
    df = pd.DataFrame(data=data)
    return df

def get_color_dict(mode):
    color_dict = {}
    if mode == "Chapter":
        color_dict["color"] = df_umap["chapter"]
        color_dict["colorbar"] = {"title": "Colorbar"}
        color_dict["colorscale"] = "rdbu"
    elif mode == "Cluster":
        color_dict["color"] = df_umap["cluster"]
    elif mode == "Descriptive":
        color_dict["color"] = df_umap["descriptive"]
    return color_dict

def get_plot(df_umap, df_pca, mode):
    fig = get_subplot_template()
    color_dict = get_color_dict(mode)
    trace1 = go.Scatter3d(
        x=df_umap["x"],
        y=df_umap["y"],
        z=df_umap["z"],
        mode="lines+markers",
        text=df_umap["labels"],
        marker=color_dict,
        name="words")

    trace2 = go.Scattergl(
        x=df_pca["x"],
        y=df_pca["y"],
        mode="markers",
        text=df_pca["labels"],
        marker=color_dict,
        name="words")

    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,2)
    fig.update_layout(uirevision='constant')
    return fig

def get_highlight_plot(mode, text):
    fig = get_subplot_template()
    highlight_umap = df_umap.loc[df_umap['labels'] == text]
    highlight_pca = df_pca.loc[df_pca['labels'] == text]
    color_dict = get_color_dict(mode)
    trace1 = go.Scatter3d(x=df_umap["x"], y=df_umap["y"], z=df_umap["z"], mode="lines+markers", text=df_umap["labels"], marker=color_dict, name="chapter")
    trace1_hl = go.Scatter3d(x=highlight_umap["x"], y=highlight_umap["y"], z=highlight_umap["z"], mode="markers", text=highlight_umap["labels"], marker=dict(color="red", size=10), name="Highlighted chapter")
    trace2 = go.Scattergl(x=df_pca["x"], y=df_pca["y"], mode="markers", text=df_pca["labels"], marker=dict(color="black"), name="chapter")
    trace2_hl = go.Scattergl(x=highlight_pca["x"], y=highlight_pca["y"], mode="markers", text=highlight_pca["labels"], marker=dict(color="red"), name="Highlighted chapter")
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace1_hl,1,1)
    fig.append_trace(trace2,1,2)
    fig.append_trace(trace2_hl,1,2)
    fig.update_layout(uirevision='constant')
    return fig

model_name = helpers.fetch_doc2vec_model_name()

#print(model_name)
new_model = gensim.models.Doc2Vec.load(f"{model_name}")
#wv = api.load('word2vec-google-news-300')
#labels = np.asarray(wv.index_to_key[:500])

dv = new_model.dv

tags = []

with open("doc2vec_tags.txt") as file:
    tags = [line.replace('\n','') for line in file.readlines()]

tags = natural_sort(tags)
#wv = api.load('word2vec-google-news-300')

#colorscale = get_tfidf_chapter(labels)

        
# Go through all of the words of tf-idf and get the highest value, map it to the chapter where it is highest, filter words where the highest is less than 0.05

#wv = api.load("word2vec-google-news-300")  # load glove vectors

#word_list_filled.append(similar_symptoms)

start_time = time.time()

df_umap = reduce_dimensions(dv, tags, 15, 0.15, 0.5, 5)

print(f"Time taken to create UMAP mapping: {time.time() - start_time}")
start_time = time.time()

df_pca = reduce_pca(dv, tags, 0.5, 5)

print(f"Time taken to create PCA mapping: {time.time() - start_time}")


#print(len(x_vals))
fig = get_plot(df_umap, df_pca, "Chapter")

#circles_pca = {}
#circles = {}

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

    x_umap = umap_cur["x"]
    y_umap = umap_cur["y"]
    x_pca = pca_cur["x"]
    y_pca = pca_cur["y"]
'''
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
'''
print(f"Time taken to calculate circles: {time.time() - start_time}")
start_time = time.time()

def highlight_group(group):
    result = []
    for tracer_ix, tracer in enumerate(fig["data"]):
        colors = ["red" if datapoint_group == group else "black" for datapoint_group in fig["data"][tracer_ix]["text"]]
        result.append(colors)
    return result

print(f"Time taken to create graph: {time.time() - start_time}")
start_time = time.time()

def run_server(fig):
    wordcloud_fig = get_base_wordcloud_fig()
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for your data.
        '''),

        html.P("Color mode:"),
        dcc.RadioItems(
            id='color-mode',
            value='Chapter',
            options=['Chapter', 'Cluster', 'Descriptive'],
        ),

        html.Div([
            html.H2("UMAP controls"),
            "n_neighbors: ",
            dcc.Input(id='n_neighbors', value=15, type='number'),
            "min_dist: ",
            dcc.Input(id='min_dist', value=0.1, type='number', min=0.01, max=1.0, step=0.01),
        ]),

        html.Div([
            html.H2("DBSCAN controls"),
            "min_samples: ",
            dcc.Input(id='min_samples', value=5, type='number'),
            "eps: ",
            dcc.Input(id='eps', value=0.05, type='number', min=0.01, max=15.0, step=0.01),
        ]),

        html.Button('Recalculate graph', n_clicks=0, id='calculate-graph'),

        dcc.Graph(
            id='example-graph',
            figure=fig
        ),

        html.Button('Reset graph', n_clicks=0, id='reset-graph'),

        html.Div(children=["Selected chapter", html.Pre(id='selected-word')]),

        dcc.Graph(
            id='wordcloud-graph',
            figure=wordcloud_fig
        ),

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



        html.Div([
            html.H2(children='Wordcloud for chapters'),
            "Input (chapter number or \"Epilogue\"): ",
            dcc.Input(id='word-1', value='', type='text', style={"margin-left": "15px"}),
            dcc.Input(id='word-2', value='', type='text', style={"margin-left": "15px"}),
            html.Button('Recalculate', n_clicks=0, id='recalculate-wordcloud'),
        ]),

        html.Div(
            children=[
                dcc.Graph(
                    id='wordcloud-below',
                    figure=get_base_wordcloud_fig2(),
                    style={'display': 'inline-block'}
                ),
                    dcc.Graph(
                    id='wordcloud-below2',
                    figure=get_base_wordcloud_fig2(),
                    style={'display': 'inline-block'}
                ),
            ]
        )
    ])
    
    @app.callback(
        Output('example-graph', 'figure'),
        Output('wordcloud-graph', 'figure'),
        Output('click-data', 'children'),
        Output('cluster-data', 'children'),
        Output('selected-word', 'children'),
        Input('example-graph', 'clickData'),
        Input('reset-graph', 'n_clicks'),
        Input('calculate-graph', 'n_clicks'),
        Input("color-mode", "value"),
        Input("n_neighbors", "value"),
        Input("min_dist", "value"),
        Input("min_samples", "value"),
        Input("eps", "value"),)
    def display_click_data(clickData, n_clicks, recalc, mode, n_neighbors, min_dist, min_samples, eps):
        global cur_chapter
        global dv
        global tags
        global df_umap
        global df_pca
        ctx = dash.callback_context
        clicked_element = ctx.triggered[0]["prop_id"].split(".")[0]
        if clicked_element == "n_neighbors" or clicked_element == "min_dist":
            dash.no_update, dash.no_update, [], [], "None"
        # Selected a new color scheme
        if clicked_element == "color-mode":
            if cur_chapter == "None":
                fig = get_plot(df_umap, df_pca, mode)
                return fig, dash.no_update, [], [], "None"
            else:
                fig = get_highlight_plot(mode, cur_chapter)
                chapter = df_umap[df_umap["labels"] == cur_chapter]["chapter"].values[0]
                highlight_umap = df_umap.loc[df_umap['labels'] == cur_chapter]
                word_list = df_umap[df_umap["chapter"] == chapter]["labels"]
                word_cluster = highlight_umap["cluster"].values[0]
                words_in_cluster = df_umap[df_umap["cluster"] == word_cluster]["labels"]
                return fig, dash.no_update, "\n".join(words_in_cluster), "\n".join(word_list), cur_chapter
        if clicked_element == "calculate-graph":
            start_time = time.time()
            df_umap = reduce_dimensions(dv, tags, n_neighbors, min_dist, min_samples, eps)
            print(f"Time taken to create UMAP mapping: {time.time() - start_time}")
            start_time = time.time()
            df_pca = reduce_pca(dv, tags, min_samples, eps)
            print(f"Time taken to create PCA mapping: {time.time() - start_time}")
            fig = get_plot(df_umap, df_pca, mode)
            return fig, dash.no_update, [], [], "None"
        # Clicked on reset button
        if clicked_element == "reset-graph":
            cur_chapter = "None"
            fig = get_plot(df_umap, df_pca, mode)
            return fig, get_base_wordcloud_fig(), [], [], "None"
        # Clicked on the graph
        if isinstance(clickData, dict):
            fig = get_subplot_template()
            text = clickData["points"][0]["text"]
            if not text == "None":
                fig = get_highlight_plot(mode, text)
                highlight_umap = df_umap.loc[df_umap['labels'] == text]
                start_time = time.time()
                wordcloud = get_wordcloud(text)
                print(f"Time taken to calculate word cloud: {time.time() - start_time}")
                '''
                fig.add_shape(dict(
                    type="circle",
                    xref="x2", yref="y2",
                    x0=highlight_pca["x"] - 0.15, y0=highlight_pca["y"] - 0.15,
                    x1=highlight_pca["x"] + 0.15, y1=highlight_pca["y"] + 0.15,
                    line=dict(color="DarkOrange")))
                '''
                chapter = df_umap[df_umap["labels"] == text]["chapter"].values[0]
                #print(chapter)
                word_list = df_umap[df_umap["chapter"] == chapter]["labels"]
                #print(word_list)
                word_cluster = highlight_umap["cluster"].values[0]
                words_in_cluster = df_umap[df_umap["cluster"] == word_cluster]["labels"]
                cur_chapter = text
                return fig, wordcloud, "\n".join(words_in_cluster), "\n".join(word_list), text

        return dash.no_update, dash.no_update, [], [], "None"
        #return json.dumps(clickData, indent=2)

    @app.callback(
        Output('wordcloud-below', 'figure'),
        Input('word-1', 'value'),
        Input('recalculate-wordcloud', 'n_clicks'))
    def display_wordcloud1(chapter, n_clicks):
        ctx = dash.callback_context
        clicked_element = ctx.triggered[0]["prop_id"].split(".")[0]
        if clicked_element == "recalculate-wordcloud":
            if chapter.isnumeric():
                chapter = f"CHAPTER {chapter}"
            return get_wordcloud(chapter)
        else:
            return dash.no_update

    @app.callback(
        Output('wordcloud-below2', 'figure'),
        Input('word-2', 'value'),
        Input('recalculate-wordcloud', 'n_clicks'))
    def display_wordcloud2(chapter, n_clicks):
        ctx = dash.callback_context
        clicked_element = ctx.triggered[0]["prop_id"].split(".")[0]
        if clicked_element == "recalculate-wordcloud":
            if chapter.isnumeric():
                chapter = f"CHAPTER {chapter}"
            return get_wordcloud(chapter)
        else:
            return dash.no_update

    if __name__ == '__main__':
        app.run_server(debug=True)


run_server(fig)

print(f"Time taken to start server: {time.time() - start_time}")
