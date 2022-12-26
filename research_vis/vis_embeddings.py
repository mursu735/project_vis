import helpers
import json
import glob
import gensim.models
import gensim.downloader as api
import plotly.graph_objects as go
import dash
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.decomposition import IncrementalPCA    # inital reduction
from sklearn.manifold import TSNE                   # final reduction
from sklearn.decomposition import PCA
import umap
import numpy as np


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
    tsne = TSNE(n_components=num_dimensions, random_state=0, perplexity=5)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

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
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels

model_name = helpers.fetch_model_name()

#print(model_name)

new_model = gensim.models.Word2Vec.load(f"{model_name}")
wv = new_model.wv
labels = np.asarray(wv.index_to_key[:100])

text_files = glob.glob("./tf_idf/*.csv")
# Go through all of the words of tf-idf and get the highest value, map it to the chapter where it is highest, filter words where the highest is less than 0.05

#wv = api.load("word2vec-google-news-300")  # load glove vectors

#word_list_filled.append(similar_symptoms)

x_vals, y_vals, labels = reduce_dimensions(wv, labels)

x_vals_pca, y_vals_pca, labels_pca = reduce_pca(wv, labels)



fig = make_subplots(rows=1, cols=2,
                    vertical_spacing=0.02,
                    subplot_titles=("Embeddings reduced with tSNE", "Embeddings reduced with PCA"))
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

for i in range(0, len(labels)):
    x_pca = x_vals_pca[i]
    y_pca = y_vals_pca[i]
    x = x_vals[i]
    y = y_vals[i]
    if labels[i] not in circles:
        circles[labels[i]] = []
    if labels_pca[i] not in circles:
        circles[labels_pca[i]] = []

    circles[labels[i]].append(dict(
        type="circle",
        xref="x1", yref="y1",
        x0=x - 5, y0=y - 5,
        x1=x + 5, y1=y + 5,
        line=dict(color="DarkOrange")))
    circles[labels_pca[i]].append(dict(
        type="circle",
        xref="x2", yref="y2",
        x0=x_pca - 0.15, y0=y_pca - 0.15,
        x1=x_pca + 0.15, y1=y_pca + 0.15,
        line=dict(color="DarkOrange")))

#trace = go.Scatter(x=x_vals, y=y_vals, mode='text', text=labels)
trace1 = go.Scatter(x=x_vals, y=y_vals, mode="markers", text=labels, name="words")
trace2 = go.Scatter(x=x_vals_pca, y=y_vals_pca, mode="markers", text=labels_pca, name="words")

fig.append_trace(trace1,1,1)
fig.append_trace(trace2,1,2)

def highlight_group(group):
    result = []
    for tracer_ix, tracer in enumerate(fig["data"]):
        colors = ["red" if datapoint_group == group else "black" for datapoint_group in fig["data"][tracer_ix]["text"]]
        result.append(colors)
    return result

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

def run_server(fig):
    app = Dash(__name__)
    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for your data.
        '''),

        dcc.Graph(
            id='example-graph',
            figure=fig
        ),

        html.Div([
            dcc.Markdown("""
                **Click Data**

                Click on points in the graph.
            """),
            html.Pre(id='click-data'),
        ], className='three columns'),
    ])
    
    @app.callback(
        Output('example-graph', 'figure'),
        Input('example-graph', 'clickData'))
    def display_click_data(clickData):
        #print(type(clickData))
        if isinstance(clickData, dict):
            fig = make_subplots(rows=1, cols=2,
                    vertical_spacing=0.02,
                    subplot_titles=("Embeddings reduced with tSNE", "Embeddings reduced with PCA"))
            text = clickData["points"][0]["text"]
            index = np.where(labels == text)[0][0].item()
            index_pca = np.where(labels_pca == text)[0][0].item()
            trace1 = go.Scatter(x=x_vals, y=y_vals, mode="markers", text=labels, marker=dict(color="black"), name="words")
            trace1_hl = go.Scatter(x=[x_vals[index]], y=[y_vals[index]], mode="markers", text=[labels[index]], marker=dict(color="red"), name="Highlighted word")
            trace2 = go.Scatter(x=x_vals_pca, y=y_vals_pca, mode="markers", text=labels_pca, marker=dict(color="black"), name="words")
            trace2_hl = go.Scatter(x=[x_vals_pca[index_pca]], y=[y_vals_pca[index_pca]], mode="markers", text=[labels_pca[index_pca]], marker=dict(color="red"), name="Highlighted word")

            fig.append_trace(trace1,1,1)
            fig.append_trace(trace1_hl,1,1)
            fig.append_trace(trace2,1,2)
            fig.append_trace(trace2_hl,1,2)
            # get marker color and circle
            #fig.update_traces(marker=dict(color=updated))
            for shape in circles[text]:
                fig.add_shape(shape)
            
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
            return fig

        return dash.no_update
        #return json.dumps(clickData, indent=2)


    if __name__ == '__main__':
        app.run_server(debug=True)


run_server(fig)
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