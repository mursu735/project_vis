import gensim.models
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import pandas as pd
from wordcloud import WordCloud
import helpers
from PIL import Image

prev_word = ""

model_name = helpers.fetch_model_name()
new_model = gensim.models.Word2Vec.load(f"{model_name}").wv

def get_most_similar_words(word1, word2):
    words = []
    similar = ""
    similar2 = ""
    similar_both = ""
    if word1 != "":
        similar = new_model.most_similar(positive=[f"{word1}"], topn=20)
        words.append(word1)
    if word2 != "":
        similar2 = new_model.most_similar(positive=[f"{word2}"], topn=20)
        words.append(word2)
    if len(words) == 2:
        similar_both = new_model.most_similar(positive=words, topn=20)
    return similar, similar2, similar_both

def get_wordcloud(chapter):
    if chapter == "":
        return get_base_wordcloud_fig()
    res = "EPILOGUE"
    #number = chapter.split(" ")
    if chapter.isnumeric():
        res = f"CHAPTER {chapter}"
    tfidf = pd.read_csv(f"./tf_idf/result_{res}.csv", sep=",", header=0, usecols=["term", "tfidf"]).set_index("term").to_dict()
    wordcloud = WordCloud(background_color="white", width=1200, height=900, max_words=20).generate_from_frequencies(tfidf["tfidf"])
    fig = px.imshow(wordcloud)
    title = "Word cloud for "
    if res == "EPILOGUE":
        title += "Epilogue"
    else:
        title += f"Chapter {chapter}"
    fig.update_layout(title=title, title_x=0.5)
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_yaxes(visible=False, showticklabels=False)
    return fig

def get_base_wordcloud_fig():
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

#tokens = sorted(new_model.index_to_key)

def run_server():
    #wordcloud_fig = get_base_wordcloud_fig()
    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = html.Div(children=[
        html.H1(children='Hello Dash'),

        html.Div(children='''
            Dash: A web application framework for your data.
        '''),

        html.Div([
        "Input (chapter number (1-136) or \"Epilogue\"): ",
        dcc.Input(id='word-1', value='', type='text', style={"margin-left": "15px"}),
        dcc.Input(id='word-2', value='', type='text', style={"margin-left": "15px"}),
        html.Button('Recalculate', n_clicks=0, id='recalculate'),
        ]),

        html.Div(
            children=[
                dcc.Graph(
                    id='wordcloud-graph',
                    figure=get_base_wordcloud_fig(),
                    style={'display': 'inline-block'}
                ),
                    dcc.Graph(
                    id='wordcloud-graph2',
                    figure=get_base_wordcloud_fig(),
                    style={'display': 'inline-block'}
                ),
            ]
        )

        #html.Div(children=[dcc.Markdown("***All available words***"), dcc.Markdown("\n".join(tokens))]),
    ])
    '''
    @app.callback(
        Output('selected-word', 'children'),
        Output('word2vec-data', 'children'),
        Output('word2vec-data2', 'children'),
        Output('word2vec-data3', 'children'),
        Input('word-1', 'value'),
        Input('word-2', 'value'),
        Input('recalculate', 'n_clicks'))
    def display_click_data(word1, word2, n_clicks):
        global prev_word
        ctx = dash.callback_context
        clicked_element = ctx.triggered[0]["prop_id"].split(".")[0]
        if clicked_element == "recalculate":
            similar, similar2, similar_both = get_most_similar_words(word1, word2)
            similar = "\n".join(map(str, similar))
            similar2 = "\n".join(map(str, similar2))
            similar_both = "\n".join(map(str, similar_both))
            #print(similar)
            prev_word = " ".join([word1, word2])
            return prev_word, similar, similar2, similar_both
        else:
            return prev_word, '', '', ''
        #return json.dumps(clickData, indent=2)
    '''
    @app.callback(
        Output('wordcloud-graph', 'figure'),
        Input('word-1', 'value'),
        Input('recalculate', 'n_clicks'))
    def display_wordcloud1(chapter, n_clicks):
        ctx = dash.callback_context
        clicked_element = ctx.triggered[0]["prop_id"].split(".")[0]
        if clicked_element == "recalculate":
            return get_wordcloud(chapter)
        else:
            return dash.no_update

    @app.callback(
        Output('wordcloud-graph2', 'figure'),
        Input('word-2', 'value'),
        Input('recalculate', 'n_clicks'))
    def display_wordcloud2(chapter, n_clicks):
        ctx = dash.callback_context
        clicked_element = ctx.triggered[0]["prop_id"].split(".")[0]
        if clicked_element == "recalculate":
            return get_wordcloud(chapter)
        else:
            return dash.no_update

    if __name__ == '__main__':
        app.run_server(debug=True)


run_server()

#print(f"Time taken to start server: {time.time() - start_time}")
