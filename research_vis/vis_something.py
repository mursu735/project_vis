import gensim.models
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import helpers

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
        "Input: ",
        dcc.Input(id='word-1', value='', type='text', style={"margin-left": "15px"}),
        dcc.Input(id='word-2', value='', type='text', style={"margin-left": "15px"}),
        html.Button('Recalculate', n_clicks=0, id='recalculate'),
        ]),

        html.Div(children=["Selected words", html.Pre(id='selected-word')]),


        dbc.Row(
            [
                dbc.Col(html.Div([dcc.Markdown("""
                        **Click Data**

                        Closest words to first word in word2vec.
                    """),
                    html.Pre(id='word2vec-data')])),
                    
                dbc.Col(html.Div([dcc.Markdown("""
                    **Click Data**

                    Closest words to second word in word2vec.
                """),
                html.Pre(id='word2vec-data2')])),

                dbc.Col(html.Div([dcc.Markdown("""
                    **Click Data**

                    Closest words to both words in word2vec.
                """),
                html.Pre(id='word2vec-data3')]))
            ]
        ),

        #html.Div(children=[dcc.Markdown("***All available words***"), dcc.Markdown("\n".join(tokens))]),
    ])
    
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

    if __name__ == '__main__':
        app.run_server(debug=True)


run_server()

#print(f"Time taken to start server: {time.time() - start_time}")
