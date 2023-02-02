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

#for word in new_model.index_to_key:
#    print(word)

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
        dcc.Input(id='text-input', value='', type='text'),
        html.Button('Recalculate', n_clicks=0, id='recalculate'),
        ]),

        html.Div(children=["Selected word", html.Pre(id='selected-word')]),


        dbc.Row(
            [
                dbc.Col(html.Div([dcc.Markdown("""
                        **Click Data**

                        Closest words in word2vec.
                    """),
                    html.Pre(id='word2vec-data')])),
                    
                dbc.Col(html.Div([dcc.Markdown("""
                    **Click Data**

                    Words in the same cluster. 
                """),
                html.Pre(id='cluster-data')]))
            ]
        ),
    ])
    
    @app.callback(
        Output('selected-word', 'children'),
        Output('word2vec-data', 'children'),
        Input('text-input', 'value'),
        Input('recalculate', 'n_clicks'))
    def display_click_data(text, n_clicks):
        global prev_word
        ctx = dash.callback_context
        clicked_element = ctx.triggered[0]["prop_id"].split(".")[0]
        if clicked_element == "recalculate":
            similar = new_model.most_similar(positive=[f"{text}"], topn=20)
            similar = "\n".join(map(str, similar))
            print(similar)
            prev_word = text
            return text, similar
        else:
            return prev_word, ''
        #return json.dumps(clickData, indent=2)

    if __name__ == '__main__':
        app.run_server(debug=True)


run_server()

#print(f"Time taken to start server: {time.time() - start_time}")
