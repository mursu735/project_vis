import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
from scipy.spatial.distance import pdist, squareform
import base64
import word2vec_helpers

def plot_timelapse_graph(coords_map, unique_times_sorted, text):
    width = word2vec_helpers.get_width()
    height = word2vec_helpers.get_height()
    image_filename = word2vec_helpers.get_image_name()

    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    fig_dict["layout"]["xaxis"] = {"range": [0, width], "autorange": False}
    fig_dict["layout"]["yaxis"] = {"range": [0, height], "autorange": False}
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "label": "play",
                    "method": "animate",
                    "args": [None,
                    {
                        "frame": {"duration": 4000, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                    }]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                }
            ]
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Date:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 1000, "easing": "cubic-in-out"},
        "pad": {"b": 100, "t": 100},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    data_dict = {
        "x": coords_map[unique_times_sorted[0]]["x"],
        "y": coords_map[unique_times_sorted[0]]["y"],
        "text": coords_map[unique_times_sorted[0]]["text"],
        "mode": "markers",
        "marker": {
            "size": 5,
            "color": coords_map[unique_times_sorted[0]]["label"]
        }
    }

    fig_dict["data"].append(data_dict)

    for time in unique_times_sorted:
        name = time.strftime('%m/%d/%Y %H:%M')
        frame = {"data": [], "name": name}
        data_dict = {
            "x": coords_map[time]["x"],
            "y": coords_map[time]["y"],
            "text": coords_map[time]["text"],
            "mode": "markers",
            "marker": {
                "size": 5,
                "color": coords_map[time]["label"]
            }
        }
        frame["data"].append(data_dict)

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [name],
            {"frame": {"duration": 1000, "redraw": False},
            "mode": "immediate",
            "transition": {"duration": 1000}}
        ],
            "label": name,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig2 = go.Figure(fig_dict)

    map_plot = base64.b64encode(open(image_filename, 'rb').read())

    fig2.update_layout(
                    title = text,
                    images = [dict(
                        source='data:image/png;base64,{}'.format(map_plot.decode()),
                        xref="paper", yref="paper",
                        x=0, y=0,
                        sizex=1, sizey=1,
                        xanchor="left",
                        yanchor="bottom",
                        sizing="fill",
                        opacity=0.6,
                        layer="below")])

    fig2.show()



# Create agglomerative clustering heatmap
def create_heatmap(matrix, word_list, text):
    fig2 = ff.create_dendrogram(matrix, labels=word_list)
    for i in range(len(fig2['data'])):
        fig2['data'][i]['yaxis'] = 'y2'

    # Create Side Dendrogram
    dendro_side = ff.create_dendrogram(matrix, orientation='right')
    for i in range(len(dendro_side['data'])):
        dendro_side['data'][i]['xaxis'] = 'x2'

    # Add Side Dendrogram Data to Figure
    for data in dendro_side['data']:
        fig2.add_trace(data)

    dendro_leaves = dendro_side['layout']['yaxis']['ticktext']
    dendro_leaves = list(map(int, dendro_leaves))

    data_dist = pdist(matrix)
    heat_data = squareform(data_dist)
    heat_data = heat_data[dendro_leaves,:]
    heat_data = heat_data[:,dendro_leaves]

    print(heat_data)

    heatmap = [
        go.Heatmap(
            x = dendro_leaves,
            y = dendro_leaves,
            z = heat_data,
            colorscale = 'RdBu'
        )
    ]

    heatmap[0]['x'] = fig2['layout']['xaxis']['tickvals']
    heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']

    # Add Heatmap Data to Figure
    for data in heatmap:
        fig2.add_trace(data)

    fig2['layout']['yaxis']['ticktext'] = fig2['layout']['xaxis']['ticktext']
    fig2['layout']['yaxis']['tickvals'] = np.asarray(dendro_side['layout']['yaxis']['tickvals'])

    # Edit Layout
    fig2.update_layout({'width':800, 'height':800,
                            'showlegend':False, 'hovermode': 'closest',
                            })
    # Edit xaxis
    fig2.update_layout(xaxis={'domain': [.15, 1],
                                    'mirror': False,
                                    'showgrid': False,
                                    'showline': False,
                                    'zeroline': False,
                                    'ticks':""})
    # Edit xaxis2
    fig2.update_layout(xaxis2={'domain': [0, .15],
                                    'mirror': False,
                                    'showgrid': False,
                                    'showline': False,
                                    'zeroline': False,
                                    'showticklabels': False,
                                    'ticks':""})

    # Edit yaxis
    fig2.update_layout(yaxis={'domain': [0, .85],
                                    'mirror': False,
                                    'showgrid': False,
                                    'showline': False,
                                    'zeroline': False,
                                    'showticklabels': False,
                                    'ticks': ""
                            })
    # Edit yaxis2
    fig2.update_layout(yaxis2={'domain':[.825, .975],
                                    'mirror': False,
                                    'showgrid': False,
                                    'showline': False,
                                    'zeroline': False,
                                    'showticklabels': False,
                                    'ticks':""})

    fig2.update_layout(title = text)

    fig2.show()

    return fig2