import plotly.graph_objects as go
import numpy as np

A = np.random.randn(30).reshape((15, 2))
centroids = np.random.randint(10, size=10).reshape((5, 2))
clusters = [1, 2, 3, 4, 5]
colors = ['red', 'green', 'blue', 'yellow', 'magenta']

fig = go.Figure(
    data=[go.Scatter(x=A[:3][:,0],
                     y=A[:3][:,1],
                     mode='markers',
                     name='cluster 1',
                     marker_color=colors[0]),
          go.Scatter(x=[centroids[0][0]],
                     y=[centroids[0][1]],
                     mode='markers',
                     name='centroid of cluster 1',
                     marker_color=colors[0],
                     marker_symbol='x')
         ],
    layout=go.Layout(
        xaxis=dict(range=[-10, 10], autorange=False),
        yaxis=dict(range=[-10, 10], autorange=False),
        title="Start Title",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None]),
                     dict(label="Pause",
                          method="animate",
                          args=[None,
                               {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}}],
                         )])]
    ),
    frames=[
    go.Frame(
    data=[go.Scatter(x=A[:3][:,0],
                     y=A[:3][:,1],
                     mode='markers',
                     name='cluster 1',
                     marker_color=colors[0]),
          go.Scatter(x=[centroids[0][0]],
                     y=[centroids[0][1]],
                     mode='markers',
                     name='centroid of cluster 1',
                     marker_color=colors[0],
                     marker_symbol='x')
         ]),
    go.Frame(
        data=[
            go.Scatter(x=A[:3][:,0],
                       y=A[:3][:,1],
                       mode='markers',
                       name='cluster 2',
                       marker_color=colors[1]),
            go.Scatter(x=[centroids[1][0]],
                       y=[centroids[1][1]],
                       mode='markers',
                       name='centroid of cluster 2',
                       marker_color=colors[1],
                       marker_symbol='x')
        ]),
    go.Frame(
        data=[
            go.Scatter(x=A[3:5][:,0],
                       y=A[3:5][:,1],
                       mode='markers',
                       name='cluster 3',
                       marker_color=colors[2]),
            go.Scatter(x=[centroids[2][0]],
                       y=[centroids[2][1]],
                       mode='markers',
                       name='centroid of cluster 3',
                       marker_color=colors[2],
                       marker_symbol='x')
        ]),
    go.Frame(
        data=[
            go.Scatter(x=A[5:8][:,0],
                       y=A[5:8][:,1],
                       mode='markers',
                       name='cluster 4',
                       marker_color=colors[3]),
        go.Scatter(x=[centroids[3][0]],
                   y=[centroids[3][1]],
                   mode='markers',
                   name='centroid of cluster 4',
                   marker_color=colors[3],
                   marker_symbol='x')]),
    go.Frame(
        data=[
            go.Scatter(x=A[8:][:,0],
                       y=A[8:][:,1],
                       mode='markers',
                       name='cluster 5',
                       marker_color=colors[4]),
            go.Scatter(x=[centroids[4][0]],
                       y=[centroids[4][1]],
                       mode='markers',
                       name='centroid of cluster 5',
                       marker_color=colors[4],
                       marker_symbol='x')
        ]),
    ])
            
fig.show()