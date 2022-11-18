# Requirements

- Python 3
- Scikit learn
- Gensim
- Plotly
- Networkx

# How to run

1. **Important**: Fetch the Microblogs.csv dataset, and put it under MC_1_Materials_3-30-2011. It is not included in the repo due to its size.

2. Run `python word2vec_train.py`

This trains the word2vec model with the Microblogs dataset to create the embeddings, and saves the model. In UNIX, it should be saved under the current directory.

Optional: Run `python word2vec_split training.py`

This splits the dataset into two categories, pre-outbreak and post-outbreak, and trains two models respectively. These are used in some visualizations.


3. Run `python word2vec_filter.py`

This runs a basic heuristic filter through the dataset and saves it in filtered_coords.txt and filtered2.txt. These are used in visualizations.

## To visualize a bar chart with the infections
Run `python word2vec_vis.py`

## To visualize the map with messages that contain symptom-related terms
Run `python word2vec_vis_map.py`

## To train a binary classifier, and visualize the data
Run `python word2vec_binary_classification.py`

When run for the first time, splits the data into multiple categories and selects the training data. **This will take some time.**

In subsequent runs, it will skip creating the training data, and instead trains the classifier immediately.

## To create a visualization of pairwise similarities

Run `python word2vec_pairwise.py`

This will create two heatmaps, first is the similarity between the words. Second is the similarity of similarities, or the cosine similarity between embedding vectors.

## To visualize the embedding distances

Run `python word2vec_dimensionality_reduction.py`

This will create a 2D graph of the distances between the embeddings.