# How to run

## Prerequisites

- Python 3
- Scikit learn
- Numpy
- Gensim
- Plotly and dash
- Networkx
- nltk
- Pandas
- Kaleido
- PyLDAvis (only for `vis_lda.py`)

## To train model

1. Fetch Moby Dick (e.g. from Gutenberg Project), place it under ``research_vis``, name it "Moby_Dick.txt", add the following text before chapter 1 "!!!!!!!!!!!!START HERE!!!!!!!!!!!!!!!!!!", and the following text after epilogue "!!!!!!!!!!!!END HERE!!!!!!!!!!!!!!!!!!"
2. Run `train_model.py` and ``train_model_doc2vec.py``, this will generate models and text files containing the name of the model

## To run visualizations

1. Extract tf_idf/tf_idf.zip and run ``wordcloud_precalc.py`` (alternatively, run ``tf_idf.py``) *Note:* If running on Windows, change ``./tf_idf/result_`` to ``./tf_idf\\result_`` on line 93 if you get an error
2. Go to research_vis directory and run any ``vis_`` file

 - ``vis_embeddings_doc2vec.py`` will start a plotly dash server containing a spherical and 2D graph of the doc2vec embeddings of each chapter. Clicking on a node will bring up a word cloud of that chapter with the words that have the highest tf-idf for that chapter. UMAP and DBSCAN parameters can be changed in the UI, and the recalculation can be triggered with the button. This computation will most likely take some time
 - ``vis_characters.py`` will visualize the number of times each character is named in each chapter. *Note:* Requires running ``tf_idf.py`` beforehand
 - ``vis_embeddings_doc2vec_1d.py`` will visualize the embeddings reduced to 1 dimension to the y-axis, with the x-axis as the number of the chapter
 - ``vis_embeddings_doc2vec.py`` will start a plotly dash server containing word2vec embeddings reduced with UMAP and PCA, UMAP parameters can be changed in the UI, and the recalculation can be triggered with the button. This computation will most likely take some time 
 - - ``vis_lda.py`` will create an HTML file that contains LDA visualization created with PyLDAvis


### Special requirements for OpenNRE

- Python 3.8.6
- Numpy version 1.2.1
- Protobuf version 3.20.*
(it is recommended to run this in virtualenv)