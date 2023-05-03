# regional-representations-graph-model
Creating and using a graph model for urban neighborhood representation learning. 

This repo contains a graph model that uses triplet loss / contrastive learning to train node embeddings. 

In this model, nodes represent regions. There are two types of data: node data, which is intra-neighborhood information, and edge data, which represents relations between neighborhoods. 

### Contents
- Dataset
  - The dataset directory contains code to download / retrieve data that I use to train the model. These are mostly scripts to query various APIs, but there are also some exploratory Jupyter notebooks to check the data. Some examples are the Google Earth Engine API, Yelp Fusion API, Mapillary API, and the U.S. Census API.
- Train
  - The train directory contains PyTorch deep learning code including dataset classes, model classes, and training loops. I need to clean it up a bit and make a config file.
- Evaluate
  - The evaluate directory contains Jupyter notebooks that I use to evaluate the node embeddings. There is code for various urban prediction tasks (both regression and classification) such as predicting Census variables, CDC health outcomes, point-of-interest category distribution. There is also a notebook for visualizing aspects of the model to inspect the learning process.
- Train on Remote Sensing Imagery
  - This directory contains code to Fine-Tune models on remote sensing datasets. This aims to improve the feature extraction from satellite imagery to give "better" information to the graph model.

The training process is sequential (back-to-back stages). First, the embeddings are trained on each data modality for the nodes and then the edges.

Lots of things to do.
