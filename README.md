# regional-representations-graph-model
Creating a graph model for urban representation learning. The model is based off the one described in this paper: [M3G](http://cs.emory.edu/~lzhao41/venues/DeepSpatial2021/papers/M3G__Deepspatial_2021.pdf).

This is a graph model that uses triplet loss / contrastive learning to train node embeddings. These embeddings are available for nodes represented in the dataset, but new nodes can be mapped to the existing embedding space. I will add this functionality to the code later.

In this model, nodes represent regions and edges represent relations between regions. Currently, the data modalities supported include any image and text data (related to the node) and any quantifiable relationship between nodes (such as spatial distance or human mobility).
This implementation includes InceptionV3 to extract features from images and Glove embeddings to extract text embedding. I was experimenting with Pytorch Geometric and created a graph for the edge portion of code, but this was totally unnecessary and I didn't use any of the built in functionality. Perhaps later if needed I can add more functionality for using PyG and triplet loss training. 

The training process is sequential (back-to-back stages). First, the embeddings are trained on each data modality for the nodes and then the edges.
