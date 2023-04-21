# regional-representations-graph-model
Creating a graph model for urban representation learning. 

This is a graph model that uses triplet loss / contrastive learning to train node embeddings. These embeddings are available for nodes represented in the dataset, but new nodes can be mapped to the existing embedding space. I will add this functionality to the code later.

In this model, nodes represent regions and edges represent relations between regions. Currently, the data modalities supported include any image and text data (related to the node) and any quantifiable relationship between nodes (such as spatial distance or human mobility).

The training process is sequential (back-to-back stages). First, the embeddings are trained on each data modality for the nodes and then the edges.

Lots of things to do.
