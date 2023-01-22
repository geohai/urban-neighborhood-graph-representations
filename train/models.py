import torch
import torch.nn as nn

# holds the node embeddings
class NodeEmbeddings(nn.Module):
    def __init__(self, num_nodes, embedding_dim=200):
        super(NodeEmbeddings, self).__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim)
    
    def forward(self, node_idx):
        # a positive or negative word is passed
        z1 = self.node_embeddings(node_idx)

        # TODO: do i need activation here?
        return z1
    