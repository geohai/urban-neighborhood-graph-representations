import torch
import torch.nn as nn

# holds the node embeddings
class NodeEmbeddings(nn.Module):
    def __init__(self, num_nodes, embedding_dim=200):
        super(NodeEmbeddings, self).__init__()
        self.node_embeddings = nn.Embedding(num_nodes, embedding_dim, max_norm=1)
        print(self.node_embeddings.weight)

        # self.linear = nn.Linear(
        #     in_features=embedding_dim,
        #     out_features=embedding_dim,
        # )
    
    def forward(self, node_idx):
        # a positive or negative word is passed
        z1 = self.node_embeddings(node_idx)

        # TODO: do i need activation here?
        return z1

    def return_embedding_by_idx(self, node_idx):
        return self.node_embeddings(node_idx)
    
    def print_history(self):
        print(self.node_embeddings.weight.grad)
    