import torch
import torch.nn.functional as F 
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import TransformerConv, TopKPooling 
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
torch.manual_seed(42)

class GNN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super(GNN, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        top_k_ratio = model_params["model_top_k_ratio"]
        self.top_k_every_n = model_params["model_top_k_every_n"]
        # dense_neurons = model_params["model_dense_neurons"]
        edge_dim = model_params["model_edge_dim"]

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.pooling_layers = ModuleList([])
        self.bn_layers = ModuleList([])

        # Transformation layer
        self.conv1 = TransformerConv(feature_size, 
                                    embedding_size, 
                                    heads=n_heads, 
                                    dropout=dropout_rate,
                                    edge_dim=edge_dim,
                                    beta=True) 

        self.transf1 = Linear(embedding_size*n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        if self.top_k_every_n == 0:
            self.top_k_every_n = 1

        # Other layers
        for i in range(self.n_layers):
            self.conv_layers.append(TransformerConv(embedding_size, 
                                                    embedding_size, #output size but per head so final result will be embedding_size*n_heads shaped
                                                    heads=n_heads, 
                                                    dropout=dropout_rate,
                                                    edge_dim=edge_dim,
                                                    beta=True))

            self.transf_layers.append(Linear(embedding_size*n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))
            if i % self.top_k_every_n == 0:
                self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio)) # TopKPooling layer drops nodes based on their importance, keeping top_k_ratio of them
            

    def forward(self, x, edge_attr, edge_index, batch_index):
        # Initial transformation
        # print(f"shape edge_index input transfConv: {edge_index.size()}")
        # print(f"shape edge_attr input transfConv: {edge_attr.size()}")
        # print(f"shape x input transfConv: {x.size()}")
        
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        if x.size(0) > 1:
            x = self.bn1(x)

        # Holds the intermediate graph representations
        global_representation = []
        transformer_edge_index=None
        attention_weights=None
        for i in range(self.n_layers):
            if i == 0: # First layer
                x,(transformer_edge_index, attention_weights) = self.conv_layers[i](x, edge_index, edge_attr,return_attention_weights=True)
                
                attention_scores_mean = torch.mean(attention_weights, dim=1)  # Risultato: (E)
             
            else:
                x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            if x.size(0) > 1:
                x = self.bn_layers[i](x)
            # Always aggregate last layer
            if i % self.top_k_every_n == 0 or i == self.n_layers: # Aggregate every top_k_every_n layers for pooling or the last layer for the final result
                x , edge_index, edge_attr, batch_index, _, _ = self.pooling_layers[int(i/self.top_k_every_n)]( #since we append pooling layers only when i % top_k_every_n == 0, we can use i/top_k_every_n as index for the current pooling layer
                    x, edge_index, edge_attr, batch_index # Pooling layers require edge_index and edge_attr in addition to x and batch_index
                    )
                
                # Add current representation
                global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1)) # Concatenate global max and mean pooling results along the batches and append to global_representation list for later use
               
        x = sum(global_representation) # Sum all global representations along dimension 0 so now the shape is [2*embedding_size]

        x = torch.relu(x) # 
        x = F.dropout(x, p=0.8, training=self.training)
        # print(f"shape x output transfConv: {x.size()}")
        # così facendo x dovrebbe essere di shape [batch_size, 2*embedding_size]
        return x,transformer_edge_index, attention_scores_mean

