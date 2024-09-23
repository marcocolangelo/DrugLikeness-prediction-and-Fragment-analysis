import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from dgl import batch
from my_code.TransfGNN import GNN


"""La classe MLP_revised serve a costruire un modello di rete neurale feedforward con layer nascosti e dropout
è il layer usato per analizzare le stringhe ECFP e alla fine come blocco di classificazione"""
class MLP_revised(nn.Module):
    def __init__(self,n_feature,n_hidden:list,n_output = 2,dropout=0.2):
        super(MLP_revised,self).__init__()

        assert isinstance(n_hidden,list),"n_hidden param must be the list"

        self.num_hidden_layers = len(n_hidden)
        self.layers = [n_feature] + n_hidden  # the layer list of NN except the output layer

        self.predict = nn.Sequential()
        self.predict.add_module('dropout_input',nn.Dropout(dropout)) # input_layers: dropout first
        for idx,(in_,out_) in enumerate(zip(self.layers[:-1],self.layers[1:])): 
            self.predict.add_module('linear{}'.format(idx),nn.Linear(in_,out_))  # add_module(dict): key -> nn.Module
            # self.predict.add_module('relu{}'.format(idx), nn.ReLU())
            # self.predict.add_module('bn{}'.format(idx),nn.BatchNorm1d(out_))
            # self.predict.add_module('ln{}'.format(idx), nn.LayerNorm(out_))
            self.predict.add_module('relu{}'.format(idx), nn.ReLU())
            self.predict.add_module('dropout'.format(idx),nn.Dropout(dropout))

        self.predict.add_module('output',nn.Linear(self.layers[-1],n_output))

    def forward(self,x):
        return self.predict(x)

class myModel(nn.Module):
    def __init__(self, feature_size, model_params):
        super(myModel, self).__init__()

        self.feature_size = feature_size
        self.embedding_size = model_params["model_embedding_size"]
        self.dropout_rate = model_params["model_dropout_rate"]
        # model_params2 = model_params.copy()
        # model_params2["model_edge_dim"] = 1

        self.GATencoder = GNN(feature_size=self.feature_size,      
                                           model_params=model_params)

        self.GATencoder_raw = GNN(feature_size=self.feature_size, model_params=model_params)
        
        
        # metto embedding_size*4 perchè voglio concatenatenare x_t e x_r che per quella che è la mia idea sono le rappresentazioni di GNN prima dei Linear layer
        # quindi x_t e x_r sono di dimensione embedding_size*2
        # quindi la dimensione della loro concatenazione sarà di embedding_size*4
        self.linear = nn.Linear(self.embedding_size * 4, self.embedding_size *2)

        # uso un layer lineare per riportare l'input di self.classify al self.embedding_size*2
        self.classify = MLP_revised(self.embedding_size * 2, [32],  # [64,32],[128]
                                    1, dropout=self.dropout_rate)  

   
        self.ln = nn.LayerNorm(self.embedding_size * 2)
        self.relu = nn.ReLU()
        self.mol_output = None


    def forward(self, data, device):
        # trees, raw = self.to_device(data, device)

        x_raw = data['mol_raws']['x']
        edge_index_raw = data['mol_raws']['edge_index']
        edge_attr_raw = data['mol_raws']['edge_attr']
        batch_index_raw = data['mol_raws']['batch']

        # Processing tree features (mol_trees)
        x_tree = data['mol_trees']['tree_node_features']
        edge_index_tree = data['mol_trees']['tree_edge_index']
        edge_attr_tree = data['mol_trees']['tree_edge_attr']
        batch_index_tree = data['mol_trees']['tree_batch_index']

       
        x_r,transformer_edge_index, attention_scores_mean = self.GATencoder_raw(x_raw, edge_attr_raw, edge_index_raw, batch_index_raw) 
       
        x_tree = x_tree.to(device)
        edge_attr_tree = edge_attr_tree.to(device)
        edge_index_tree = edge_index_tree.to(device)
        batch_index_tree = batch_index_tree.to(device)

        x_t,transformer_edge_index_tree, attention_scores_mean_tree = self.GATencoder(x_tree, edge_attr_tree, edge_index_tree, batch_index_tree)

        # print(f"scores: {attention_scores_mean_tree}")
        # print(f"att_scores_len: {len(attention_scores_mean_tree)}")

        x = torch.cat([x_t, x_r], dim=-1)

        #la concatenazione ha portato la dimensione a embedding_size*4 quindi la passo attraverso un layer lineare per riportarla a embedding_size*2
        x = self.linear(x)  #verifica però se la dimensione è giusta per la questione batch
        if x.size(0) > 1:
            x = self.ln(x)
        
        y = self.classify(self.relu(x))
        return y,transformer_edge_index, attention_scores_mean,transformer_edge_index_tree, attention_scores_mean_tree



    # def to_device(self, mol_batch, device):
    #     tree = mol_batch['mol_trees']
    #     raw = mol_batch['mol_raws']
        

    #     return tree, raw

    # def test_(self,tree,raw_h,device):
    #     assert len(tree) == len(raw_h)
    #     all_data = []
    #     for i in range(len(raw_h)):
    #         tt = tree[i].nodes_dict
    #         r = raw_h[i]
    #         cliques = []
    #         for key in tt:
    #             clique = tt[key]['clique']
    #             cliques.append(torch.sum(r[clique],dim=0))
    #         try:
    #             all_data.append(torch.stack(cliques,dim=0))
    #         except:
    #             print(tree[i].smiles)
    #             all_data.append(torch.sum(r[:],dim=0))
    #             return

    #     assert len(all_data) == len(tree)
    #     for i in range(len(tree)):
    #         tree[i].ndata['h'] = all_data[i].cpu()

    #     return batch(tree).to(device)       #forse da eliminare batch qui?
    
    # def test2_(self, tree, raw_x, batch_index, edge_attr_raw,edge_index_raw, device):
    #     num_graphs = batch_index.max().item() + 1  # Numero di grafi

    #     all_data = []
    #     global_batch_index = []
    #     global_edge_attr = []
    #     global_edge_index = []

    #     for i, single_tree in enumerate(tree):
    #         cliques = []
    #         edge_attr_tree = []
    #         edge_index_tree = []

    #         # Maschera per gli indici del batch corrente
    #         mask = (batch_index == i).to(device)
    #         local_ids = torch.arange(mask.sum().item(), device=device)

    #         for clique_idx, clique in single_tree.nodes_dict.items():
    #             clique_ids = clique['clique']
    #             global_ids = local_ids[clique_ids]

    #             # Somma le feature degli atomi nella clique
    #             clique_sum = torch.sum(raw_x[global_ids], dim=0)
    #             cliques.append(clique_sum)

    #             # Processa gli archi associati alla clique
    #             bonds = torch.tensor(clique['bonds'], device=device).t()
    #             for j in range(edge_index_raw.size(1)):
    #                 if torch.all(torch.eq(edge_index_raw[:, j].unsqueeze(1), bonds)).any():
    #                     edge_index_tree.append([clique_idx, clique_idx])
    #                     edge_attr_tree.append(edge_attr_raw[j])

    #         # Stack delle somme delle feature delle cliques
    #         tree_data = torch.stack(cliques, dim=0)
    #         all_data.append(tree_data)

    #         # Crea un batch_index globale per i nodi di questo albero
    #         global_batch_index.append(torch.full((tree_data.size(0),), i, dtype=torch.long, device=device))

    #         # Concatena edge_index e edge_attr per questo albero
    #         global_edge_index.append(torch.tensor(edge_index_tree, dtype=torch.long, device=device).t())
    #         global_edge_attr.append(torch.stack(edge_attr_tree))

    #     # Concatena tutti i dati dei nodi, batch_index, edge_index e edge_attr in un singolo tensore
    #     global_data = torch.cat(all_data, dim=0)
    #     global_batch_index = torch.cat(global_batch_index, dim=0)
    #     global_edge_index = torch.cat(global_edge_index, dim=1)
    #     global_edge_attr = torch.cat(global_edge_attr, dim=0)

    #     return global_data, global_batch_index, global_edge_index, global_edge_attr





        

