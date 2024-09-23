import concurrent.futures
import pandas as pd
import torch
from torch_geometric.data import Dataset
import os
from tqdm import tqdm
import deepchem as dc
from rdkit import Chem
from deepchem.feat.graph_data import GraphData
from my_code.tree_building import DGLMolTree
import numpy as np

class ExtendedGraphData(GraphData):
    def __init__(self, node_features, edge_index, edge_features, mol_tree):
        super().__init__(
            node_features=node_features,
            edge_index=edge_index,
            edge_features=edge_features
        )
        self.mol_tree = mol_tree  # Aggiungi l'albero come proprietà

    def to_pyg_graph(self):
        # Converti il grafo in un formato utilizzabile da PyTorch Geometric
        data = super().to_pyg_graph()
        data.mol_tree = self.mol_tree  # Includi l'albero nel grafo PyG
        return data


class MoleculeDataset(Dataset):
    def __init__(self, root, filename, test=False,final = False, transform=None, pre_transform=None):
        self.test = test
        self.filename = filename
        self.final = final
        super(MoleculeDataset, self).__init__(root, transform, pre_transform)
        

    @property
    def raw_file_names(self):
        return self.filename

    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)] + \
                   [f'tree_data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)] + \
                   [f'tree_data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process_molecule(self, index, row_tuple):
        _, row = row_tuple  # row_tuple è una tupla (index, row)
        data = {}

        # Se il SMILES contiene un punto, non è valido
        if "." in str(row["smiles"]):
            return
        
        # Estrai le features della molecola
        mol = Chem.MolFromSmiles(row["smiles"])
        featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)
        f = featurizer._featurize(mol)
        data = f.to_pyg_graph()
        if self.final == False:
            data.y = self._get_label(row["Drug"])
        data.smiles = row["smiles"]

        # Creazione dell'albero basato sulla decomposizione della molecola
        tt,tree = self.crea_features_albero(data.smiles, data.x,data.edge_attr,data.edge_index)

        
        # Costruisci il grafo dell'albero
        try: 
            tree_graph = ExtendedGraphData(
                node_features=np.asarray(tt['node_features']),
                edge_index=np.asarray(tt['edge_index'], dtype=int),
                edge_features=np.asarray(tt['edge_attr'], dtype=float),
                mol_tree=tree
            ).to_pyg_graph()
        except:
            print(f"smiles: {data.smiles}")
            print(f"edge_features: {tt['edge_attr']}")
            print(f"shape edge_features: {tt['edge_attr'].size()}")
            exit(1)
            

        # Salva i dati e l'albero in due file separati per poterli caricare separatamente in seguito
        data_filename = f'data_{index}.pt' if not self.test else f'data_test_{index}.pt'
        tree_filename = f'tree_data_{index}.pt' if not self.test else f'tree_data_test_{index}.pt'

        torch.save(data, os.path.join(self.processed_dir, data_filename))
        torch.save(tree_graph, os.path.join(self.processed_dir, tree_filename))


    def process(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        
        # Parallelizzazione con ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(self.process_molecule, self.data.index, self.data.iterrows()), total=self.data.shape[0]))

    def _get_label(self, label):
        return torch.tensor([label], dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
            data_filename = f'data_{idx}.pt' if not self.test else f'data_test_{idx}.pt'
            tree_filename = f'tree_data_{idx}.pt' if not self.test else f'tree_data_test_{idx}.pt'
            
            # Carica entrambi i grafi
            molecule_data = torch.load(os.path.join(self.processed_dir, data_filename))
            tree_data = torch.load(os.path.join(self.processed_dir, tree_filename))

            return molecule_data, tree_data

    def crea_features_albero(self, smiles, raw_x,raw_edge_attr,raw_edge_index):
        if "." in smiles:
            print("Codifica Smiles non valida")
            return None,None
        tree = DGLMolTree(smiles)
        # if smiles == 'C1OCOCO1':
        #     print("C1OCOCO1")
        tree.add_mol_edge_features(raw_edge_index,raw_edge_attr)
        cliques = []
        for clique_idx, clique in tree.nodes_dict.items():
            clique_ids = clique['clique']
            clique_sum = torch.mean(raw_x[clique_ids], dim=0)
            cliques.append(clique_sum)

        if not cliques:
            print(f"Nessuna clique estratta da smiles: {smiles}")
            
        tree_data = torch.stack(cliques, dim=0)
        tree_edge_index = tree.get_edge_index()
        tree_edge_attr = tree.get_edge_attr()

        # Se non ci sono archi, restituisci i archi vuoti
        if tree_edge_index.size(0) == 0 or tree_edge_index.size(1) == 0:
            tree_features = {
            'node_features': tree_data,
            'edge_index': torch.empty([2, 0]),
            'edge_attr':torch.empty([0, 11]),
        }
        else:   
            
            tree_features = {
                'node_features': tree_data,
                'edge_index': tree_edge_index,
                'edge_attr': tree_edge_attr,
        }

        #print shapes
        # print(f"raw_x_shape: {raw_x.size()}")
        # print(f"raw_edge_index_shape: {raw_edge_index.size()}")
        # print(f"raw_edge_attr_shape: {raw_edge_attr.size()}")
        
        # print("\n")

        # print(f"tree node_features_shape: {tree_data.size()}")
        # print(f"tree edge_index_shape: {tree_edge_index.size()}")
        # print(f"tree edge_attr_shape: {tree_edge_attr.size()}")
        # print("\n")

        
        return tree_features,tree
