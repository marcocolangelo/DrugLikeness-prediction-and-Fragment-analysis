import pandas as pd
import copy,os
import numpy as np
from dgl import DGLGraph
import rdkit.Chem as Chem
from tqdm import tqdm

from .chemutils import get_clique_mol, tree_decomp, get_mol, get_smiles

"""mol_tree.py
Contenuto: Definisce la struttura dati dell'albero molecolare e il vocabolario per le molecole.
Funzioni Principali:
Vocab: Classe che gestisce il vocabolario delle molecole, mappando le stringhe SMILES a indici numerici."""

"""get_Vocab(data_path)
Descrizione: Genera un vocabolario di sottostrutture molecolari (cliques) da un file CSV contenente stringhe SMILES.
Input: Percorso del file CSV.
Output: File vocabulary_<data_name>.txt contenente le sottostrutture uniche."""
def get_Vocab(data_path = None): # Write the functional clusters of the molecules after decomposition using the tree to the glossary vocabulary.txt
    assert data_path is not None ,"Path to data must be required,plz check"
    assert isinstance(data_path,str), "Path to data must be string"

    if not os.path.exists(data_path):
        raise ValueError("Path to data is not found")

    # data_name = os.path.split(data_path.strip(".csv"))[-1]
    data_name = os.path.split(data_path)[-1].split('.')[0]
    output_path = './dataset' + '/vocabulary_' + data_name + '.txt'
    data=pd.read_csv(data_path)['Smiles']

    print('To get the vocabulary of {}'.format(data_name))
    result=set(())
    for i,smiles in enumerate(tqdm(data)):
        temp=DGLMolTree(smiles)
        for key in temp.nodes_dict:
            result.update([temp.nodes_dict[key]['smiles']]) 

    print('\n\tGet Vocabulary Finished!')

    with open(output_path,"w") as f:
        for csmiles in result:
            f.writelines(csmiles+"\n") #write the smiles to the file vocabulary.txt and save each of them in a new line

"""Descrizione: Simile a get_Vocab, ma prende un DataFrame come input invece di un file CSV.
Input: Nome del dataset, DataFrame con le stringhe SMILES, percorso del file di output.
Output: File vocabulary_<data_name>.txt."""
def get_Vocab_df(data_name,df,output_path = None): # Scrivere i gruppi funzionali delle molecole dopo la decomposizione utilizzando l'albero del glossario vocabulary.txt

    if output_path is None:
        output_path = './dataset' + '/vocabulary_' + data_name + '.txt'

    data=df['Smiles']

    print('To get the vocabulary of {}'.format(data_name))
    result=set(())
    for i,smiles in enumerate(tqdm(data)):
        temp=DGLMolTree(smiles)
        for key in temp.nodes_dict:
            result.update([temp.nodes_dict[key]['smiles']])
        # print('\r{}/{} smiles to get cliques vocabulary..'.format(i + 1, len(data)), end='')

    print('\n\tGet Vocabulary Finished!')

    with open(output_path,"w") as f:
        for csmiles in result:
            f.writelines(csmiles+"\n")


def get_slots(smiles):   #Obtaining atomic signatures of functional blocks
    mol = Chem.MolFromSmiles(smiles)
    return [(atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()) for atom in mol.GetAtoms()]

"""Vocab: Classe che gestisce il vocabolario delle molecole, mappando le stringhe SMILES a indici numerici.
Metodi
__init__(self, smiles_list): Inizializza il vocabolario con una lista di stringhe SMILES.
get_index(self, smiles): Restituisce l'indice di una stringa SMILES nel vocabolario.
get_smiles(self, idx): Restituisce la stringa SMILES corrispondente a un indice.
get_slots(self, idx): Restituisce le caratteristiche degli atomi di un gruppo funzionale dato un indice.
size(self): Restituisce la dimensione del vocabolario."""
class Vocab(object):   #Functional group vocabulary hash function to build a word list file (txt): hash function for index and content
    def __init__(self, smiles_list):
        self.vocab = smiles_list
        self.vmap = {x:i for i,x in enumerate(self.vocab)} 
        # self.slots = [get_slots(smiles) for smiles in self.vocab]
        self.slots = None
        
    def get_index(self, smiles):
        if smiles not in self.vmap:
            print(f'{smiles} not in vocab.')

        return self.vmap.get(smiles,0)  # get the smiles. else unknow

    def get_smiles(self, idx):
        return self.vocab[idx]

    def get_slots(self, idx):
        return copy.deepcopy(self.slots[idx])

    def size(self):
        return len(self.vocab)

"""Un DGLGraph è una struttura dati fornita dalla libreria Deep Graph Library (DGL), 
progettata per rappresentare grafi in modo efficiente e per facilitare l'implementazione e l'addestramento di modelli di deep learning su grafi. """
"""Nel contesto di MSSGAT, un DGLGraph viene utilizzato per rappresentare le molecole come grafi, dove gli atomi sono nodi e i legami chimici sono archi. 
Questa rappresentazione consente di applicare modelli di attenzione sui grafi (Graph Attention Networks, GAT) 
per apprendere rappresentazioni significative delle molecole, che possono poi essere utilizzate per predire proprietà molecolari.

Viene usata in get_batches e quindi anche in multi_process per rapprsentare le molecole come grafi prima del training"""
class DGLMolTree(DGLGraph):
    def __init__(self, smiles):
        DGLGraph.__init__(self)
        self.nodes_dict = {}
        self.warning = False

        if smiles is None:
            print("\n\n\n\n#############SMILES string is None#############\n\n\n\n")
            return
        else:
            print("molecole: ", smiles, "\n\n\n\n")

        self.smiles = smiles
        self.mol = get_mol(smiles)

        # cliques: a list of list of atom indices
        # edges: a list of list of edge by atoms src and dst
        cliques, edges = tree_decomp(self.mol)

        root = 0
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            if cmol is None:
                print("C'è una clique anomala nella molecola: ", smiles, " con clique: ", c, "\n\n\n\n")
                self.warning = True
                return
            csmiles = get_smiles(cmol)
            self.nodes_dict[i] = dict(
                smiles=csmiles,
                mol=get_mol(csmiles),
                clique=c,
            )
            if min(c) == 0: # if the clique contains the atom with index 0 so the root atom
                root = i #then the root is the index of this clique

        self.add_nodes(len(cliques)) #remember that DGLMolTree is a subclass of DGLGraph so it has the add_nodes method

        # The clique with atom ID 0 becomes root
        if root > 0:
            for attr in self.nodes_dict[0]:
                self.nodes_dict[0][attr], self.nodes_dict[root][attr] = self.nodes_dict[root][attr], self.nodes_dict[0][attr] #swap the root atom with the atom with index 0

        # Add edges following the breadth-first order in clique tree decomposition (edges are bi-directional so we add both directions and to do this we add the edges twice so here is explained why we have 2 * len(edges) edges)
        src = np.zeros((len(edges) * 2,), dtype='int')
        dst = np.zeros((len(edges) * 2,), dtype='int')
        for i, (_x, _y) in enumerate(edges):
            x = 0 if _x == root else root if _x == 0 else _x
            y = 0 if _y == root else root if _y == 0 else _y
            src[2 * i] = x #2 is caused by the fact that we add the edges twice because they are bi-directional 
            dst[2 * i] = y
            src[2 * i + 1] = y
            dst[2 * i + 1] = x

        self.add_edges(src, dst) #remember that DGLMolTree is a subclass of DGLGraph so it has the add_edges method


    def treesize(self):
        return self.number_of_nodes()

