from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import io
import cairosvg
import matplotlib.patches as mpatches
from PIL import Image
import torch

from my_code.tree_building import DGLMolTree
from my_code.unify_high_mol import unify_fragments

def colors_higlight(attention_scores_normalized,edge_index):

    # Supponiamo di avere già edge_index, attention_weights e una molecola (mol)
    # edge_index: Tensor di dimensione (2, E)
    # attention_weights: Tensor di dimensione (E, H) -> useremo la media su H
    # print(f"shape di edge_index: {edge_index.shape}")
    # print(f"shape di attention_scores_normalized: {attention_scores_normalized.shape}")


    #per ogni elemento in edge_index, se il valore è maggiore di pred_edge_len, sottrai pred_edge_len
    #edge_index = edge_index - pred_edge_len

    # Passo 3: Converti edge_index in una lista di tuple (sorgente, destinazione)
    edges = edge_index.t().tolist()[1::2]  # Ottieni una lista di coppie (sorgente, destinazione)
    # print(f"shape di edges: {len(edges)}")

    # Passo 4: Crea una mappa dei colori basata sui punteggi di attenzione
    # Qui usiamo un semplice mapping lineare dei colori (da bianco a rosso)
    colors = {}
    for i, (start, end) in enumerate(edges):
        score = attention_scores_normalized[i].item()
        
        # Assegna il colore in base al valore di score
        if score < 0.50:
            color = (1.0, 1.0, 1.0)
        elif 0.50 <= score <= 0.60:
            color = (1.0, 1.0, 0.0)  # Giallo
        elif 0.61 <= score <= 0.80:
            color = (1.0, 0.65, 0.0) # Arancione
        elif 0.81 <= score <= 1.0:
            color = (1.0, 0.0, 0.0)  # Rosso
        else:
            color = (0.5, 0.5, 0.5)  # Grigio per valori fuori range (opzionale)

        colors[(start, end)] = color  # Assegna il colore
    return edges, colors

def draw_molecule_with_attention(mol, edges, colors, attention_scores_normalized):
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    opts = drawer.drawOptions()
    
    num_atoms = mol.GetNumAtoms()  # Ottieni il numero totale di atomi nella molecola
    
    # Creiamo una lista per gli indici dei legami da evidenziare e un dizionario per i colori
    highlight_bonds = []
    highlight_bond_colors = {}
    bonds = mol.GetBonds()
    #stampa id atomi del bond
    # for bond in bonds:
        # print(f"ID atomi del bond: {bond.GetBeginAtomIdx()} {bond.GetEndAtomIdx()}")

    for i, (start, end) in enumerate(edges):
        # Assicurati che start sia sempre minore o uguale a end
        if start > end:
            start, end = end, start
        
        # Controlla se gli indici sono validi
        if start >= num_atoms or end >= num_atoms:
            continue
        
        bond = mol.GetBondBetweenAtoms(start, end)
        
    
        # Controlla se il legame esiste
        if bond is None:
            continue
        
        bond_idx = bond.GetIdx()
        
        # Aggiungi il legame agli highlight
        highlight_bonds.append(bond_idx)
        highlight_bond_colors[bond_idx] = tuple(colors.get((start, end), colors.get((end, start))))

    # Disegna la molecola con i legami evidenziati
    drawer.DrawMolecule(mol,highlightAtoms=[], highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg



def higlight_mol(attention_scores_normalized, edge_index, smiles,id,final=False):
    # Passo 6: Visualizza l'immagine
    mol = Chem.MolFromSmiles(smiles)
    edges, colors = colors_higlight(attention_scores_normalized, edge_index)
    svg = draw_molecule_with_attention(mol, edges, colors, attention_scores_normalized)
    
    # Converti SVG a PNG
    png_data = cairosvg.svg2png(bytestring=svg)
    
    # Crea un'immagine PIL dalla stringa PNG
    image = Image.open(io.BytesIO(png_data))
    
    # Crea la figura combinata con matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(image)
    ax.axis('off')  # Rimuovi gli assi

    # Aggiungi la legenda

    yellow_patch = mpatches.Patch(color=(1.0, 1.0, 0.0), label='0.50 - 0.60')
    orange_patch = mpatches.Patch(color=(1.0, 0.65, 0.0), label='0.61 - 0.80')
    red_patch = mpatches.Patch(color=(1.0, 0.0, 0.0), label='0.81 - 1.00')

    ax.legend(handles=[yellow_patch, orange_patch, red_patch], 
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Mostra l'immagine con la legenda
    if final == True:
        plt.savefig(f'data/test_data/TCMBANK/images/mol_raw/mol_{id}_final.png')
    else:
        plt.savefig(f'data/test_data/images/mol_raw/mol_{id}.png')
    # plt.show()
    plt.close()


####### TREE DECOMPOSITION ########

# def map_tree_nodes_to_molecule(tree_node_scores, tree_to_molecule_mapping):
#     # tree_node_scores: Lista o array contenente gli attention scores per ogni nodo dell'albero
#     # tree_to_molecule_mapping: Dizionario o lista che mappa ogni nodo dell'albero a uno o più nodi della molecola
#     molecule_node_scores = {}

#     for tree_node, molecule_nodes in tree_to_molecule_mapping.items():
#         score = tree_node_scores[tree_node]
#         for molecule_node in molecule_nodes:
#             if molecule_node not in molecule_node_scores:
#                 molecule_node_scores[molecule_node] = 0

#             if molecule_node_scores[molecule_node] < score:
#                 molecule_node_scores[molecule_node] = score   # Distribuisce uniformemente se ci sono più nodi

#     return molecule_node_scores

def map_tree_edges_to_molecule(tree_edge_index, tree_to_molecule_mapping):
    """
    Crea una mappatura tra gli edge dell'albero e gli edge della molecola.
    Per ogni edge nell'albero (tra clique), si ottiene il set di edge corrispondenti nella molecola.
    """
    molecule_edge_mapping = {}
    

    for i, (start, end) in enumerate(tree_edge_index.t().tolist()):
       
        if len(tree_to_molecule_mapping) == 1:
            try:
                molecule_edge_mapping[(start, end)] = tree_to_molecule_mapping[start]
                break
            except Exception as e:
                print(f"Errore: {e}")
                break
            

        clique_start_bonds = tree_to_molecule_mapping[start]
        clique_end_bonds = tree_to_molecule_mapping[end]
        
        # Trova tutti i possibili archi molecolari tra i nodi della clique_start e clique_end
        # molecule_edges = [(a, b) for a in clique_start for b in clique_end]

        #concatena le liste di bonds semplicemente con concatenzaione di liste
        molecule_edges = clique_start_bonds + clique_end_bonds
        molecule_edge_mapping[(start, end)] = molecule_edges

    return molecule_edge_mapping

# def draw_molecule_with_node_attention(mol, molecule_node_scores):
#     drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
#     opts = drawer.drawOptions()
    
#     highlight_atoms = []
#     atom_colors = {}

#     for atom_idx, score in molecule_node_scores.items():
#         highlight_atoms.append(atom_idx)
        
#         if score < 0.50:
#             color = (1.0, 1.0, 1.0)
#         elif 0.50 <= score <= 0.60:
#             color = (1.0, 1.0, 0.0)  # Giallo
#         elif 0.61 <= score <= 0.80:
#             color = (1.0, 0.65, 0.0) # Arancione
#         elif 0.81 <= score <= 1.0:
#             color = (1.0, 0.0, 0.0)  # Rosso
#         else:
#             color = (0.5, 0.5, 0.5)  # Grigio per valori fuori range (opzionale)
        
#         atom_colors[atom_idx] = color
    
#     # Disegna la molecola con i nodi evidenziati
#     drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=atom_colors)
#     drawer.FinishDrawing()
#     svg = drawer.GetDrawingText()
#     return svg

def draw_molecule_with_edge_attention(mol, tree_edge_index,molecule_edge_mapping, tree_edge_attention_scores, alpha=1.0):
    """
    Disegna la molecola evidenziando i legami (bonds) in base ai punteggi di attenzione dell'albero
    """
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    opts = drawer.drawOptions()

    num_atoms = mol.GetNumAtoms()  # Ottieni il numero totale di atomi nella molecola

    # Creiamo una lista per gli indici dei legami da evidenziare e un dizionario per i colori
    highlight_bonds = []
    highlight_bond_colors = {}

    edge_index = 0

    for (tree_start, tree_end), molecule_edges in molecule_edge_mapping.items():
        score = tree_edge_attention_scores[edge_index]
        
        if score < 0.50:
            color = (1.0, 1.0, 1.0, alpha)
        elif 0.50 <= score <= 0.60:
            color = (1.0, 1.0, 0.0, alpha)  # Giallo
        elif 0.61 <= score <= 0.80:
            color = (1.0, 0.65, 0.0, alpha)  # Arancione
        elif 0.81 <= score <= 1.0:
            color = (1.0, 0.0, 0.0, alpha)  # Rosso
        else:
            color = (0.5, 0.5, 0.5, alpha)  # Grigio per valori fuori range

        # Applica il colore a ciascun legame molecolare associato all'arco dell'albero
        for start, end in molecule_edges:
            if start > end:
                start, end = end, start

            # Controlla se gli indici sono validi
            if start >= num_atoms or end >= num_atoms:
                continue

            
            bond = mol.GetBondBetweenAtoms(start, end)
            assert bond is not None, f"Edge {start}-{end} does not exist in molecule"
            bond_idx = bond.GetIdx()

            highlight_bonds.append(bond_idx)
            highlight_bond_colors[bond_idx] = color
        
        edge_index += 1

    # Disegna la molecola con i legami evidenziati
    drawer.DrawMolecule(mol, highlightAtoms=[], highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

def highlight_molecule_with_tree_edge_attention(smiles, tree_edge_index, attention_scores_mean_tree, tree_to_molecule_mapping, id,final = False):
    mol = Chem.MolFromSmiles(smiles)
    
    # Mappa gli edge dell'albero ai corrispondenti edge della molecola
    molecule_edge_mapping = map_tree_edges_to_molecule(tree_edge_index, tree_to_molecule_mapping)

    # Disegna la molecola evidenziando gli edge in base ai pesi di attenzione sugli edge dell'albero
    svg = draw_molecule_with_edge_attention(mol,tree_edge_index, molecule_edge_mapping, attention_scores_mean_tree)

    # Converti SVG a PNG
    png_data = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(png_data))
    
    # Crea la figura combinata con matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(image)
    ax.axis('off')  # Rimuovi gli assi

    # Aggiungi la legenda
    yellow_patch = mpatches.Patch(color=(1.0, 1.0, 0.0), label='0.50 - 0.60')
    orange_patch = mpatches.Patch(color=(1.0, 0.65, 0.0), label='0.61 - 0.80')
    red_patch = mpatches.Patch(color=(1.0, 0.0, 0.0), label='0.81 - 1.00')

    ax.legend(handles=[yellow_patch, orange_patch, red_patch], 
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Mostra l'immagine con la legenda
    if final == True:
        plt.savefig(f'data/test_data/TCMBANK/images/tree_decomp/mol_{id}_final.png')
    else:
        plt.savefig(f'data/test_data/images/tree_decomp/mol_{id}.png')

    plt.close()

def highlight_molecule_with_high_fragments(smiles,high_score_fragments,medium_score_fragments,tree_to_molecule_mapping, id,final = False):
    mol = Chem.MolFromSmiles(smiles)

    high_score_atom_bonds = set()

    if len(tree_to_molecule_mapping) > 1:
        for f in high_score_fragments:
            high_score_atom_bonds.update(tree_to_molecule_mapping[f])

        medium_score_fragments = set()
        for f in medium_score_fragments:
            medium_score_fragments.update(tree_to_molecule_mapping[f])
    else:
        high_score_atom_bonds = tree_to_molecule_mapping[0]
        medium_score_fragments = []
        
    if len(high_score_atom_bonds) == 0 and len(medium_score_fragments) == 0:
        return False

    svg = draw_molecule_with_high_fragments(mol,high_score_atom_bonds,medium_score_fragments)

    # Converti SVG a PNG
    png_data = cairosvg.svg2png(bytestring=svg)
    image = Image.open(io.BytesIO(png_data))
    
    # Crea la figura combinata con matplotlib
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(image)
    ax.axis('off')  # Rimuovi gli assi

    # Aggiungi la legenda
    yellow_patch = mpatches.Patch(color=(1.0, 1.0, 0.0), label='0.50 - 0.60')
    red_patch = mpatches.Patch(color=(1.0, 0.0, 0.0), label='0.81 - 1.00')

    ax.legend(handles=[yellow_patch, red_patch], 
              bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Mostra l'immagine con la legenda
    if final == True:
        plt.savefig(f'data/test_data/TCMBANK/images/tree_decomp/mol_{id}_final.png')
    else:
        plt.savefig(f'data/test_data/images/tree_decomp/mol_{id}.png')
    plt.close()
    return True

def colorize_bonds(mol,molecule_edges,num_atoms,highlight_bonds,highlight_bond_colors,color):
        (start, end) = molecule_edges
        if start > end:
            start, end = end, start

        # Controlla se gli indici sono validi
        assert not(start >= num_atoms or end >= num_atoms)
            
        
        bond = mol.GetBondBetweenAtoms(start, end)
        assert bond is not None, f"Edge {start}-{end} does not exist in molecule"
        bond_idx = bond.GetIdx()

        highlight_bonds.append(bond_idx)
        highlight_bond_colors[bond_idx] = color
                

def draw_molecule_with_high_fragments(mol,high_score_atom_bonds,medium_score_fragments,alpha=1.0):
    """
    Disegna la molecola evidenziando i legami (bonds) in base ai frammenti estratti e al loro score
    """
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    opts = drawer.drawOptions()

    num_atoms = mol.GetNumAtoms()  # Ottieni il numero totale di atomi nella molecola

    # Creiamo una lista per gli indici dei legami da evidenziare e un dizionario per i colori
    highlight_bonds = []
    highlight_bond_colors = {}

    for edge in high_score_atom_bonds:
        color = (1.0, 0.0, 0.0, alpha)  # Rosso
        # Applica il colore a ciascun legame molecolare associato all'arco dell'albero
        colorize_bonds(mol,tuple(edge),num_atoms,highlight_bonds=highlight_bonds,highlight_bond_colors=highlight_bond_colors,color=color)
          
        
    for edge in medium_score_fragments:
        color = (1.0, 1.0, 0.0, alpha)  # Giallo
        colorize_bonds(mol,tuple(edge),num_atoms,highlight_bonds=highlight_bonds,highlight_bond_colors=highlight_bond_colors,color=color)

    # Disegna la molecola con i legami evidenziati
    drawer.DrawMolecule(mol, highlightAtoms=[], highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg

# def highlight_molecule_with_tree_attention(smiles, transformer_edge_index_tree, attention_scores_mean_tree, tree_to_molecule_mapping,id):
#     mol = Chem.MolFromSmiles(smiles)
    
#     # Mappa gli scores dei nodi dell'albero ai nodi della molecola
#     molecule_node_scores = map_tree_nodes_to_molecule(attention_scores_mean_tree, tree_to_molecule_mapping)
    
#     # Disegna la molecola con gli attention scores proiettati sui nodi molecolari
#     svg = draw_molecule_with_node_attention(mol, molecule_node_scores)
    
#     # Converti SVG a PNG
#     png_data = cairosvg.svg2png(bytestring=svg)
#     image = Image.open(io.BytesIO(png_data))
    
#     # Crea la figura combinata con matplotlib
#     fig, ax = plt.subplots(figsize=(12, 6))
#     ax.imshow(image)
#     ax.axis('off')  # Rimuovi gli assi

#     # Aggiungi la legenda

#     yellow_patch = mpatches.Patch(color=(1.0, 1.0, 0.0), label='0.50 - 0.60')
#     orange_patch = mpatches.Patch(color=(1.0, 0.65, 0.0), label='0.61 - 0.80')
#     red_patch = mpatches.Patch(color=(1.0, 0.0, 0.0), label='0.81 - 1.00')

#     ax.legend(handles=[yellow_patch, orange_patch, red_patch], 
#               bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

#     # Mostra l'immagine con la legenda
#     plt.savefig(f'data/test_data/images/tree_decomp/mol_{id}.png')




###### HIGHLIGHT MOL PER GRAPH RACCHIUDE ENTRAMBE LE AGGREGAZIONI  ---- COMMENTATA PERCHE' SOTTO SCRITTA VERSIONE PIU ELEGANTE   ########


def highlight_mol_per_graph(raw_edge_index, raw_attention_scores,tree_edge_index, tree_attention_scores, batch,tree_batch,prev_batch_num_graphs,deep_search = False,deep_search_threshold = 0.79,final=False):
    # Ottieni il numero di grafi nel batch
    node_batch = batch.batch # batch.batch è un tensore che assegna a ciascun nodo il grafo di appartenenza tramite un indice
    tree_node_batch = tree_batch.batch
    num_graphs = node_batch.max().item() + 1 #numero di grafi presenti nel batch
    high_score_fragments = []

    for i in range(num_graphs):
        index = prev_batch_num_graphs + i
        smiles = batch.smiles[i]

        # Seleziona i nodi e gli edge appartenenti al grafo i-esimo
        node_mask = (node_batch == i)
        edge_mask = node_mask[raw_edge_index[0]] & node_mask[raw_edge_index[1]]
        if edge_mask.sum() == 0:
            continue

        # Estrai gli indici locali dei nodi e edge_index corrispondente in batch
        local_edge_index = raw_edge_index[:, edge_mask] - node_mask.nonzero(as_tuple=True)[0].min()
        local_attention_scores = raw_attention_scores[edge_mask]

        # Seleziona i nodi e gli edge appartenenti all'albero i-esimo 
        tree_node_mask = (tree_node_batch == i)
        tree_edge_mask = tree_node_mask[tree_edge_index[0]] & tree_node_mask[tree_edge_index[1]]
        if tree_edge_mask.sum() == 0:
            continue

        # Estrai gli indici locali dei nodi e edge_index corrispondente in tree_batch
        tree_local_edge_index = tree_edge_index[:, tree_edge_mask] - tree_node_mask.nonzero(as_tuple=True)[0].min()
        tree_local_attention_scores = tree_attention_scores[tree_edge_mask]

        # Mappa i nodi dell'albero ai nodi della molecola
        tree_to_molecule_mapping = {}
        nodes_dict = tree_batch.mol_tree[i].nodes_dict
        tree_edges = tree_batch.mol_tree[i].all_edges()

        for tree_node_idx, node_data in nodes_dict.items():
            raw_bonds = node_data['bonds']  # Ottieni la lista degli indici degli atomi associati a questo nodo
            tree_to_molecule_mapping[tree_node_idx] = raw_bonds



        ###### Cattura di frammenti ad alta rilevanza ######

        # Inizializza variabili per la ricerca approfondita
        high_score_fragments_for_graph = None
        medium_score_fragments_for_graph = None
        visited_nodes = set()

        # if deep_search == True search all the clique connected trhough tree_local_attention_scores[j] > 0.79
        if deep_search == True:
            high_score_fragments_for_graph = set()
            medium_score_fragments_for_graph = set()
            #prendi gli edge con attenzione maggiore di 0.79
            for j in range(len(tree_local_attention_scores)):
                if tree_local_attention_scores[j] > deep_search_threshold:
                    start = tree_local_edge_index[0][j].item()
                    end = tree_local_edge_index[1][j].item()
                    if start == end:
                        continue
                    high_score_fragments_for_graph.add(start)
                    high_score_fragments_for_graph.add(end)

                if tree_local_attention_scores[j] <= deep_search_threshold and tree_local_attention_scores[j] >= 0.50:
                    start = tree_local_edge_index[0][j].item()
                    end = tree_local_edge_index[1][j].item()
                    if start == end:
                        continue
                    medium_score_fragments_for_graph.add(start)
                    medium_score_fragments_for_graph.add(end)
        ###### Fine cattura di frammenti ad alta rilevanza ######



       ######### PARTE PER ESTRARRE FRAMMENTI CON ATTENZIONE MAGGIORE DI 0.79 ########## 
       #riempi high_score_edges con gli edge che hanno attenzione maggiore di 0.79
        for j in range(len(tree_local_attention_scores)):
            if tree_local_attention_scores[j] > deep_search_threshold:

                if len(tree_edges[0]) == 0: #se non ci sono edge, non ci sono frammentI ma tutta la molecola è un frammento
                    high_score_fragments.append((smiles,0))
                    continue

                #prendi le coppie di nodi associate a questo edge
                start = tree_local_edge_index[0][j].item()
                end = tree_local_edge_index[1][j].item()

                if start == end: #se start == end, non ci sono frammenti
                    continue
                

                #estrai numero di frammenti con cui ciascun frammento è connesso
                #conta numero di occorrenze di start in bonds
                first_elements = tree_edges[0]
                start_count = torch.sum(first_elements == start).item() - 1 #-1 perchè da considerare il link tra start e end
                end_count = torch.sum(first_elements == end).item() - 1 
                #assert per cui start_bonds o end_bonds sono 0 è possibile solo quando tree_edges == 1, ovvero tra start e end
                assert start_count >= 0 and end_count >= 0, "start_bonds e end_bonds non possono essere negativi"
                assert (start_count != 0 or end_count != 0) or int(len(tree_edges[0])/2) == 1, "almeno uno tra start_bonds e end_count non può essere 0 se ci sono più di un edge"
                

                try:
                    if deep_search == True:
                        if start not in visited_nodes and end not in visited_nodes:
                            final_mol,connected_high_score_frags = unify_fragments(tree_batch.mol_tree[i], [start, end], high_score_fragments_for_graph, deep_search=deep_search)
                            visited_nodes.update(connected_high_score_frags)
                        else:
                            continue
                    else:
                        final_mol,_ = unify_fragments(tree_batch.mol_tree[i], [start, end], high_score_fragments_for_graph, deep_search=deep_search)

                except Exception:
                    #lancia eccezione se non è possibile unire i due frammenti
                    print(f"Errore durante l'unione dei frammenti {start} e {end} di smiles {smiles}")
                    exit(-1)

                # Sanifica molecola prima di convertirla in SMILES
                assert final_mol is not None, "Problema con estrazione frammenti"
                try:
                    Chem.SanitizeMol(final_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except Exception as e:
                    print(f"Errore di sanitizzazione: {e}")
                    continue
                final_smiles = Chem.MolToSmiles(final_mol)

                #conta numero di frammenti con cui ciascun frammento è connesso
                neig_count = 0
                if deep_search == False:
                    neig_count = start_count + end_count
                else:
                    for frag in connected_high_score_frags:
                        neig_count += torch.sum(first_elements == frag).item() - 1
                    
                
                final_smiles_num_neig = (final_smiles,neig_count,index)
                high_score_fragments.append(final_smiles_num_neig)

        ######### FINE PARTE PER ESTRARRE FRAMMENTI CON ATTENZIONE MAGGIORE DI 0.79 ##########



        # Visualizza la molecola associata a questo grafo
        
        higlight_mol(local_attention_scores, local_edge_index,smiles,index,final)
        # la terza condizione è per gestire quei casi con molecole particolari che vengono scomposte in modo errato
        if deep_search == False or len(high_score_fragments) == 0 :
            highlight_molecule_with_tree_edge_attention(smiles, tree_local_edge_index, tree_local_attention_scores, tree_to_molecule_mapping, index,final)
        #versione modificata di highlight_molecule_with_tree_edge_attention che usa high_score_fragments_for_graph
        else:
            # ho aggiunto un controllo per vedere se la funzione ritorna False, significa che non si è riusciti a catturare alcun arco, in tal caso chiamo la funzione per evidenziare gli edge dell'albero
            res = highlight_molecule_with_high_fragments(smiles,high_score_fragments_for_graph,medium_score_fragments_for_graph,tree_to_molecule_mapping, index,final)
           

        #a causa del batch size, i nodi e gli edge dei grafi successivi sono spostati di un certo numero di posizioni per cui dobbiamo tenerne conto
        # print(f"len(local_edge_index) :{len(local_edge_index)}")
        # print(f"edge mask len {len(edge_mask)}")
        # #numero elementi di edge mask che sono True
        # print(f"edge mask sum {edge_mask.sum()}")

    
    return batch.num_graphs,high_score_fragments



# def get_local_indices(batch, raw_edge_index, graph_idx):
#     """
#     Estrae gli indici locali per i nodi e gli edge appartenenti a un grafo specifico.
#     """
#     node_mask = (batch.batch == graph_idx)
#     edge_mask = node_mask[raw_edge_index[0]] & node_mask[raw_edge_index[1]]
#     if edge_mask.sum() == 0:
#         return None, None  # Nessun edge per questo grafo
#     local_edge_index = raw_edge_index[:, edge_mask] - node_mask.nonzero(as_tuple=True)[0].min()
#     return local_edge_index, edge_mask


# def extract_high_attention_edges(tree_local_edge_index, tree_local_attention_scores, threshold=0.79):
#     """
#     Estrae gli edge che hanno un'attenzione maggiore di una soglia specificata.
#     """
#     high_score_edges = []
#     for j in range(len(tree_local_attention_scores)):
#         if tree_local_attention_scores[j] > threshold:
#             start = tree_local_edge_index[0][j].item()
#             end = tree_local_edge_index[1][j].item()
#             if start != end:  # Evita auto-loop
#                 high_score_edges.append((start, end))
#     return high_score_edges


# def process_high_attention_fragments(tree, high_score_edges, deep_search, visited_nodes, high_score_fragments_for_graph, tree_edges, index):
#     """
#     Processa gli edge ad alta attenzione e unisce i frammenti. Restituisce i frammenti, il numero di vicini (neig_count) e l'indice.
#     """
#     high_score_fragments = []
#     for start, end in high_score_edges:
#         try:
#             # Unisci i frammenti con deep_search o senza
#             if deep_search:
#                 if start not in visited_nodes and end not in visited_nodes:
#                     final_mol, connected_high_score_frags = unify_fragments(tree, [start, end], high_score_fragments_for_graph, deep_search=deep_search)
#                     visited_nodes.update(connected_high_score_frags)
#                 else:
#                     continue
#             else:
#                 final_mol = unify_fragments(tree, [start, end], high_score_fragments_for_graph, deep_search=deep_search)[0]

#             # Sanifica molecola
#             Chem.SanitizeMol(final_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
#             final_smiles = Chem.MolToSmiles(final_mol)

#             # Conta i vicini
#             first_elements = tree_edges[0]
#             neig_count = 0

#             if deep_search == False:
#                 # Se deep_search è disattivato, conta solo i vicini di start e end
#                 start_count = torch.sum(first_elements == start).item() - 1
#                 end_count = torch.sum(first_elements == end).item() - 1
#                 neig_count = start_count + end_count
#             else:
#                 # Se deep_search è attivato, conta i vicini di tutti i frammenti connessi
#                 for frag in connected_high_score_frags:
#                     neig_count += torch.sum(first_elements == frag).item() - 1

#             high_score_fragments.append((final_smiles, neig_count, index))
        
#         except Exception as e:
#             print(f"Errore durante l'unione dei frammenti {start} e {end}: {e}")
#             continue

#     return high_score_fragments


# def higlight_mol_per_graph(raw_edge_index, raw_attention_scores, tree_edge_index, tree_attention_scores, batch, tree_batch, prev_batch_num_graphs, deep_search=False):
#     """
#     Funzione principale che processa ciascun grafo nel batch per trovare frammenti ad alta rilevanza.
#     """
#     num_graphs = batch.batch.max().item() + 1
#     high_score_fragments = []

#     for i in range(num_graphs):
#         index = prev_batch_num_graphs + i
#         smiles = batch.smiles[i]

#         # Estrarre indici locali per nodi e edge
#         local_edge_index, edge_mask = get_local_indices(batch, raw_edge_index, i)
#         if local_edge_index is None:
#             continue

#         tree_local_edge_index, tree_edge_mask = get_local_indices(tree_batch, tree_edge_index, i)
#         if tree_local_edge_index is None:
#             continue

#         local_attention_scores = raw_attention_scores[edge_mask]
#         tree_local_attention_scores = tree_attention_scores[tree_edge_mask]

#         # Mappa tra i nodi dell'albero e la molecola
#         tree_to_molecule_mapping = {tree_node_idx: node_data['bonds'] for tree_node_idx, node_data in tree_batch.mol_tree[i].nodes_dict.items()}

#         # Inizializza variabili per la ricerca approfondita
#         high_score_fragments_for_graph = set()
#         visited_nodes = set()

#         if deep_search:
#             high_score_edges = extract_high_attention_edges(tree_local_edge_index, tree_local_attention_scores)
#             for start, end in high_score_edges:
#                 high_score_fragments_for_graph.update([start, end])

#         # Estrai frammenti con attenzione > 0.79
#         high_score_edges = extract_high_attention_edges(tree_local_edge_index, tree_local_attention_scores)
#         high_score_fragments.extend(process_high_attention_fragments(tree_batch.mol_tree[i], high_score_edges, deep_search, visited_nodes, high_score_fragments_for_graph, tree_batch.mol_tree[i].all_edges(), index))

#         # Visualizza la molecola associata a questo grafo
#         higlight_mol(local_attention_scores, local_edge_index, smiles, index)
#         highlight_molecule_with_tree_edge_attention(smiles, tree_local_edge_index, tree_local_attention_scores, tree_to_molecule_mapping, index)

#     return batch.num_graphs, high_score_fragments