from collections import deque
from rdkit import Chem
from rdkit.Chem import Draw
try:
    from my_code.tree_building import DGLMolTree
except:
    from tree_building import DGLMolTree

def bfs_connected_high_fragments(tree, start_nodes, high_fragment):
    """
    Cerca tutti i nodi collegati a partire dai nodi iniziali che fanno parte della lista high_fragment.
    
    Args:
    - tree: Oggetto DGLMolTree, rappresenta l'albero molecolare con un attributo 'edges' che contiene gli archi.
    - start_nodes: Lista di nodi iniziali da cui avviare la ricerca.
    - high_fragment: Lista dei frammenti (nodi) che sono di alto interesse.
    
    Returns:
    - connected_fragments: Set di nodi che fanno parte della lista high_fragment e sono collegati ai nodi iniziali.
    """
    # Set per tracciare i nodi visitati
    visited = set()

    # Set per contenere i frammenti connessi trovati
    connected_fragments = set()

    # Coda per la BFS
    queue = deque(start_nodes)

    # Costruisci una mappa dei vicini da tree.edges
    neighbors_map = {}
    edges = tree.mol_edge_map.keys()
    for (start, end) in edges:
        if start not in neighbors_map:
            neighbors_map[start] = []
        if end not in neighbors_map:
            neighbors_map[end] = []
        neighbors_map[start].append(end)
        neighbors_map[end].append(start)

    # Inizia la BFS
    while queue:
        node = queue.popleft()
        
        # Se abbiamo già visitato questo nodo, lo saltiamo
        if node in visited:
            continue

        # Marca il nodo come visitato
        visited.add(node)

        # Se il nodo è nella lista high_fragment, lo aggiungiamo
        if node in high_fragment:
            connected_fragments.add(node)

        # Espandiamo la ricerca a tutti i vicini del nodo corrente
        neighbors = neighbors_map.get(node, [])  # Ottieni i vicini del nodo
        for neighbor in neighbors:
            # Aggiungiamo il vicino alla coda solo se non è stato visitato e se è nella lista high_fragment
            if neighbor not in visited and neighbor in high_fragment:
                queue.append(neighbor)

    return connected_fragments


def extract_substructure_preserving_rings(mol, atom_indices):
    """
    Estrae una sottostruttura da una molecola preservando gli anelli, la stereochimica e altre proprietà atomiche.
    
    Args:
    - mol: Molecola RDKit da cui estrarre la sottostruttura.
    - atom_indices: Lista degli indici atomici da includere nella sottostruttura.
    
    Returns:
    - submol: Sottostruttura come oggetto RDKit.
    """
    # Crea un nuovo oggetto RWMol per permettere modifiche
    rw_mol = Chem.RWMol()

    # Mappa tra indici atomici della molecola originale e quelli del nuovo submol
    atom_map = {}

    # Aggiungi gli atomi specificati, preservando proprietà come stereochimica e carica
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        new_atom = Chem.Atom(atom.GetSymbol())
        new_atom.SetFormalCharge(atom.GetFormalCharge())
        new_atom.SetChiralTag(atom.GetChiralTag())  # Preserva la configurazione chirale
        new_atom.SetHybridization(atom.GetHybridization())  # Preserva l'ibridazione
        new_atom.SetNumExplicitHs(atom.GetNumExplicitHs())  # Preserva il numero di idrogeni espliciti
        new_idx = rw_mol.AddAtom(new_atom)
        atom_map[idx] = new_idx

    # Aggiungi i legami tra gli atomi specificati, preservando proprietà come ordine e stereochimica
    for bond in mol.GetBonds():
        start_atom = bond.GetBeginAtomIdx()
        end_atom = bond.GetEndAtomIdx()
        
        # Aggiungi il legame solo se entrambi gli atomi sono nella lista specificata
        if start_atom in atom_map and end_atom in atom_map:
            new_bond_type = bond.GetBondType()
            rw_mol.AddBond(atom_map[start_atom], atom_map[end_atom], new_bond_type)

            # Preserva la stereochimica del legame (es. legami E/Z)
            stereo = bond.GetStereo()
            if stereo != Chem.BondStereo.STEREONONE:
                rw_mol.GetBondBetweenAtoms(atom_map[start_atom], atom_map[end_atom]).SetStereo(stereo)

    # Sanitizza la molecola per verificarne la validità
    try:
        Chem.SanitizeMol(rw_mol)
    except ValueError:
        return rw_mol

    # Rimuovi gli idrogeni espliciti se presenti
    submol = Chem.RemoveHs(rw_mol)
    
    return submol


def unify_fragments(tree,fragments,high_fragments = None,deep_search = False):
    """
    Unisce frammenti ad alta rilevanza dalla struttura ad albero della molecola e restituisce una sottostruttura.
    
    Args:
    - tree: Struttura ad albero della molecola.
    - high_score_fragments_for_graph: Lista di frammenti ad alta rilevanza.
    
    Returns:
    - submol: Molecola combinata risultante dai frammenti uniti.
    """
    tree_nodes = tree.nodes_dict
    connected_fragments = None

    # Avvia la BFS a partire dai due nodi
    assert high_fragments is not None or deep_search == False, "Se si vuole fare una ricerca approfondita, i frammenti di alta rilevanza devono essere specificati"
       

    if deep_search == True and high_fragments is not None:
        connected_fragments = bfs_connected_high_fragments(tree, fragments, high_fragments)
        fragments = connected_fragments

    id_mol = set()
    for i in fragments:
        clique = tree_nodes[i]['clique']
        id_mol.update(clique)

    submol = extract_substructure_preserving_rings(tree.mol, id_mol)

    return submol,connected_fragments


# Esempio di utilizzo:
if __name__ == "__main__":
    # SMILES di esempio
    smiles = 'NC(=O)N[C@@H](Cc1ccc(O)cc1)C(=O)O'
    original_mol = Chem.MolFromSmiles(smiles)
    # Draw.MolToImage(original_mol).show()
    tree = DGLMolTree(smiles)
    print(tree.nodes_dict)
    print("/n")
    print(tree.mol_edge_map)
    high_frags = [10,12,3,11]
    frags = [5,10]

    #trasforma frags in tupla
    (clique1,clique2) = tuple(frags)
    assert tree.has_edges_between(clique1,clique2) , "I frammenti devono essere nodi dell'albero"

    submol,connected_frags = unify_fragments(tree, frags,high_frags, deep_search=True)

    print("Frammenti connessi:")
    print(connected_frags)

    # Visualizza il SMILES della sottostruttura estratta
    print(Chem.MolToSmiles(submol))
    Draw.MolToImage(submol).show()

