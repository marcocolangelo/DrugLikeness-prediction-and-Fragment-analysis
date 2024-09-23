try:
    from my_code.tree_building import DGLMolTree
except:
    from tree_building import DGLMolTree
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import Draw


def combine_molecules_with_shared_atom(mol1, mol2, new_shared_atom_idx_1, new_shared_atom_idx_2):
    """
    Combina due molecole fondendo gli atomi comuni invece di duplicarli.
    
    Args:
    - mol1: Prima molecola come oggetto RDKit (Mol o RWMol).
    - mol2: Seconda molecola come oggetto RDKit (Mol o RWMol).
    - new_shared_atom_idx_1: Indice dell'atomo condiviso nella prima molecola.
    - new_shared_atom_idx_2: Indice dell'atomo condiviso nella seconda molecola.
    
    Returns:
    - combined_mol: Molecola risultante dalla fusione delle due molecole.
    """
    
    # Converte entrambe le molecole in RWMol per permettere modifiche
    mol1_rw = Chem.RWMol(mol1)
    mol2_rw = Chem.RWMol(mol2)
    # mol2_rw.RemoveAtom(shared_atom_idx_1)

    # Combina le molecole mantenendo il riferimento degli atomi
    combined_mol = Chem.CombineMols(mol1_rw, mol2_rw)
    combined_mol_rw = Chem.RWMol(combined_mol)



    # Elimina l'atomo duplicato nella molecola combinata
    combined_mol_rw.RemoveAtom(new_shared_atom_idx_1)
    new_shared_atom_idx_2 = new_shared_atom_idx_2 - 1

    for neighbor in mol1_rw.GetAtomWithIdx(new_shared_atom_idx_1).GetNeighbors():
        neighbor_idx = neighbor.GetIdx() 
        bond_type = mol1_rw.GetBondBetweenAtoms(new_shared_atom_idx_1, neighbor_idx).GetBondType()
        if neighbor_idx > new_shared_atom_idx_1:
            neighbor_idx = neighbor_idx - 1
        if neighbor_idx != new_shared_atom_idx_2:
            combined_mol_rw.AddBond(new_shared_atom_idx_2, neighbor_idx, bond_type)
        else:
            return None


    # Solo per la stampa a fine di debug
    for x in combined_mol_rw.GetAtoms(): 
         print(x.GetIdx(), x.GetSymbol())


    # Sanitizza la molecola per garantire che sia valida
    try:
        Chem.SanitizeMol(combined_mol_rw)
    except:
        return combined_mol_rw
    
    return combined_mol_rw

def combine_molecules_no_shared_atom(mol1, mol2, atom_idx_1, atom_idx_2, bond_type=Chem.BondType.SINGLE):
    """
    Combina due molecole aggiungendo un legame tra due atomi di frontiera senza atomi condivisi.
    
    Args:
    - mol1: Prima molecola come oggetto RDKit (Mol o RWMol).
    - mol2: Seconda molecola come oggetto RDKit (Mol o RWMol).
    - atom_idx_1: Indice dell'atomo di frontiera nella prima molecola.
    - atom_idx_2: Indice dell'atomo di frontiera nella seconda molecola.
    - bond_type: Tipo di legame da creare (default: legame singolo).
    
    Returns:
    - combined_mol: Molecola risultante dalla combinazione delle due molecole.
    """
    
    # Converte entrambe le molecole in RWMol per permettere modifiche
    mol1_rw = Chem.RWMol(mol1)
    mol2_rw = Chem.RWMol(mol2)

    # Combina le molecole mantenendo il riferimento degli atomi
    combined_mol = Chem.CombineMols(mol1_rw, mol2_rw)
    combined_mol_rw = Chem.RWMol(combined_mol)
    
    # Calcola il nuovo indice dell'atomo nella seconda molecola
    new_atom_idx_2 = atom_idx_2 + mol1_rw.GetNumAtoms()
    

    # Aggiunge il legame tra gli atomi di frontiera
    combined_mol_rw.AddBond(atom_idx_1, new_atom_idx_2, bond_type)

    # Sanitizza la molecola per garantire che sia valida
    Chem.SanitizeMol(combined_mol_rw)

    return combined_mol_rw



def unify_frags(tree, idx_tree_node1, idx_tree_node2):
    tree_nodes = tree.nodes_dict

    smiles1 = tree_nodes[idx_tree_node1]['smiles']
    smiles2 = tree_nodes[idx_tree_node2]['smiles']

    print(f"smiles1: {smiles1}")
    print(f"smiles2: {smiles2}")

    atom_indices1 = tree_nodes[idx_tree_node1]['clique']
    atom_indices2 = tree_nodes[idx_tree_node2]['clique']

    shared_atom = None
    for i in range(len(atom_indices1)):
        for j in range(len(atom_indices2)):
            if atom_indices1[i] == atom_indices2[j]:
                shared_atom = atom_indices1[i]

                new_shared_atom_id_1 = shared_atom - min(atom_indices1)
                new_shared_atom_id_2 = shared_atom - min(atom_indices2) + len(atom_indices1) - 1
                break

    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)

    new_mol = None

    if shared_atom is not None:
        new_mol= combine_molecules_with_shared_atom(mol1, mol2, new_shared_atom_id_1,new_shared_atom_id_2)

        # aggiungi condizione per quando abbiamo un collegamento tree che prevede legame tra due atomi dello stesso nodo
        #......................


    else:
        (frontier1, frontier2) = tree.mol_edge_map[(idx_tree_node1, idx_tree_node2)]
        bond_type = tree.mol.GetBondBetweenAtoms(frontier1, frontier2).GetBondType()
        new_mol = combine_molecules_no_shared_atom(mol1, mol2, frontier1, frontier2, bond_type=bond_type)
    if new_mol is not None:
        return new_mol
    else:
        return None
        
    
    
smiles = 'Cc1ccc(N)cc1-c1nnnn1C1CC1'
original_mol = Chem.MolFromSmiles(smiles)
Draw.MolToImage(original_mol).show()
tree = DGLMolTree(smiles)


# Combina i tensori in una matrice [2, E]
tree_edges = tree.all_edges()
print(tree_edges)

node1 = 4
node2 = 7

mol_edge_map = tree.mol_edge_map

print(mol_edge_map)

try:
    
    frag_mol = unify_frags(tree, node1, node2)
    if frag_mol is not None:
        print(Chem.MolToSmiles(frag_mol))
        Draw.MolToImage(frag_mol).show()
    else:
        print("Tentativo di loop, frammenti non validi")

        
except:
    print("Fragmenti non validi,concatenazione")
    tree_nodes = tree.nodes_dict
    smiles1 = tree_nodes[node1]['smiles']
    smiles2 = tree_nodes[node2]['smiles']
    smiles_tot = smiles2  + smiles1
    frag_mol = Chem.MolFromSmiles(smiles_tot)
    Draw.MolToImage(frag_mol).show()




# fragment = original_mol.GetSubstructMatches(new_mol)[0]
# original_mol.


# # Esempio di utilizzo:
# # Carichiamo due molecole semplici dai loro SMILES
# mol1 = Chem.MolFromSmiles(smiles1)  # Etanolo
# mol2 = Chem.MolFromSmiles(smiles2)  # Dimetilammina



# (atom_idx_1,atom_idx_2) = mol_edge_map[(idx_tree_node1,idx_tree_node2)]

# print(f"a1: {atom_idx_1}, a2: {atom_idx_2}")

# # Supponiamo di voler unire l'atomo di ossigeno (indice 2 in mol1) con l'atomo di azoto (indice 0 in mol2)
# combined_molecule = combine_molecules(mol1, mol2, atom_idx_1=atom_idx_1, atom_idx_2=atom_idx_2, bond_type=Chem.BondType.SINGLE)

# # Stampa il risultato in formato SMILES
# print(Chem.MolToSmiles(combined_molecule))

# # Disegna la molecola risultante
# from rdkit.Chem import Draw
# Draw.MolToImage(combined_molecule).show()
