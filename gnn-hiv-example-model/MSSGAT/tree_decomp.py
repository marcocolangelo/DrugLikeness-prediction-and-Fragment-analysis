
from collections import defaultdict
from scipy.sparse import csr_matrix
import rdkit.Chem as Chem
from scipy.sparse.csgraph import minimum_spanning_tree

"""Formation of a Minimum Spanning Tree (MST) formed by cliques"""
def tree_decomp(mol):

    MST_MAX_WEIGHT = 100  #The maximum weight of an edge in the Minimum Spanning Tree (MST) formed by cliques - it's used to ensure that the MST doesn't include edges that are too heavy.
    #the weight of an edge in the MST is used during the process of forming the MST to ensure that the MST doesn't include edges that are too heavy.
    n_atoms = mol.GetNumAtoms() 
    if n_atoms == 1:  
        return [[0]], []

    cliques = [] 

    """It then iterates over all bonds in the molecule. 
    For each bond, it gets the indices of the beginning and end atoms. 
    If the bond is not part of a ring, it adds the pair of atoms to the cliques list."""
    for bond in mol.GetBonds(): 
        a1 = bond.GetBeginAtom().GetIdx() 
        a2 = bond.GetEndAtom().GetIdx() 
        if not bond.IsInRing(): 
            cliques.append([a1, a2])

    """In the context of graph theory and chemistry, a ring is a cycle in the graph that represents a molecule, 
    and the SSSR is a set of rings such that no ring in the set is a combination of other rings in the set. 
    In other words, it's the smallest set of the smallest possible rings that can be found in the molecular structure."""
    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)]  
    cliques.extend(ssr)  

#for each atom in the molecule, it creates an empty list in the nei_list list.
#then it iterates over all cliques in the cliques list.
#finally it iterates over all atoms in the current clique and appends the index of the clique to the list of neighbors of the atom.
    nei_list = [[] for i in range(n_atoms)]  #nei stands for NEighbors Indexes
    for i in range(len(cliques)):  
        for atom in cliques[i]: 
            nei_list[atom].append(i)    #nei_list is a list of lists where the i-th list contains the indexes of the cliques that contain the i-th atom

#now the function wants to 
    edges = defaultdict(int)
    for atom in range(n_atoms):  
        if len(nei_list[atom]) <= 1: #if a
            continue
        #cnei (clicque neighbors) is a list of the indexes of the cliques that contain the current atom.
        cnei = nei_list[atom]  
        bonds = [c for c in cnei if len(cliques[c]) == 2] #bonds is a list of the indexes of the cliques that contain only two atoms.
        rings = [c for c in cnei if len(cliques[c]) > 4]  #rings is a list of the indexes of the cliques that contain more than four atoms.
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): # if the number of bonds is greater than 2 or if there are two bonds and more than two neighbors for the current atom
            cliques.append([atom])  #then it means that the atom is a branching point and it adds the atom to the cliques list. where a branching point (punto di ramificazione) is a point where the molecule branches off into two or more directions.
            c2 = len(cliques) - 1 #c2 is the index of the current atom, so the new clique
            for c1 in cnei:  
                edges[(c1, c2)] = 1 #add and edge between the current atom and the new clique
        elif len(rings) > 2:  #if the number of rings in which the current atom is contained is greater than 2
            cliques.append([atom]) #then it means that the atom is a branching point and it adds the atom to the cliques list.
            c2 = len(cliques) - 1 #c2 is the index of the current atom, so the new clique
            for c1 in cnei: 
                """foundamental part: we sfavorite the breaking points between the rings assigning a weight of MST_MAX_WEIGHT - 1 to the edge
                this helps to preserve the rings in the MST because we represent the rings as single structures instead of groups of single atoms"""
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1 
        else: #if the current atom is not a branching point
            """In summary, this part is ensuring that the weight of each edge in the edges dictionary accurately reflects 
            the current number of atoms shared by the two cliques it connects."""
            for i in range(len(cnei)): #for each clique that contains the current atom
                for j in range(i + 1, len(cnei)): 
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2]) #inter is the set of atoms that are in both cliques
                    if edges[(c1, c2)] < len(inter): #if the number of atoms in the intersection is greater than the number of atoms in the intersection of the two cliques
                        #This could happen if the function is being called multiple times and the molecule structure is changing, or if there's some non-determinism in the order or way the cliques are processed.
                        edges[(c1, c2)] = len(inter)  #fix the dictionary value 

    """ha lo scopo di trasformare gli edge del grafo per preparare la costruzione dell'albero di spanning minimo (MST).
    Trasformazione:
    La trasformazione u + (MST_MAX_WEIGHT - v,):
    u è una coppia di indici dei cliques (es. (c1, c2)).
    (MST_MAX_WEIGHT - v,) sottrae il valore v dal massimo peso MST_MAX_WEIGHT, invertendo l'ordine dei pesi.
    """
    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]  #edges is a list of tuples where each tuple contains the indexes of the cliques that are connected by an edge (u) and the weight of the edge (v)


    if len(edges) == 0: 
        return cliques, edges

    row, col, data = list(zip(*edges))
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]

    """Alla fine della funzione tree_decomp, otteniamo due cose principali:

    Liste delle Cliques:
    Una lista di cliques, dove ogni clique è rappresentata come una lista di indici di atomi. Queste cliques possono includere singoli legami, anelli, e atomi di ramificazione.
    
    Edge dell'Albero di Spanning Minimo (MST):
    Una lista di edge che costituiscono l'albero di spanning minimo del grafo delle cliques. Ogni edge è rappresentato come una tupla di indici delle cliques connesse."""
    return (cliques, edges)