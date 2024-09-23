import os
import time
import numpy as np
import rdkit.Chem as Chem
import tqdm
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.ML.Descriptors import MoleculeDescriptors

from IPython.display import SVG
from cairosvg import svg2png,svg2pdf


from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
from dgl import unbatch
from dgllife.utils import CanonicalAtomFeaturizer,AttentiveFPAtomFeaturizer
from dgllife.utils import CanonicalBondFeaturizer


# get node_feature from dgllife
"""Questa funzione è stata modificata per poter essere utilizzata con il dataset di Chembl
Serve a creare un dizionario con le features degli atomi di una molecola"""
def get_dgl_node_feature(mol):
    '''
    """A default featurizer for atoms.

    The atom features include:
    * **One hot encoding of the atom type**. The supported atom types include
      ``C``, ``N``, ``O``, ``S``, ``F``, ``Si``, ``P``, ``Cl``, ``Br``, ``Mg``,
      ``Na``, ``Ca``, ``Fe``, ``As``, ``Al``, ``I``, ``B``, ``V``, ``K``, ``Tl``,
      ``Yb``, ``Sb``, ``Sn``, ``Ag``, ``Pd``, ``Co``, ``Se``, ``Ti``, ``Zn``,
      ``H``, ``Li``, ``Ge``, ``Cu``, ``Au``, ``Ni``, ``Cd``, ``In``, ``Mn``, ``Zr``,
      ``Cr``, ``Pt``, ``Hg``, ``Pb``.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 10``.
    * **One hot encoding of the number of implicit Hs on the atom**. The supported
      possibilities include ``0 - 6``.
    * **Formal charge of the atom**.
    * **Number of radical electrons of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.
    '''

    atom_featurizer = CanonicalAtomFeaturizer(atom_data_field='h') # atom_data_field='h' means the atom features are stored in the field 'h' of the atom
    feats = atom_featurizer(mol)

    return feats['h']

# get bond_feature (dgl sta per la libreria dgllife usata)
def get_dgl_bond_feature(mol):
    # 13 size 
    #Questo commento serve a ricordare che ogni legame nel grafico molecolare sarà rappresentato da un vettore di 13 dimensioni. 
    # Queste dimensioni possono includere varie proprietà del legame, come il tipo di legame, la coniugazione, la presenza in anelli, 
    # e altre caratteristiche chimiche rilevanti.
    bond_featurizer = CanonicalBondFeaturizer(bond_data_field='feat', self_loop=True)
    feats = bond_featurizer(mol)
    return feats['feat']



'''get molecule node features for graph (three function below)'''
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs that not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


"""Esempio di Utilizzo
Supponiamo di avere un atomo di carbonio che ha le seguenti proprietà:

Simbolo: C
Grado: 3
Carica formale: 0
Elettroni radicali: 0
Ibridazione: SP3
Aromaticità: False
Numero di idrogeni totali: 3
Chiralità: S
L'output della funzione sarà un array contenente la codifica di queste caratteristiche tramite one-hot encoding per ciascuna di esse."""

def atom_features(atom,bool_id_feat=False,explicit_H=False,use_chirality=True):
    if bool_id_feat:
        pass
    else:
        results = one_of_k_encoding_unk(
          atom.GetSymbol(),
          [
            'B',
            'C',
            'N',
            'O',
            'F',
            'Si',
            'P',
            'S',
            'Cl',
            'As',
            'Se',
            'Br',
            'Te',
            'I',
            'At',
            'other'
          ]) + one_of_k_encoding(atom.GetDegree(),
                                 [0, 1, 2, 3, 4, 5, 6 ,7, 8 , 9 , 10]) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                        SP3D, Chem.rdchem.HybridizationType.SP3D2,'other'
                  ]) + [atom.GetIsAromatic()]
        # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
        if not explicit_H:
            results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                      [0, 1, 2, 3, 4])
            
        """Dettagli:
        atom.GetProp('_CIPCode') recupera la configurazione di chiralità dell'atomo (se presente). 
        I valori possibili sono 'R' (rectus) e 'S' (sinister), che descrivono la configurazione stereochimica.
        one_of_k_encoding_unk è una funzione che esegue una codifica one-hot per la configurazione di chiralità. 
        Se la configurazione è 'R', il risultato sarà [1, 0]; se è 'S', sarà [0, 1]. Se non è né 'R' né 'S', il risultato sarà [0, 0]."""
        if use_chirality:
            try:
                results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
            except:
                results = results + [False, False
                                     ] + [atom.HasProp('_ChiralityPossible')]
                
        

        return np.array(results)

"""usaule ad atoms_features, ma per i legami"""
def bond_features(bond,self_loop=False):
    # for bond object type..: Chem.rdchem.Bond
    feature = one_of_k_encoding(bond.GetBondType(),[Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC]) + \
        [bond.GetIsConjugated()] + [bond.IsInRing()] + one_of_k_encoding(bond.GetStereo(),[Chem.rdchem.BondStereo.STEREONONE,Chem.rdchem.BondStereo.STEREOANY,Chem.rdchem.BondStereo.STEREOZ,Chem.rdchem.BondStereo.STEREOE])

    if self_loop:
        feature += [False]  # indicate index: self loops or not
    return np.array(feature).astype(float)


"""set_atommap(mol): Questa funzione prende un oggetto mol (molecola) come input e imposta un mappaggio degli atomi. 
Questo viene fatto assegnando un numero univoco ad ogni atomo nella molecola. 
Questi numeri sono utili per tracciare gli atomi specifici attraverso le trasformazioni chimiche. 
Ad esempio, se stai cercando di capire come una reazione chimica modifica una molecola, 
potresti utilizzare un mappaggio degli atomi per vedere esattamente quali atomi nella molecola di partenza corrispondono a quali atomi nel prodotto. 
In questa funzione, il numero assegnato a ciascun atomo è semplicemente l'indice dell'atomo nella molecola 
(più uno, poiché l'indicizzazione inizia da zero)."""
def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num) 

"""get_mol(smiles): Questa funzione prende una stringa SMILES (smiles) come input e restituisce un oggetto mol che rappresenta la molecola corrispondente.
 Se la stringa SMILES non può essere convertita in una molecola valida, la funzione restituisce None. Prima di restituire la molecola, 
la funzione Chem.Kekulize(mol) viene chiamata per assicurarsi che la rappresentazione della molecola sia nella forma di Kekulé."""
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print('Error in chemutils/smiles, non può essere convertita per cui sarà trasformata in None:', smiles)
        return None
    Chem.Kekulize(mol)
    return mol

"""get_smiles(mol): Questa funzione prende una molecola (mol) come input e restituisce la sua rappresentazione SMILES. 
SMILES è una stringa che rappresenta in modo univoco la struttura chimica di una molecola."""
def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)

"""sanitize(mol): Questa funzione prende un oggetto mol (molecola) come input e tenta di "sanificare" la molecola. 
La sanificazione della molecola coinvolge la conversione della molecola in una stringa SMILES utilizzando la funzione get_smiles(mol), 
quindi la conversione della stringa SMILES in un oggetto mol utilizzando la funzione get_mol(smiles). 
Se durante questo processo si verifica un'eccezione, la funzione restituisce None. In caso contrario, restituisce la molecola sanificata."""
def sanitize(mol): 
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        print('Error in chemutils/sanitize, restituito None: ', e)
        return None
    return mol

"""copy_atom(atom): Questa funzione prende un oggetto atom (atomo) come input e crea una copia di esso. 
La copia viene creata creando un nuovo oggetto new_atom con lo stesso simbolo dell'atomo originale e 
impostando la stessa carica formale e il numero di mappa dell'atomo."""
def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom

"""copy_edit_mol(mol): Questa funzione prende un oggetto mol (molecola) come input e crea una copia modificabile della molecola.
 La copia viene creata creando un nuovo oggetto new_mol vuoto. Quindi, per ogni atomo nella molecola originale, 
 viene creata una copia utilizzando la funzione copy_atom(atom) e viene aggiunta alla nuova molecola. 
 Successivamente, per ogni legame nella molecola originale, vengono ottenuti gli indici degli atomi di inizio e fine del legame, 
il tipo di legame e il legame vengono aggiunti alla nuova molecola. Infine, la nuova molecola viene restituita."""
def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol

"""get_clique_mol(mol, atoms): Questa funzione prende in input una molecola (mol) e un insieme di atomi (atoms). 
Utilizza la funzione Chem.MolFragmentToSmiles per ottenere la rappresentazione SMILES del frammento della molecola che consiste solo 
degli atomi specificati. Questa rappresentazione SMILES viene poi convertita in un oggetto molecola, 
che viene modificato per essere editabile con la funzione copy_edit_mol. 
Infine, la molecola viene "sanificata" con la funzione sanitize e restituita."""
def get_clique_mol(mol, atoms):
    try:
        smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True) #take only a part of the moleculeù
    except:
        print('Error in chemutils/get_clique_mol, restituito None')
        return None
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    new_mol = sanitize(new_mol)  # We assume this is not None
    return new_mol

"""Questa funzione prende in input una stringa SMILES e restituisce l'impronta digitale di Morgan della molecola corrispondente. 
L'impronta digitale di Morgan è una rappresentazione binaria della molecola che cattura le sue caratteristiche strutturali. 
Questa funzione utilizza un raggio di 1 e una lunghezza di impronta digitale di 512 bit.
Un raggio di 1 significa che per ogni atomo nella molecola, l'impronta digitale di Morgan includerà informazioni solo sull'atomo stesso e sui suoi atomi direttamente legati."""
def get_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    # fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=512)
    npfp = np.array(list(fp.ToBitString())).astype('float')
    return npfp


def get_descriptors(smiles):
    mol = get_mol(smiles)
    des_list = ['MolLogP','NumHAcceptors','NumHeteroatoms','NumHDonors','MolWt','NumRotatableBonds',
                'RingCount','Ipc','HallKierAlpha','NumValenceElectrons','NumSaturatedRings','NumAliphaticRings','NumAromaticRings']
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list) #- simpleList: list of simple descriptors to be calculated - format of simpleList: a list of strings which are functions in the rdkit.Chem.Descriptors module

    result = np.array(calculator.CalcDescriptors(mol)) #result is a numpy array of the calculated descriptors
    result[np.isnan(result)] = 0 #replace NaN with 0
    return result.astype("float")


# highlight the crucial atoms-bonds in a molecule
"""Parametri:
iter_nums: Numero di iterazioni.
output_batch: Batch di output del modello.
input_batch: Batch di input delle molecole.
RGB: Colore per l'evidenziazione.
radius: Raggio degli atomi evidenziati.
size: Dimensioni dell'immagine generata.
Funzionamento:
Estrae le informazioni sugli atomi e sui legami cruciali e chiama highlight_mol per visualizzare la molecola."""

class highlight_mol(object):
    def __init__(self, verbose=False):
        self.verbose = verbose

    #quindi quando chiami highlight_mol, in realtà stai chiamando il metodo __call__ .
    def __call__(self, iter_nums, output_batch, input_batch, RGB=(236 / 256., 173 / 256., 158 / 256.), radius=0.5,
                  size=(400, 200)):

        print('=========highlight crucial cluster============')
        output_batch = unbatch(output_batch) #Revert the batch operation by split the given graph into a list of small ones.  If the node_split or the edge_split is not given, it calls DGLGraph.batch_num_nodes and DGLGraph.batch_num_edges of the input graph to get the information.
        output_batch_size = len(output_batch)
        # print(f"output_batch_size: {output_batch_size}")
        for idx, chem in enumerate(tqdm.tqdm(output_batch)):
            # try:
                bonds = set()
                
                #chem.edata['e']: Contiene i dati relativi ai legami del grafico, come le caratteristiche o i punteggi associati ai legami.
                #.argmax(): Restituisce l'indice del legame con il valore massimo tra tutti i legami, forse in termini di attention.
                max_eid = chem.edata['e_f'].argmax()
                u, v = chem.find_edges(max_eid) #Returns the source and destination node IDs of the edges with the specified edge IDs.
                mol = get_mol(input_batch['smiles'][idx]) #mol è la molecola correntemente in esame
                clique = input_batch['mol_trees'][idx].nodes_dict[int(u)]['clique'] # it's extracting the 'clique' information for that node and storing it in the variable clique
                clique2 = input_batch['mol_trees'][idx].nodes_dict[int(v)]['clique']
                """a 'clique' is a subset of vertices of an undirected graph such that every two distinct vertices in the clique are adjacent. In other words, it's a subset of the graph where every node is connected to every other node."""

                for ix in range(len(clique)):
                    if (ix < len(clique) - 1):
                        bonds.add(mol.GetBondBetweenAtoms(clique[ix], clique[ix + 1]).GetIdx()) #GetBondBetweenAtoms: Returns the bond between two atoms in the clique
                #     elif (ix == len(clique) - 1 and len(clique) > 1):
                #         bonds.add(mol.GetBondBetweenAtoms(clique[0], clique[-1]).GetIdx())
                #
                # for ix in range(len(clique2)):
                #     if (ix < len(clique2) - 1):
                #         bonds.add(mol.GetBondBetweenAtoms(clique2[ix], clique2[ix + 1]).GetIdx())
                #     elif (ix == len(clique2) - 1 and len(clique2) > 1):
                #         bonds.add(mol.GetBondBetweenAtoms(clique2[0], clique2[-1]).GetIdx())
            # except:
            #     print('error in get clique in chemutils.py/ highlight_mol/ __call__')
            #     pass
            # clique = clique + clique2
                self.highlight_mol(mol, clique, bonds, c=RGB, r=radius, s=size, output_name='mol_' + str(idx + 1) + str(iter_nums))


            # time.sleep(0.01)
            # print('\r{}/{}th to highlight..'.format(idx + 1, output_batch_size), end='')

    def find_tree_node(self, graph_node, mol_tree):
        # Cerca il nodo dell'albero che corrisponde al nodo del grafo grezzo
        for node_id, node_data in mol_tree.nodes_dict.items():
            if graph_node in node_data['graph_node_ids']:
                return node_id
        raise ValueError(f"Nodo {graph_node} non trovato nella rappresentazione ad albero")


    def highlight_mol(self, mol, atoms_id=None, bonds_id=None, c=(1, 0, 0), r=0.5, s=(400, 200), output_name='test'):
            """
            Highlights specific atoms and bonds in a molecule and generates an image.

            Args:
                mol (Molecule): The molecule object.
                atoms_id (list): List of atom IDs to highlight. Default is None.
                bonds_id (list): List of bond IDs to highlight. Default is None.
                c (tuple): RGB color values for highlighting. Default is (1, 0, 0) (red).
                r (float): Radius of highlighted atoms. Default is 0.5.
                s (tuple): Size of the generated image. Default is (400, 200).
                output_name (str): Name of the output image file. Default is 'test'.

            Returns:
                None
            """
            atom_hilights = {}
            bond_hilights = {}
            radii = {}

            
            if atoms_id:
                for atom in atoms_id:
                    atom_hilights[int(atom)] = c
                    radii[int(atom)] = r

            if bonds_id:
                for bond in bonds_id:
                    bond_hilights[int(bond)] = c

            self.generate_image(mol, list(atom_hilights.keys()), list(bond_hilights.keys()),
                           atom_hilights, bond_hilights, radii, s, output_name + '.png', False)

    def generate_image(self, mol, highlight_atoms, highlight_bonds, atomColors, bondColors, radii, size, output,
                           isNumber=False):
            """
            Generates an image of a molecule with highlighted atoms and bonds.

            Args:
                mol (rdkit.Chem.rdchem.Mol): The molecule to be drawn.
                highlight_atoms (list): List of atom indices to be highlighted.
                highlight_bonds (list): List of bond indices to be highlighted.
                atomColors (list): List of colors for highlighted atoms.
                bondColors (list): List of colors for highlighted bonds.
                radii (list): List of radii for highlighted atoms.
                size (tuple): The size of the output image in pixels (width, height).
                output (str): The output file path for the image.
                isNumber (bool, optional): Whether to label atoms with their indices. Defaults to False.
            """
            if self.verbose:
                print('\thighlight_atoms_id:', highlight_atoms)
                print('\thighlight_bonds_id:', highlight_bonds)
                print('\tatoms_colors:', atomColors)
                print('\tbonds_colors:', bondColors)

            view = rdMolDraw2D.MolDraw2DSVG(size[0], size[1]) #it creates a window for the 2D drawing of a molecule in SVG format with the specified size
            tm = rdMolDraw2D.PrepareMolForDrawing(mol)  #Prepare a molecule for drawing. This function generates coordinates for the atoms in the molecule if they do not already exist.

            option = view.drawOptions()
            if isNumber:
                for atom in mol.GetAtoms():
                    option.atomLabels[atom.GetIdx()] = atom.GetSymbol() + str(atom.GetIdx() + 1) #this line is adding the atom index to the atom symbol in the atom label for each atom in the molecule

            view.DrawMolecule(tm, highlightAtoms=highlight_atoms, highlightBonds=highlight_bonds,
                              highlightAtomColors=atomColors, highlightBondColors=bondColors, highlightAtomRadii=radii)
            view.FinishDrawing()

            svg = view.GetDrawingText() #Get the SVG text of the drawing
            SVG(svg.replace('svg:', '')) #Display the SVG image but first remove the 'svg:' string from the SVG text 

            try:
                os.makedirs('./mol_picture/', exist_ok=False)
            except:
                pass
            
            
            svg2png(bytestring=svg, write_to='./mol_picture/' + output)  # svg to png format


def tree_decomp_old(mol):

    MST_MAX_WEIGHT = 100  
    n_atoms = mol.GetNumAtoms() 
    if n_atoms == 1:  
        return [[0]], []

    # step 1 :
    cliques = [] 
    for bond in mol.GetBonds(): 
        a1 = bond.GetBeginAtom().GetIdx() 
        a2 = bond.GetEndAtom().GetIdx()  
        if not bond.IsInRing(): 
            cliques.append([a1, a2])

    ssr = [list(x) for x in Chem.GetSymmSSSR(mol)] 
    cliques.extend(ssr)  

    nei_list = [[] for i in range(n_atoms)]  
    for i in range(len(cliques)):  
        for atom in cliques[i]: 
            nei_list[atom].append(i) 


    for i in range(len(cliques)): 
        if len(cliques[i]) <= 2:  
            continue
        for atom in cliques[i]:  
            for j in nei_list[atom]: 
                if i >= j or len(cliques[j]) <= 2: 
                    continue
                inter = set(cliques[i]) & set(cliques[j])  
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []

    cliques = [c for c in cliques if len(c) > 0] 
    nei_list = [[] for i in range(n_atoms)] 
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)


    edges = defaultdict(int)
    for atom in range(n_atoms): 
        if len(nei_list[atom]) <= 1: 
            continue
        cnei = nei_list[atom]  
        bonds = [c for c in cnei if len(cliques[c]) == 2] 
        rings = [c for c in cnei if len(cliques[c]) > 4]  
        if len(bonds) > 2 or (len(bonds) == 2 and len(cnei) > 2): 
            cliques.append([atom])  
            c2 = len(cliques) - 1
            for c1 in cnei: 
                edges[(c1, c2)] = 1
        elif len(rings) > 2:  
            cliques.append([atom])
            c2 = len(cliques) - 1
            for c1 in cnei: 
                edges[(c1, c2)] = MST_MAX_WEIGHT - 1
        else: 
            for i in range(len(cnei)):
                for j in range(i + 1, len(cnei)):
                    c1, c2 = cnei[i], cnei[j]
                    inter = set(cliques[c1]) & set(cliques[c2])
                    if edges[(c1, c2)] < len(inter):
                        edges[(c1, c2)] = len(inter)  

    edges = [u + (MST_MAX_WEIGHT - v,) for u, v in edges.items()]


    if len(edges) == 0: 
        return cliques, edges

    row, col, data = list(zip(*edges))
    n_clique = len(cliques)
    clique_graph = csr_matrix((data, (row, col)), shape=(n_clique, n_clique))
    junc_tree = minimum_spanning_tree(clique_graph)
    row, col = junc_tree.nonzero()
    edges = [(row[i], col[i]) for i in range(len(row))]

    return (cliques, edges)

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



# for test
if __name__ == '__main__':
    m1 = Chem.MolFromSmiles('C1C2C3(CC1)C(CC2)CCC3') # 三环[6.3.0.01,5]十一烷
    m2 = Chem.MolFromSmiles('C1=CC=CC=C1C(=O)O')   # 苯甲酸
    m3 = Chem.MolFromSmiles('C1CCC(CC1)(C)C') # 1,1-二甲基环己烷
    m4 = Chem.MolFromSmiles('C1CC2CC1C(=O)CC2=O') # 二环[3.2.1]辛烷-2,4-二酮 （含桥）
    m5 = Chem.MolFromSmiles('CCCC(=O)O') # 丁酸
    (cliques,edges) = tree_decomp(m2)
    
    print(cliques)

    #usa highlight_mol
