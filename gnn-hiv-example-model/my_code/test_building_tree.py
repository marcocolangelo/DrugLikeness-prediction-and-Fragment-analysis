import unittest
import torch
from rdkit import Chem

from tree_building import DGLMolTree, get_mol

class TestMolecularEdgeMap(unittest.TestCase):
    
    def setUp(self):
        self.smiles = 'Cc1cnc(N)c(Cl)c1'
        self.mol_tree = DGLMolTree(self.smiles)
        
        # Estrarre gli edges
        self.tree_edges = self.mol_tree.get_edge_index()
        self.mol = get_mol(self.smiles)
        
        # Estrarre la mappa degli archi molecolari
        self.mol_edge_map = self.mol_tree.mol_edge_map
        
        # Creazione della lista degli archi molecolari
        self.molecular_edges = []
        for bond in self.mol.GetBonds():
            self.molecular_edges.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))

    def test_tree_edges_exist_in_mol_edge_map(self):
        """Test che verifica se i tree_edge riportati in mol_edge_map esistono realmente."""
        print(self.mol_edge_map)
        print(self.tree_edges)
        for tree_edge in self.tree_edges.t().tolist():
            tree_edge_tuple = tuple(tree_edge)
            self.assertIn(tree_edge_tuple, self.mol_edge_map, f"Tree edge {tree_edge_tuple} non trovato in mol_edge_map")

    def test_molecular_edges_exist_in_mol_edge_map(self):
        """Test che verifica se i mol_edge riportati in mol_edge_map esistono davvero nella molecola."""
        for mol_edge in self.mol_edge_map.values():
            self.assertIn(mol_edge, self.molecular_edges, f"Mol edge {mol_edge} non trovato tra gli archi molecolari della molecola")

    def test_mol_edge_map_association_sense(self):
        """Test che verifica se l'associazione tra tree edge e mol edge è sensata."""
        for tree_edge, mol_edge in self.mol_edge_map.items():
            tree_clique_1 = self.mol_tree.nodes_dict[tree_edge[0]]['clique']
            tree_clique_2 = self.mol_tree.nodes_dict[tree_edge[1]]['clique']
            
            self.assertTrue(
                mol_edge[0] in tree_clique_1 and mol_edge[1] in tree_clique_2 or 
                mol_edge[0] in tree_clique_2 and mol_edge[1] in tree_clique_1,
                f"L'associazione tra tree edge {tree_edge} e mol edge {mol_edge} non è sensata"
            )

    def test_indices_exist_in_mol_and_tree_edges(self):
        """Test che verifica se gli indici riportati in tree edge e mol edge sono realmente esistenti."""
        for tree_edge, mol_edge in self.mol_edge_map.items():
            # Verifica se gli indici di mol_edge esistono realmente nella molecola
            self.assertLess(mol_edge[0], self.mol.GetNumAtoms(), f"Indice {mol_edge[0]} non esiste nella molecola")
            self.assertLess(mol_edge[1], self.mol.GetNumAtoms(), f"Indice {mol_edge[1]} non esiste nella molecola")
            
            # Verifica se gli indici di tree_edge esistono realmente nell'albero
            self.assertLess(tree_edge[0], self.mol_tree.treesize(), f"Indice {tree_edge[0]} non esiste nell'albero")
            self.assertLess(tree_edge[1], self.mol_tree.treesize(), f"Indice {tree_edge[1]} non esiste nell'albero")

    def test_consistency_of_edge_map_size(self):
        """Test che verifica la coerenza delle dimensioni di mol_edge_map e tree_edge_attr."""
        self.assertEqual(len(self.mol_edge_map), self.tree_edges.size(1), 
                        "Il numero di tree edges non corrisponde al numero di mol edges nella mappa")

    def test_no_duplicate_edges(self):
        """Test che verifica l'assenza di duplicati in mol_edge_map."""
        seen_tree_edges = set()
        seen_mol_edges = set()
        for tree_edge, mol_edge in self.mol_edge_map.items():
            self.assertNotIn(tree_edge, seen_tree_edges, f"Duplicato trovato in tree_edge: {tree_edge}")
            self.assertNotIn(mol_edge, seen_mol_edges, f"Duplicato trovato in mol_edge: {mol_edge}")
            seen_tree_edges.add(tree_edge)
            seen_mol_edges.add(mol_edge)

    def test_bidirectional_edges(self):
        """Test che verifica la bidirezionalità degli archi."""
        for tree_edge, mol_edge in self.mol_edge_map.items():
            reverse_tree_edge = (tree_edge[1], tree_edge[0])
            reverse_mol_edge = (mol_edge[1], mol_edge[0])
            self.assertIn(reverse_tree_edge, self.mol_edge_map, 
                        f"Arco bidirezionale mancante per tree_edge {tree_edge}")
            self.assertEqual(self.mol_edge_map[reverse_tree_edge], reverse_mol_edge, 
                            f"Incoerenza nell'arco bidirezionale per tree_edge {tree_edge}")

    def test_valid_edge_attributes(self):
        """Test che verifica la validità degli attributi degli archi."""
        valid_bond_types = {Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                            Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC}
        for bond in self.mol.GetBonds():
            self.assertIn(bond.GetBondType(), valid_bond_types, 
                        f"Tipo di legame non valido: {bond.GetBondType()} per bond tra {bond.GetBeginAtomIdx()} e {bond.GetEndAtomIdx()}")

    def test_no_isolated_edges(self):
        """Test che verifica l'assenza di nodi isolati nell'albero delle clique."""
        connectivity = torch.zeros(self.mol_tree.treesize())
        for src, dst in self.tree_edges.t().tolist():
            connectivity[src] += 1
            connectivity[dst] += 1
        self.assertTrue(torch.all(connectivity > 0), "Alcuni nodi nell'albero delle clique sono isolati")


if __name__ == '__main__':
    unittest.main()