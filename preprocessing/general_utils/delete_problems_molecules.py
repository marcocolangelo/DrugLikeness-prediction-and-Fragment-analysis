import pandas as pd
from rdkit import Chem
import deepchem as dc

def filter_smiles_by_atoms(input_file, output_file, smiles_column):
    # Leggi il file CSV in un DataFrame
    df = pd.read_csv(input_file)
    
    # Istanzia il featurizer di DeepChem
    featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    # Funzione per validare e contare gli atomi
    def is_valid_molecule(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except Exception as e:
            print(f"Invalid SMILES: {smiles} for error: {e}")
            return False
        if mol is None or mol.GetNumAtoms() < 2:
            return False
        return True

    # Filtra molecole con SMILES non validi o meno di 2 atomi
    df = df[df[smiles_column].apply(is_valid_molecule)]

    # Funzione per featurizzare la molecola
    def can_featurize(smiles):
        mol = Chem.MolFromSmiles(smiles)
        try:
            featurizer._featurize(mol)
            return True
        except Exception:
            return False

    # Filtra il DataFrame mantenendo solo le molecole che possono essere featurizzate
    df_filtered = df[df[smiles_column].apply(can_featurize)]

    # Salva il risultato in un nuovo file CSV
    df_filtered.to_csv(output_file, index=False)

    print(f"Samples with invalid SMILES or less than two atoms removed. Result saved in {output_file}")

# Percorsi dei file
input_file = 'TCMC_compounds.csv'
output_file = 'preprocessed_TCMC_compounds.csv'
smiles_column = 'smiles'  # Sostituisci con il nome della colonna che contiene gli SMILES

# Esegui la funzione
filter_smiles_by_atoms(input_file, output_file, smiles_column)
