# Importa le librerie necessarie
from rdkit import Chem
import pandas as pd
import numpy as np
import os

# Definisci i percorsi dei file
FRAGMENTS_FILE = 'fragments_smarts.txt'  # File con i frammenti SMARTS
MOLECULES_FILE = 'TCMBANK_molecule_predictions.csv'         # File CSV con molecole e predizioni

# 1. Definisci i pattern SMARTS per i frammenti
print("Caricamento dei frammenti SMARTS...")
with open(FRAGMENTS_FILE, 'r') as f:
    fragments_smarts_list = [line.strip() for line in f if line.strip()]

# Converte i pattern SMARTS in oggetti Molecola RDKit
fragment_patterns = []
invalid_fragments = []
for idx, smarts in enumerate(fragments_smarts_list):
    frag = Chem.MolFromSmarts(smarts)
    if frag is None:
        print(f"Frammento SMARTS non valido alla riga {idx+1}: {smarts}")
        invalid_fragments.append(idx)
    else:
        fragment_patterns.append((idx, frag))

print(f"Totale frammenti validi: {len(fragment_patterns)}")
print(f"Totale frammenti non validi: {len(invalid_fragments)}")

# 2. Carica le molecole con predizioni
print("\nCaricamento delle molecole e delle predizioni...")
molecules_df = pd.read_csv(MOLECULES_FILE)

# Verifica che le colonne richieste siano presenti
if not {'smiles', 'pred'}.issubset(molecules_df.columns):
    raise ValueError("Il file delle molecole deve contenere le colonne 'smiles' e 'pred'.")

# Converti i SMILES in oggetti Molecola RDKit
print("Conversione dei SMILES in oggetti RDKit Molecola...")
def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

molecules_df['mol'] = molecules_df['smiles'].apply(smiles_to_mol)

# Identifica e rimuovi molecole non valide
invalid_mols = molecules_df[molecules_df['mol'].isna()]
if not invalid_mols.empty:
    print(f"Molecole non valide trovate: {len(invalid_mols)}")
    print("Rimozione delle molecole non valide...")
    molecules_df = molecules_df.dropna(subset=['mol']).reset_index(drop=True)

print(f"Totale molecole valide: {len(molecules_df)}")

# 3. Costruisci la matrice di incidenza dei frammenti
print("\nCostruzione della matrice di incidenza dei frammenti...")
num_molecules = len(molecules_df)
num_fragments = len(fragment_patterns)

# Inizializza una matrice di zeri
incidence_matrix = np.zeros((num_molecules, num_fragments), dtype=int)

# Itera su ogni molecola e verifica la presenza di ciascun frammento
for i, mol in enumerate(molecules_df['mol']):
    if (i+1) % 1000 == 0 or i == num_molecules -1:
        print(f"Processamento molecola {i+1} di {num_molecules}...")
    for j, (frag_id, frag) in enumerate(fragment_patterns):
        if mol.HasSubstructMatch(frag):
            incidence_matrix[i, j] = 1

# Aggiungi la matrice di incidenza al DataFrame
print("Creazione del DataFrame della matrice di incidenza...")
incidence_df = pd.DataFrame(incidence_matrix, columns=[f'frag_{frag_id}' for frag_id, _ in fragment_patterns])
molecules_df = pd.concat([molecules_df, incidence_df], axis=1)

# 4. Separazione delle molecole in base alla predizione
print("\nSeparazione delle molecole in base alla predizione 'pred'...")
pred_zero_df = molecules_df[molecules_df['pred'] == 0].reset_index(drop=True)
pred_one_df = molecules_df[molecules_df['pred'] == 1].reset_index(drop=True)

print(f"Molecole predette come 0: {len(pred_zero_df)}")
print(f"Molecole predette come 1: {len(pred_one_df)}")

# 5. Costruisci matrici di incidenza separate per ciascun gruppo
def build_incidence_matrix(df, group_label):
    print(f"\nCostruzione della matrice di incidenza per il gruppo pred = {group_label}...")
    incidence = df[[col for col in incidence_df.columns]].copy()
    return incidence

incidence_pred_0 = build_incidence_matrix(pred_zero_df, 0)
incidence_pred_1 = build_incidence_matrix(pred_one_df, 1)

# 6. Calcola le frequenze dei frammenti per ciascun gruppo
print("\nCalcolo delle frequenze dei frammenti per ciascun gruppo...")
freq_pred_0 = incidence_pred_0.sum().sort_values(ascending=False)
freq_pred_1 = incidence_pred_1.sum().sort_values(ascending=False)

# 7. Salva i risultati in file CSV
output_dir = 'output_analysis'
os.makedirs(output_dir, exist_ok=True)

print(f"\nSalvataggio delle matrici di incidenza e delle frequenze in '{output_dir}'...")
# Matrici di incidenza
incidence_pred_0.to_csv(os.path.join(output_dir, 'incidence_matrix_pred_0.csv'), index=False)
incidence_pred_1.to_csv(os.path.join(output_dir, 'incidence_matrix_pred_1.csv'), index=False)

# Frequenze dei frammenti
freq_pred_0.to_csv(os.path.join(output_dir, 'fragment_frequencies_pred_0.csv'), header=['frequency'])
freq_pred_1.to_csv(os.path.join(output_dir, 'fragment_frequencies_pred_1.csv'), header=['frequency'])

print("Processo completato con successo!")

# 8. (Opzionale) Analisi aggiuntive
# Ad esempio, puoi esportare le molecole con i loro frammenti per ulteriori analisi
# molecules_df.to_csv(os.path.join(output_dir, 'molecules_with_fragments.csv'), index=False)
