import os
import pandas as pd

def clean_smiles(smiles):
    # Controlla se il valore è NaN
    if isinstance(smiles, float) and pd.isna(smiles):
        return smiles
    # Rimuove i caratteri @, /, e \
    return smiles.replace('@', '').replace('/', '').replace('\\', '')

def find_smiles_matches(non_ambig_file, smiles_file, output_file, no_match_file):
    # Leggi il file con i campi "Smiles" e "TCMBank_ID"
    smiles_df = pd.read_csv(smiles_file)
    # Pulisci i campi "Smiles"
    smiles_df['Smiles'] = smiles_df['Smiles'].apply(clean_smiles)
    smiles_dict = dict(smiles_df[['Smiles', 'TCMBank_ID']].values)
    smiles_set = set(smiles_df['Smiles'].values)

    # Liste per memorizzare i risultati
    matches = []
    no_matches = []

    # Leggi il file CSV
    csv_df = pd.read_csv(non_ambig_file)
    # Pulisci i campi "Smiles" nel file corrente
    csv_df['SMILES'] = csv_df['SMILES'].apply(clean_smiles)
    
    if 'SMILES' in csv_df.columns:
        # Filtra righe con canonicalsmiles vuoto
        csv_df = csv_df.dropna(subset=['SMILES'])
        canonicalsmiles_set = set(csv_df['SMILES'].values)

        # Cerca corrispondenze per ogni smile nel file corrente
        for smile in smiles_set:
            if smile in canonicalsmiles_set:
                matches.append({
                    'Smiles': smile,
                    'TCMBank_ID': smiles_dict[smile],
                })
            else:
                no_matches.append({
                    'Smiles': smile,
                    'TCMBank_ID': smiles_dict[smile]
                })
    else:
        print("Il file CSV non contiene la colonna 'Smiles'.")

    # Crea DataFrame dai risultati e salva nei file CSV
    if matches:
        matches_df = pd.DataFrame(matches)
        matches_df.to_csv(output_file, index=False)
    else:
        print("Nessuna corrispondenza trovata.")

    if no_matches:
        no_matches_df = pd.DataFrame(no_matches)
        no_matches_df.to_csv(no_match_file, index=False)
    else:
        print("Tutti gli Smiles hanno trovato una corrispondenza.")

if __name__ == "__main__":
    # non_drugs = 'nondrugs_dbs/processed/NEW_non_ambiguous_compounds.csv'
    # smiles_file = 'TCMBANK/TCMBANK_ingredient_all.csv'
    # output_file = 'NEW_non_drugs.csv'
    # no_match_file = 'NEW_still_non_verified_drugs.csv'

    non_drugs = 'nondrugs_dbs/raw/susdat_NORMAN.csv'
    smiles_file = 'TCMBANK/TCMBANK_ingredient_all.csv'
    output_file = 'NORMAN_non_drugs.csv'
    no_match_file = 'NORMAN_still_non_verified_drugs.csv'
    
    find_smiles_matches(non_drugs, smiles_file, output_file, no_match_file)
