import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def process_file(file_path, smiles_set, smiles_dict):
    matches = []
    try:
        # Leggi il file CSV
        csv_df = pd.read_csv(file_path, dtype={'canonicalsmiles': str})
        if 'canonicalsmiles' in csv_df.columns:
            # Filtra righe con canonicalsmiles vuoto
            csv_df = csv_df.dropna(subset=['canonicalsmiles'])
            # Cerca corrispondenze per ogni smile nel file corrente
            for smile in smiles_set:
                if smile in csv_df['canonicalsmiles'].values:
                    matches.append({
                        'Smiles': smile,
                        'TCMBank_ID': smiles_dict[smile],
                        'FilePath': file_path
                    })
    except Exception as e:
        print(f"Errore nella lettura del file {file_path}: {e}")
    return matches

def find_smiles_matches(input_dir, smiles_file, output_file, no_match_file):
    # Leggi il file con i campi "Smiles" e "TCMBank_ID"
    smiles_df = pd.read_csv(smiles_file, dtype={'Smiles': str, 'TCMBank_ID': str})
    smiles_set = set(smiles_df['Smiles'].values)
    smiles_dict = dict(smiles_df[['Smiles', 'TCMBank_ID']].values)

    matches = []
    file_paths = []

    # Itera attraverso tutti i file nella directory di input
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                file_paths.append(os.path.join(root, file))

    # Usa ThreadPoolExecutor per parallelizzare l'elaborazione dei file
    with ThreadPoolExecutor() as executor:
        results = executor.map(lambda file_path: process_file(file_path, smiles_set, smiles_dict), file_paths)
        for result in results:
            matches.extend(result)

    # Trova gli elementi che non hanno trovato corrispondenza
    matched_smiles = set([match['Smiles'] for match in matches])
    no_matches = [{'Smiles': smile, 'TCMBank_ID': smiles_dict[smile]} for smile in smiles_set if smile not in matched_smiles]

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
    input_dir = 'drugs_dbs'
    smiles_file = 'TCMBANK/TCMBANK_ingredient_all.csv'
    output_file = 'NEW_drugs.csv'
    no_match_file = 'NEW_non_verified_drugs.csv'
    
    find_smiles_matches(input_dir, smiles_file, output_file, no_match_file)
