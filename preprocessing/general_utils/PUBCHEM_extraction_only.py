import pandas as pd
import requests

def fetch_smiles(pubchem_id):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_id}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        try:
            print(data['PropertyTable']['Properties'][0]['CanonicalSMILES'])

            smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
            return smiles
        except:
            print(f"SMILES not found for PubChem ID {pubchem_id}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch SMILES for PubChem ID {pubchem_id}: {e}")
        return None

def add_smiles_to_csv(input_file, output_file):
    # Leggi il file CSV con i PubChem ID
    df = pd.read_csv(input_file)
    
    # Assicurati che ci sia una colonna denominata "PubChem ID"
    if 'PubChem ID' not in df.columns:
        print("Il file CSV deve contenere una colonna 'PubChem ID'.")
        return
    
    # Crea una nuova colonna per i SMILES
    df['SMILES'] = df['PubChem ID'].apply(lambda x: fetch_smiles(x) if pd.notna(x) else None)
    
    # Salva il DataFrame aggiornato in un nuovo file CSV
    df.to_csv(output_file, index=False)
    print(f"File con SMILES salvato in {output_file}")

# Esempio di utilizzo:
input_file = "tcm_medicinal_compounds_pubchem_ids.csv"  # Replace with the actual path to your input file
output_file = "GPT_compounds_output_file_drugs.csv"  # Replace with the desired path for the output file
add_smiles_to_csv(input_file, output_file)
