import csv
import requests
import tqdm

def get_smiles_from_pubchem(pubchem_id):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{pubchem_id}/property/CanonicalSMILES/TXT"
    max_retries = 2
    retries = 0
    while retries < max_retries:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text.strip()
        else:
            # print(f"Errore: {response.status_code}. Riprova...")
            retries += 1
    # print("Massimo numero di tentativi raggiunto. Impossibile ottenere i dati.")
    return None

def add_smiles_and_drug_column(input_csv, output_csv):
    data = []

    # Leggi il file CSV esistente
    with open(input_csv, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        fields = reader.fieldnames + ["Drug", "SMILES"]

        # Usa tqdm per mostrare una barra di caricamento
        for row in tqdm.tqdm(reader, total=sum(1 for row in csv.DictReader(open(input_csv)))):
            if row["PubChem ID"] and row["PubChem ID"] != "NA":
                diseases = int(row["Diseases Associated with This Ingredient"])
                formulas = int(row["Formulas Containing This Ingredient"])
                row["Drug"] = 1 if diseases > 0 or formulas > 0 else 0
                row['SMILES'] = get_smiles_from_pubchem(row["PubChem ID"])
                data.append(row)

    # Scrivi i dati aggiornati in un nuovo file CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)

# Specifica il file CSV di input e di output
input_csv = 'ETCM_data.csv'
output_csv = 'ETCM_data_with_drug_and_smiles.csv'

# Aggiungi la colonna "Drug" e i SMILES, poi salva i dati nel nuovo file CSV
add_smiles_and_drug_column(input_csv, output_csv)

print(f"Dati aggiornati salvati in '{output_csv}'")
