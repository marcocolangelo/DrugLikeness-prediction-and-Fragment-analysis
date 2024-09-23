import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Funzione per verificare se una sezione contiene parole chiave rilevanti
def contains_drug_info(sections, keywords):
    for section in sections:
        if any(keyword in section.get('TOCHeading', '') for keyword in keywords):
            return True
        for subsection in section.get('Section', []):
            if contains_drug_info([subsection], keywords):
                return True
    return False

def fetch_pubchem_info(pubchem_id):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{pubchem_id}/JSON"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        sections = data.get('Record', {}).get('Section', [])
        # Parole chiave per identificare informazioni sui farmaci
        drug_keywords = ['Drug and Medication Information', 'Pharmacology', 'Clinical Trials', 'Therapeutic Uses', 'Drug', 'Medication']
        if contains_drug_info(sections, drug_keywords):
            return pubchem_id, 'Present'
    except requests.exceptions.RequestException as e:
        print(f"Request error for PubChem ID {pubchem_id}: {e}")
    except Exception as e:
        print(f"Error processing PubChem ID {pubchem_id}: {e}")
    return pubchem_id, 'Not Present'

def main(csv_file):
    df = pd.read_csv(csv_file)
    if 'PubChem_CID' not in df.columns:
        print("CSV file must contain a 'PubChem_CID' column")
        return

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_pubchem_id = {executor.submit(fetch_pubchem_info, pubchem_id): pubchem_id for pubchem_id in df['PubChem_CID']}
        for future in tqdm(as_completed(future_to_pubchem_id), total=len(future_to_pubchem_id)):
            try:
                pubchem_id, result = future.result()
                results.append((pubchem_id, result))
            except Exception as e:
                pubchem_id = future_to_pubchem_id[future]
                print(f"Error processing PubChem ID {pubchem_id}: {e}")
                results.append((pubchem_id, 'Error'))

    result_df = pd.DataFrame(results, columns=['PubChem_CID', 'Drug and Medication Information'])
    result_df.to_csv('TM_MC_pubchem_results.csv', index=False)
    print("Results saved to TM_MC_pubchem_results.csv")

if __name__ == "__main__":
    csv_file = 'TM_MC/TM_MC_medicinal_compound.csv'  # replace with your CSV file path
    main(csv_file)
