import pandas as pd

def merge_files(properties_file, status_file, output_file):
    # Leggi il file delle proprietà chimiche
    properties_df = pd.read_csv(properties_file)
    
    # Leggi il file dello status
    status_df = pd.read_csv(status_file)
    
    # Verifica che le colonne necessarie siano presenti
    if 'CID' not in properties_df.columns or 'SMILES' not in properties_df.columns:
        print("Il file delle proprietà chimiche deve contenere le colonne 'CID' e 'SMILES'")
        return
    
    if 'PubChem_CID' not in status_df.columns or 'Drug and Medication Information' not in status_df.columns:
        print("Il file dello status deve contenere le colonne 'PubChem_CID' e 'Drug and Medication Information'")
        return
    
    # Converti entrambe le colonne al tipo stringa per evitare problemi di tipi di dati
    properties_df['CID'] = properties_df['CID'].astype(str)
    status_df['PubChem_CID'] = status_df['PubChem_CID'].astype(str)
    # status_df['Drug and Medication Information'] = status_df['Drug and Medication Information'].map({'Present': "1", 'Non present': "0"})
    
    # Unisci i due DataFrame sui rispettivi ID
    merged_df = pd.merge(status_df, properties_df, left_on='PubChem_CID', right_on='CID', how='inner')
    
    # Mappa 'Present' e 'Non present' a 1 e 0
    merged_df['Drug and Medication Information'] = merged_df['Drug and Medication Information'].map({'Present': 1, 'Not Present': 0})
    
    # Seleziona le colonne SMILES e il binario
    output_df = merged_df[['SMILES', 'Drug and Medication Information']]
    
    # Salva il risultato in un nuovo file CSV
    output_df.to_csv(output_file, index=False, float_format='%.0f')
    print(f"File di output salvato come {output_file}")

if __name__ == "__main__":
    properties_file = 'TM_MC\TM_MC_chemical_property.csv'  # sostituisci con il percorso del tuo file delle proprietà chimiche
    status_file = 'TM_MC\TM_MC_pubchem_results.csv'  # sostituisci con il percorso del tuo file di status
    output_file = 'TM_MC_pubchem_output.csv'  # sostituisci con il percorso del file di output desiderato
    
    merge_files(properties_file, status_file, output_file)
