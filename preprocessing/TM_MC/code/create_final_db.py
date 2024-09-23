import pandas as pd

def merge_drug_non_drug_files(drugs_file, non_drugs_file, output_file):
    # Leggi i file CSV
    drugs_df = pd.read_csv(drugs_file)
    non_drugs_df = pd.read_csv(non_drugs_file)
    
    # Estrai il campo Smiles e aggiungi il campo Drug
    drugs_df = drugs_df[['Smiles']]
    drugs_df['Drug'] = 1
    
    non_drugs_df = non_drugs_df[['Smiles']]
    non_drugs_df['Drug'] = 0
    
    # Combina i due DataFrame
    combined_df = pd.concat([drugs_df, non_drugs_df], ignore_index=True)
    
    # Salva il DataFrame combinato in un nuovo file CSV
    combined_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    drugs_file = 'TCMBANK/NEW_drugs.csv'  # Sostituisci con il percorso del file drugs
    non_drugs_file = 'TCMBANK/NEW_non_drugs.csv'  # Sostituisci con il percorso del file non_drugs
    output_file = 'TCMBANK/TCMBANK_final_dataset.csv'  # Sostituisci con il percorso del file di output
    
    merge_drug_non_drug_files(drugs_file, non_drugs_file, output_file)
