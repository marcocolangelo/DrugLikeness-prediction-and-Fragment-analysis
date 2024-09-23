import pandas as pd

def find_matching_smiles(file1, file2, output_file):
    # Leggi i file CSV utilizzando pandas
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Esegui un merge tra i due DataFrame sui ZINC ID
    result_df = pd.merge(df2[['ZINCID', 'Drug']], df1[['ZINCID', 'Smiles']], on='ZINCID', how='inner')

    print('Numero di SMILES trovati:', len(result_df))

    # Scrivi i risultati nel file di output
    result_df.to_csv(output_file, index=False)

# Esempio di utilizzo della funzione
file1 = 'ZINC\\new\\ZINC_smiles_output.csv'
file2 = 'ZINC\\new2\\new_ZINC_ids_labelled.csv'
output_file = 'ZINC\\new2\\new2_ZINC_smiles_output.csv'
find_matching_smiles(file1, file2, output_file)
