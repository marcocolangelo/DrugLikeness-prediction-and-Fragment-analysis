import pandas as pd

def remove_common_elements(file1_path, file2_path, output_path, column_name):
    # Leggi i due file CSV in DataFrame
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Trova gli elementi comuni nella colonna specificata
    common_elements = set(df1[column_name]).intersection(set(df2[column_name]))

    # Filtra il secondo DataFrame rimuovendo le righe con elementi comuni
    df2_filtered = df2[~df2[column_name].isin(common_elements)]

    # Salva il risultato in un nuovo file CSV
    df2_filtered.to_csv(output_path, index=False)

    # Calcola la quantità di elementi eliminati
    num_elements_removed = len(df2) - len(df2_filtered)

    print(f"Righe comuni rimosse: {num_elements_removed}. Risultato salvato in {output_path}")

# Percorsi dei file
file1_path = 'ZINC\\new\\ZINC_final_balanced_test_set.csv'
file2_path = 'ZINC\\new\\ZINC_final_balanced_validation_set.csv'
output_path = 'ZINC\\new\\output.csv'
column_name = 'smiles'  # Sostituisci con il nome della colonna di interesse

# Esegui la funzione
remove_common_elements(file1_path, file2_path, output_path, column_name)
