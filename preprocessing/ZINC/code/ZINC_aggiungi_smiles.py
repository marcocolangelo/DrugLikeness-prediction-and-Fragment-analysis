import pandas as pd
import os

def subtract_datasets(file1, file2):



    dataset1 = pd.read_csv(file1, header=None, names=['colonna'])
    dataset2 = pd.read_csv(file2)['ZINCID']

    # Assicurati che entrambe le colonne di confronto siano stringhe
    dataset1['colonna'] = dataset1['colonna'].astype(str)
    dataset2 = dataset2.astype(str)

    # Trova le differenze ovvero gli elementi presenti in dataset1 ma non in dataset2
    diff1_to_2 = dataset1[~dataset1['colonna'].isin(dataset2)]

    print(f"Numero di elementi in dataset1: {len(dataset1)}")

    # Salva i risultati su file
    diff1_to_2.to_csv('ZINC\\new\ZINC_in_vivo_nuovi_samples.csv', index=False, header=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Subtract two CSV datasets and save the differences.")
    parser.add_argument("file1", help="Path to the first CSV file")
    parser.add_argument("file2", help="Path to the second CSV file")
    # parser.add_argument("output_dir", help="Directory to save the output files")

    args = parser.parse_args()

    subtract_datasets(args.file1, args.file2)
