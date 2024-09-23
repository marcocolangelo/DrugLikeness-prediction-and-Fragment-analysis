import pandas as pd
import os

def subtract_datasets(file1, file2):



    dataset1 = pd.read_csv(file1, header=None, names=['colonna'])
    dataset2 = pd.read_csv(file2, header=None, names=['colonna'])

    # Assicurati che entrambe le colonne di confronto siano stringhe
    dataset1['colonna'] = dataset1['colonna'].astype(str)
    dataset2['colonna'] = dataset2['colonna'].astype(str)

    # Trova le differenze
    diff1_to_2 = dataset1[~dataset1['colonna'].isin(dataset2['colonna'])]
    # diff2_to_1 = dataset2[~dataset2['colonna'].isin(dataset1['colonna'])]

    # Stampa i risultati
    print("Presenti in dataset1 ma non in dataset2:")
    print(diff1_to_2)

    print("\nPresenti in dataset2 ma non in dataset1:")
    print(dataset2)

    # Salva i risultati su file
    diff1_to_2.to_csv('diff1_to_2.csv', index=False, header=False)
    dataset2.to_csv('diff2_to_1.csv', index=False, header=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Subtract two CSV datasets and save the differences.")
    parser.add_argument("file1", help="Path to the first CSV file")
    parser.add_argument("file2", help="Path to the second CSV file")
    # parser.add_argument("output_dir", help="Directory to save the output files")

    args = parser.parse_args()

    subtract_datasets(args.file1, args.file2)
