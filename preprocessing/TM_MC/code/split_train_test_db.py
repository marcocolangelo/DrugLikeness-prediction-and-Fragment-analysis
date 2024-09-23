import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

def prepare_datasets(file, output_train, output_test):
    # Legge il file CSV
    df = pd.read_csv(file)

    # Assicurarsi che il campo "smiles" e "label" siano presenti nel file
    if 'smiles' not in df.columns or 'Drug' not in df.columns:
        raise ValueError('Il file deve contenere i campi "smiles" e "label"')

    # Divide il dataset in base alle etichette
    positive_df = df[df['Drug'] == 1]
    negative_df = df[df['Drug'] == 0]

    # Calcola il numero di campioni per ogni etichetta
    num_positive_samples = len(positive_df)
    num_negative_samples = len(negative_df)

    # Calcola il numero di campioni per l'addestramento e il test
    num_train_positive = int(num_positive_samples * 0.8)
    num_train_negative = int(num_negative_samples * 0.8)
    num_test_positive = num_positive_samples - num_train_positive
    num_test_negative = num_negative_samples - num_train_negative

    # Estrae i campioni per l'addestramento e il test
    train_positive_df = positive_df.sample(n=num_train_positive, random_state=42)
    train_negative_df = negative_df.sample(n=num_train_negative, random_state=42)
    test_positive_df = positive_df.drop(train_positive_df.index)
    test_negative_df = negative_df.drop(train_negative_df.index)

    # Unisce i dataset di addestramento e test
    train_df = pd.concat([train_positive_df, train_negative_df], ignore_index=True)
    test_df = pd.concat([test_positive_df, test_negative_df], ignore_index=True)

    # Salva i risultati nei file di output
    train_df.to_csv(output_train, index=False)
    test_df.to_csv(output_test, index=False)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Divide compounds into training and test sets, balancing the distribution of positive and negative labels.")
    parser.add_argument("file", help="Path to the CSV file")
    parser.add_argument("output_train", help="Path to save the training CSV file")
    parser.add_argument("output_test", help="Path to save the test CSV file")

    args = parser.parse_args()

    prepare_datasets(args.file, args.output_train, args.output_test)
