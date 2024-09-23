import pandas as pd
from sklearn.utils import shuffle

# Supponiamo di avere un dataset caricato in un DataFrame
# Carica il tuo dataset
df = pd.read_csv('ZINC\\new2\\new2_ZINC_smiles_output.csv')  # Sostituisci 'tuo_dataset.csv' con il tuo file

# Visualizza la distribuzione delle classi
print("Distribuzione delle classi prima del bilanciamento:")
print(df['Drug'].value_counts())

# Separare le classi
class_0 = df[df['Drug'] == 0]
class_1 = df[df['Drug'] == 1]

# Determina il numero minimo di campioni tra le due classi
min_samples = min(len(class_0), len(class_1))

# Sottocampiona le classi per avere lo stesso numero di campioni
class_0_under = class_0.sample(n=min_samples, random_state=42)
class_1_under = class_1.sample(n=min_samples, random_state=42)

# Unisci le due classi sottocampionate
df_balanced = pd.concat([class_0_under, class_1_under])

# Mescola il dataset bilanciato
df_balanced = shuffle(df_balanced, random_state=42)

# Visualizza la nuova distribuzione delle classi
print("\nDistribuzione delle classi dopo il bilanciamento:")
print(df_balanced['Drug'].value_counts())

# Salva il dataset bilanciato nella cartella specificata
df_balanced.to_csv('ZINC/new2/new2_ZINC_balanced_smiles_output.csv', index=False)
