
# file1 = "ZINC\\new\\ZINC_in_vivo_nuovi_samples.csv"
# output_file = "ZINC\\new\\ZINC_in_vivo_nuovi_samples_labelled.csv"

# header = "ZINC,label\n"

# with open(file1, 'r') as f1, open(output_file, 'w') as output:
#     output.write(header)
    
#     # Scrivi gli elementi del primo file con etichetta 0
#     for line in f1:
#         output.write(f"{line.strip()},1\n")

import pandas as pd

input_file = "ZINC\\new\\ZINC_in_vivo_nuovi_samples_smiles_output.csv"
df = pd.read_csv(input_file)  # Sostituisci 'tuo_dataset.csv' con il tuo file

df["Drug"] = 1

df.to_csv('ZINC\\new\\NEW_ZINC_in_vivo_nuovi_samples_smiles_output.csv', index=False, header=False)