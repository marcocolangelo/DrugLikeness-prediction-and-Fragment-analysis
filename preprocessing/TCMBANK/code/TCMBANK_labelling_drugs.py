import pandas as pd

# Read the input file
input_file = 'TCMBANK/VERY_NEW_non_drugs.csv'
df = pd.read_csv(input_file)

# Create a new DataFrame with desired columns
new_df = pd.DataFrame({'Smiles': df['Smiles'], 'Drug': 0})

# Write the new DataFrame to a new file
output_file = 'TCMBANK/VERY_NEW_non_drugs_labelled.csv'
new_df.to_csv(output_file, index=False)