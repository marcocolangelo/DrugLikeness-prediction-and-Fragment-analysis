import pandas as pd
from sklearn.model_selection import train_test_split

# Read the input CSV file
data = pd.read_csv("TCMBANK/NEW_drugs_labelled.csv")

# Split the data into train and test sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Set the desired header for the train and test sets
header = ["Smiles", "Drug"]

# Save the train and test sets to separate CSV files with the desired header
train_data.to_csv("TCMBANK_drug_train.csv", index=False, header=header)
test_data.to_csv("TCMBANK_drug_test.csv", index=False, header=header)