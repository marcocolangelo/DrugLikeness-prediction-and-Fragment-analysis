import pandas as pd

# Load raw dataset
data = pd.read_csv("data/raw/TCMBANK_total_small_db_train.csv")
data.index = data["index"]
data["Drug"].value_counts()
start_index = data.iloc[0]["index"]

# Apply oversampling

# Check how many additional samples we need
neg_class = data["Drug"].value_counts()[0]
pos_class = data["Drug"].value_counts()[1]
print(f"neg_class: {neg_class}")
print(f"pos_class: {pos_class}")
multiplier = int(pos_class/neg_class) - 1

print(f"Multiplier: {multiplier}")

# Replicate the dataset for the positive class
replicated_pos = [data[data["Drug"] == 1]] * multiplier

# Append replicated data
data = pd.concat([data] + replicated_pos, ignore_index=True)
print(data.shape)

# Shuffle dataset
data = data.sample(frac=1).reset_index(drop=True)

# Re-assign index (This is our ID later)
index = range(start_index, start_index + data.shape[0])
data.index = index
data["index"] = data.index
data.head()

# Save
data.to_csv("data/raw/TCMBANK_total_small_db_train_oversampled.csv")
