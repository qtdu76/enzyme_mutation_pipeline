import pandas as pd
import ast

# Path to your CSV file
csv_file = "/rds/general/user/qt121/home/pipeline/output/1bn7/mutation/beta_150.0/msk_1bn7_beta_150.0_simRes.csv"

# Load CSV, skipping metadata rows
df = pd.read_csv(csv_file, skiprows=6)

# Read metadata manually to get the protected indices
with open(csv_file, "r") as f:
    lines = f.readlines()

protected_indices_str = lines[4].split(",", 1)[1].strip()  # "Protected Indices" is in row 5

print("Raw protected_indices_str:", repr(protected_indices_str))

# Convert protected indices from string to a list
protected_indices = ast.literal_eval(protected_indices_str.replace('"', '').strip())

# Extract only mutation rows (skip rows with missing iteration numbers)
mutation_rows = df[df['Iteration'].notna()].copy()

# Convert 1-based protected indices to 0-based
protected_indices = [idx - 1 for idx in protected_indices]

# Ensure sequences are strings
mutation_rows['Sequence'] = mutation_rows['Sequence'].astype(str).str.strip()

# Get the first and last mutated sequences
first_sequence = mutation_rows.iloc[0]['Sequence']
last_sequence = mutation_rows.iloc[-1]['Sequence']

# Generate dictionaries for first and last sequences
first_seq_dict = {idx: first_sequence[idx] for idx in protected_indices}
last_seq_dict = {idx: last_sequence[idx] for idx in protected_indices}

# Print results
print("\n🔹 First Mutated Sequence (Protected Residues):")
print(first_seq_dict)

print("\n🔹 Last Mutated Sequence (Protected Residues):")
print(last_seq_dict)
