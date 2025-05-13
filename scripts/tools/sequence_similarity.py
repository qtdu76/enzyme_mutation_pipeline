import pandas as pd
from Bio.Align import substitution_matrices

# === Load file and extract metadata ===
csv_path = "/rds/general/user/qt121/home/pipeline/output/1ua7/mutation/beta_1000.0/msk_1ua7_beta_1000.0_simRes.csv"
output_path = "/rds/general/user/qt121/home/pipeline/output/1ua7/mutation/beta_1000.0/sequence_identity_similarity_output.csv"

with open(csv_path, "r") as f:
    lines = f.readlines()

# Extract metadata lines (first 6 lines) and reference sequence
metadata_lines = lines[:6]
ref_line = next(line for line in metadata_lines if line.startswith("Protein Sequence"))
reference_sequence = ref_line.strip().split(",")[1]

# Read data starting from row 7 (index 6)
df = pd.read_csv(csv_path, skiprows=6)

# === Load BLOSUM62 matrix ===
blosum62 = substitution_matrices.load("BLOSUM62")

# === Compute identity ===
def compute_identity(seq1, seq2):
    matches = sum(a == b for a, b in zip(seq1, seq2))
    return matches / len(seq1) * 100

# === Compute similarity (position-by-position, no alignment) ===
def compute_similarity(seq1, seq2, matrix):
    total = 0
    max_total = 0
    for a, b in zip(seq1, seq2):
        if a == "-" or b == "-":
            continue
        pair = (a, b)
        reverse_pair = (b, a)
        score = matrix.get(pair) or matrix.get(reverse_pair) or 0
        max_score = matrix.get((a, a)) or 0
        total += score
        max_total += max_score
    return (total / max_total) * 100 if max_total else 0

# === Apply to all sequences ===
identities = []
similarities = []

for seq in df['Sequence']:
    identities.append(compute_identity(reference_sequence, seq))
    similarities.append(compute_similarity(reference_sequence, seq, blosum62))

# Add results to DataFrame
df['Sequence Identity (%)'] = identities
df['Sequence Similarity (%)'] = similarities

# === Save output including metadata ===
with open(output_path, "w") as f:
    # Write metadata lines
    for line in metadata_lines:
        f.write(line)
    # Write updated DataFrame
    df.to_csv(f, index=False)
