import pandas as pd
from .embedding_utils import get_embeddings, SequenceEmbeddingPair
from .mutation_utils import point_mutation


def get_protein_sequence(info_csv_path):
    """
    Retrieves the protein sequence from the info.csv file generated during preprocessing.

    Parameters:
    - info_csv_path: Path to the info.csv file.

    Returns:
    - The protein sequence as a string.
    """
    # Load the info.csv file
    df = pd.read_csv(info_csv_path)
    
    if 'sequence' in df.columns:
        return df.at[0, 'sequence']
    else:
        raise ValueError("'sequence' column not found in info.csv.")
        
def generate_sequence_pairs(sequence, esm2_pipeline, protected_indices_0):
    """
    Generates the original and mutated sequence embedding pairs.

    Parameters:
    - sequence (str): The initial protein sequence.
    - esm2_pipeline: Model or pipeline used to generate embeddings.
    - protected_indices_0 (list[int]): 0-based indices that must not be mutated.

    Returns:
    - A tuple containing:
        1. prev_pair (SequenceEmbeddingPair of the original sequence)
        2. mut_pair (SequenceEmbeddingPair of the mutated sequence)
    """
    # ✅ Step 1: Retrieve the embedding for the initial sequence
    prev_embeddings = get_embeddings(sequence, esm2_pipeline)
    prev_pair = SequenceEmbeddingPair(sequence, prev_embeddings, protected_indices_0)

    # ✅ Step 2: Mutate the sequence using the standalone point_mutation function
    mut_sequence = point_mutation(sequence, protected_indices_0)

    # ✅ Step 3: Generate the embedding for the mutated sequence
    mut_embeddings = get_embeddings(mut_sequence, esm2_pipeline)
    mut_pair = SequenceEmbeddingPair(mut_sequence, mut_embeddings, protected_indices_0)

    return prev_pair, mut_pair

def hamming_distance(seq1, seq2):
    assert len(seq1) == len(seq2), "Sequences must be of equal length"
    return sum(c1 != c2 for c1, c2 in zip(seq1, seq2))