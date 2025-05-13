import random
from .embedding_utils import get_embeddings, SequenceEmbeddingPair

def point_mutation(sequence, protected_indices_0):
    """
    Perform a single point mutation on a protein sequence while preserving protected residues.

    Parameters:
    - sequence (str): The current protein sequence.
    - protected_indices_0 (list[int]): 0-based indices that must not be mutated.

    Returns:
    - mutated_sequence (str): The newly mutated sequence.
    """
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"

    # ✅ Ensure there are available mutation positions
    available_positions = [i for i in range(len(sequence)) if i not in protected_indices_0]
    if not available_positions:
        raise ValueError("No available positions for mutation. All positions are protected.")

    # ✅ Randomly select a mutation position
    position = random.choice(available_positions)

    # ✅ Ensure the selected position is not in protected_indices_0 (for safety)
    assert position not in protected_indices_0, (
        f"❌ ERROR: Attempted to mutate a protected index {position + 1} (1-based)!"
    )

    # ✅ Select a random new amino acid (different from original)
    original_residue = sequence[position]
    new_amino_acid = random.choice([aa for aa in amino_acids if aa != original_residue])

    # ✅ Apply mutation while maintaining sequence length
    mutated_sequence = sequence[:position] + new_amino_acid + sequence[position + 1:]

    return mutated_sequence


def mutate_sequence_pair(prev_pair, esm2_pipeline, protected_indices_0):
    """
    Mutates the sequence in prev_pair and returns a new SequenceEmbeddingPair
    for the mutated sequence, along with the previous SequenceEmbeddingPair.

    Parameters:
    - prev_pair: SequenceEmbeddingPair representing the last accepted sequence and its embedding.
    - esm2_pipeline: Model or pipeline used to generate embeddings.
    - protected_indices_0: A list of indices (0-based) that must not be mutated.

    Returns:
    - A tuple containing:
        1. mut_pair: SequenceEmbeddingPair of the mutated sequence.
        2. prev_pair: SequenceEmbeddingPair of the original (previous) sequence.
    """

    prev_sequence = prev_pair.sequence

    # ✅ Apply mutation
    mut_sequence = point_mutation(prev_sequence, protected_indices_0)

    # 🚨 Debugging Assertion: Ensure protected residues remain unchanged
    assert all(mut_sequence[idx] == prev_sequence[idx] for idx in protected_indices_0), (
        f"❌ ERROR: Protected residues were altered!\n"
        f"Original:  {prev_sequence}\n"
        f"Mutated:   {mut_sequence}\n"
        f"Protected Indices (0-based): {protected_indices_0}"
    )

    # ✅ Generate embeddings for the mutated sequence
    mut_embeddings = get_embeddings(mut_sequence, esm2_pipeline)
    mut_pair = SequenceEmbeddingPair(mut_sequence, mut_embeddings)

    return mut_pair, prev_pair

