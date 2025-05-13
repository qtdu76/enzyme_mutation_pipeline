import torch

# Define the function to get embeddings
def get_embeddings(sequence, esm2_pipeline):
    """
    Generate embeddings for a given sequence using the esm2_pipeline.
    
    Args:
        sequence (str): The sequence for which embeddings are generated.
        esm2_pipeline: The pipeline object used to compute embeddings.
        
    Returns:
        torch.Tensor: The embedding tensor for the given sequence.
    """
    # Get the embeddings
    embeddings = esm2_pipeline(sequence)
    
    # Convert to a torch tensor
    embeddings_tensor = torch.tensor(embeddings[0])  # Assuming one sequence
    
    return embeddings_tensor

class SequenceEmbeddingPair:
    def __init__(self, sequence, embedding, protected_indices_0=None):
        """
        Initialize a SequenceEmbeddingPair object.

        Parameters:
            sequence (str): The protein sequence.
            embedding (torch.Tensor): The corresponding embedding for the sequence.
            protected_indices_0 (list[int], optional): 0-based indices that must not be mutated.
        """
        self.sequence = sequence
        self.embedding = embedding
        self.protected_indices_0 = protected_indices_0 if protected_indices_0 else []

    def __repr__(self):
        protected = ", ".join(map(str, self.protected_indices_0[:5]))
        more = "..." if len(self.protected_indices_0) > 5 else ""
        return (f"SequenceEmbeddingPair(sequence={self.sequence[:10]}..., "
                f"embedding_shape={self.embedding.shape}, "
                f"protected_indices_0=[{protected}{more}])")


def calc_embedding_diff2ref(embedding1, embedding2, Eref, protected_indices_0):
    """
    Computes the embedding similarity difference (ΔE) using only protected residues.

    Parameters:
    - embedding1: Tensor of previous sequence embeddings.
    - embedding2: Tensor of mutated sequence embeddings.
    - Eref: Tensor of reference embeddings.
    - protected_indices_0: List or Tensor of protected residue indices (0-based).

    Returns:
    - dE: Change in similarity score after mutation.
    - Emut: Similarity score of mutated sequence.
    """

    # ✅ Normalize embeddings to unit vectors
    norms1 = torch.norm(embedding1, p=2, dim=1, keepdim=True)
    norms2 = torch.norm(embedding2, p=2, dim=1, keepdim=True)
    normsR = torch.norm(Eref, p=2, dim=1, keepdim=True)

    normalized_embedding1 = embedding1 / norms1
    normalized_embedding2 = embedding2 / norms2
    normalized_Eref = Eref / normsR

    # ✅ Extract only the protected rows (0-based indices)
    embedding1_prot = normalized_embedding1[protected_indices_0]
    embedding2_prot = normalized_embedding2[protected_indices_0]
    Eref_prot = normalized_Eref[protected_indices_0]

    # ✅ Compute per-residue dot products for previous sequence
    per_amino_acid_dot_productsPrev = torch.sum(embedding1_prot * Eref_prot, dim=1)
    total_similarity_scorePrev = torch.sum(per_amino_acid_dot_productsPrev).item()
    num_residuesPrev = per_amino_acid_dot_productsPrev.size(0)
    Eprev = 1 - (total_similarity_scorePrev / num_residuesPrev)

    # ✅ Compute per-residue dot products for mutated sequence
    per_amino_acid_dot_productsMut = torch.sum(embedding2_prot * Eref_prot, dim=1)
    total_similarity_scoreMut = torch.sum(per_amino_acid_dot_productsMut).item()
    num_residuesMut = per_amino_acid_dot_productsMut.size(0)
    Emut = 1 - (total_similarity_scoreMut / num_residuesMut)

    # ✅ Compute ΔE
    dE = Emut - Eprev

    return dE, Emut

