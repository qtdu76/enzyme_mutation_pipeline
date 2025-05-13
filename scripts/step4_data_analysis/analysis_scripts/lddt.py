import numpy as np
from .tm_score import read_pdb
from scripts.logging_config import setup_logger


def calculate_pairwise_distances(coords):
    """
    Compute pairwise distances for a set of coordinates.

    Args:
        coords (np.ndarray): Atomic coordinates (N, 3).
    
    Returns:
        np.ndarray: Pairwise distance matrix (N, N).
    """
    # Calculate pairwise distances
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    
    # Debugging: Print the shape and a sample of the distance matrix
    #print(f"Pairwise distance matrix calculated. Shape: {distances.shape}")
    #print(f"Sample distances (first 5 rows):\n{distances[:5, :5]}")

    return distances

def apply_cutoff(distances, cutoff=15.0):
    """
    Apply a cutoff to a pairwise distance matrix.

    Args:
        distances (np.ndarray): Pairwise distance matrix (N, N).
        cutoff (float): Distance cutoff in Å.
    
    Returns:
        np.ndarray: Boolean mask of distances within the cutoff.
    """
    return distances <= cutoff

def calculate_preservation_fractions(ref_distances, model_distances, neighbor_mask, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """
    Calculate preservation fractions for given thresholds, avoiding double-counting.

    Args:
        ref_distances (np.ndarray): Reference pairwise distances (N, N).
        model_distances (np.ndarray): Model pairwise distances (N, N).
        neighbor_mask (np.ndarray): Boolean mask of valid pairs.
        thresholds (tuple): Thresholds for distance preservation.
    
    Returns:
        list: Preservation fractions for each threshold.
    """
    # Use only the upper triangle (excluding diagonal) to avoid double-counting
    upper_triangle = np.triu(np.ones(ref_distances.shape, dtype=bool), k=1)
    effective_mask = neighbor_mask & upper_triangle  # Combine neighbor mask with upper triangle

    fractions = []
    for t in thresholds:
        preserved = (np.abs(ref_distances - model_distances) < t) & effective_mask
        fraction = np.sum(preserved) / np.sum(effective_mask)
        fractions.append(fraction)
    return fractions

def calculate_lddt(ref_coords, model_coords, cutoff=15.0, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """
    Calculate the lDDT score between reference and model structures.

    Args:
        ref_coords (np.ndarray): Reference atomic coordinates (N, 3).
        model_coords (np.ndarray): Model atomic coordinates (N, 3).
        cutoff (float): Distance cutoff in Å.
        thresholds (tuple): Thresholds for distance preservation.
    
    Returns:
        float: Global lDDT score.
    """
    # Step 1: Compute pairwise distances
    ref_distances = calculate_pairwise_distances(ref_coords)
    model_distances = calculate_pairwise_distances(model_coords)

    #print(f"Ref distances (shape {ref_distances.shape}):\n{ref_distances[:5, :5]}")
    #print(f"Model distances (shape {model_distances.shape}):\n{model_distances[:5, :5]}")

    # Step 2: Apply cutoff
    neighbor_mask = apply_cutoff(ref_distances, cutoff)
    #print(f"Neighbor mask (shape {neighbor_mask.shape}):\n{neighbor_mask[:5, :5]}")
    #print(f"Total pairs within cutoff: {np.sum(neighbor_mask)}")

    # Step 3: Calculate preservation fractions
    fractions = calculate_preservation_fractions(ref_distances, model_distances, neighbor_mask, thresholds)

    #print(f"Preservation fractions: {fractions}")

    # Step 4: Compute  lDDT
    global_lddt = np.mean(fractions)

    return global_lddt

def calc_lddt(reference_file, target_file, protected_indices, pdb_code, cutoff=15.0, thresholds=(0.5, 1.0, 2.0, 4.0)):
    """
    Calculate lDDT scores for the full structure and protected region.

    Args:
        reference_file (str): Path to reference PDB.
        target_file (str): Path to target PDB.
        protected_indices (list[int]): 1-based protected residue indices.
        pdb_code (str): For logging.
        cutoff (float): Distance cutoff.
        thresholds (tuple): Distance thresholds.

    Returns:
        tuple: (global lDDT, protected lDDT)
    """
    logger = setup_logger("lDDT", pdb_code)
    logger.info(f"Calculating lDDT | Cutoff: {cutoff}, Thresholds: {thresholds}")

    # Read coordinates
    ref_coords = read_pdb(reference_file, chain_id="A")
    target_coords = read_pdb(target_file, chain_id="A")

    logger.info(f"Loaded coordinates — ref: {ref_coords.shape}, target: {target_coords.shape}")

    # Global lDDT
    lddt_global = calculate_lddt(ref_coords, target_coords, cutoff, thresholds)
    logger.info(f"Global lDDT: {lddt_global:.4f}")

    # Protected indices
    idxs = np.array([i - 1 for i in protected_indices])  # convert to 0-based
    ref_prot = ref_coords[idxs]
    target_prot = target_coords[idxs]

    if len(ref_prot) != len(target_prot):
        logger.error("Mismatch in protected region lengths")
        raise ValueError(f"Protected ref: {len(ref_prot)}, target: {len(target_prot)}")

    lddt_protected = calculate_lddt(ref_prot, target_prot, cutoff, thresholds)
    logger.info(f"Protected lDDT: {lddt_protected:.4f}")

    return lddt_global, lddt_protected


