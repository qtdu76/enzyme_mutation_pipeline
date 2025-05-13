import numpy as np
from .alignment import extract_pdb_sequence, align_sequences
from scripts.logging_config import setup_logger

def calculate_distance_matrix(native_coords, template_coords):
    """
    calculate the distance between corresponding residue pairs
    parameters:
    native_coords: (N, 3)
    template_coords: (N, 3)
    return:
    distances: (N,)
    """
    return np.sqrt(np.sum((native_coords - template_coords) ** 2, axis=1))

def kabsch(native_coords, template_coords):
    """
    kabsch superposition
    parameters:
    native_coords: (N, 3)
    template_coords: (N, 3)
    return:
    - R: rotation matrix (3, 3)
    - translation: translation vector (3,)
    """
    centroid_native_coords = np.mean(native_coords, axis=0)
    centroid_template_coords = np.mean(template_coords, axis=0)
    native_coords_centered = native_coords - centroid_native_coords
    template_coords_centered = template_coords - centroid_template_coords

    H = template_coords_centered.T @ native_coords_centered
    U, S, Vt = np.linalg.svd(H)

    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] = -Vt[-1, :]
        R = U @ Vt
    translation = centroid_native_coords - centroid_template_coords @ R

    return R, translation

def read_pdb(file_path, chain_id=None):
    """
    Read PDB file and return atomic coordinates of all CA atoms, optionally filtering by chain.

    Args:
        file_path (str): Path to the PDB file.
        chain_id (str or None): Chain ID to filter by. If None, include all chains.

    Returns:
        np.ndarray: Atomic coordinates of all CA atoms (N, 3).
    """
    coords = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                if chain_id is None or line[21].strip() == chain_id:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
    return np.array(coords)



# calculate TM-score
def calculate_tm_score(native_coords, template_coords, d0):
    """
    calculate the TM-score between two sets of atomic coordinates
    parameters:
    native_coords: (N, 3)
    template_coords: (N, 3)
    return:
    tm_score
    """
    L_native = len(native_coords) 
    
    # calculate the distance between corresponding residue pairs
    distances = calculate_distance_matrix(native_coords, template_coords)
    
    # calculate TM-score
    tm_score = np.sum(1 / (1 + (distances / d0) ** 2)) / L_native
    
    return tm_score


def calc_TMs(reference_file, target_file, protected_esm_indices, esm_sequence, pdb_code):
    """
    Calculate TM-scores for the whole structure and the protected region using aligned residue mapping.
    """
    logger = setup_logger("TMscore", pdb_code)

    # Step 1: Extract residue IDs and sequence from reference
    ref_res_ids, ref_seq = extract_pdb_sequence(reference_file)

    # Step 2: Align ESM sequence to reference PDB sequence
    alignment = align_sequences(esm_sequence, ref_seq, ref_res_ids)

    # Step 3: Get coordinate maps
    ref_coords = read_pdb(reference_file, chain_id="A")  # returns {res_id: coord}
    target_coords = read_pdb(target_file, chain_id="A")  # same here

    logger.info(f"Ref CA count: {len(ref_coords)} | Target CA count: {len(target_coords)}")

    # === Global TM-score ===
    try:
        ref_all = np.array([ref_coords[rid] for rid in ref_res_ids if rid in ref_coords])
        target_all = np.array([target_coords[i - 1] for i in range(1, len(esm_sequence)+1) if (i - 1) in target_coords])

        if len(ref_all) != len(target_all):
            raise ValueError(f"Global TM-score mismatch: ref={len(ref_all)}, target={len(target_all)}")

        d0 = max(1.0, 1.24 * (len(ref_all) - 15) ** (1 / 3) - 1.8)
        R, t = kabsch(ref_all, target_all)
        aligned = target_all @ R + t
        tm_whole = calculate_tm_score(ref_all, aligned, d0)

    except Exception as e:
        logger.error(f"Failed to compute global TM-score: {e}")
        tm_whole = None

    # === Protected TM-score ===
    try:
        protected_pairs = [ (esm_i, pdb_id) for esm_i, pdb_id in alignment if esm_i in protected_esm_indices ]

        missing_ref = [p for _, p in protected_pairs if p not in ref_coords]
        missing_target = [i for i, _ in protected_pairs if (i - 1) not in target_coords]

        if missing_ref or missing_target:
            logger.warning("⚠️ Missing protected residues:")
            if missing_ref:
                logger.warning(f"Missing in reference (IDs): {missing_ref}")
            if missing_target:
                logger.warning(f"Missing in target (ESM indices): {missing_target}")

        valid_pairs = [ (esm_i, pdb_id) for esm_i, pdb_id in protected_pairs
                        if pdb_id in ref_coords and (esm_i - 1) in target_coords ]

        if not valid_pairs:
            raise ValueError("No valid protected residue pairs found.")

        ref_prot = np.array([ref_coords[pdb_id] for _, pdb_id in valid_pairs])
        target_prot = np.array([target_coords[esm_i - 1] for esm_i, _ in valid_pairs])

        d0_prot = max(1.0, 1.24 * (len(ref_prot) - 15) ** (1 / 3) - 1.8)
        R_prot, t_prot = kabsch(ref_prot, target_prot)
        aligned_prot = target_prot @ R_prot + t_prot
        tm_protected = calculate_tm_score(ref_prot, aligned_prot, d0_prot)

    except Exception as e:
        logger.error(f"Failed to compute protected TM-score: {e}")
        tm_protected = None

    logger.info(f"TM Global: {tm_whole} | TM Protected: {tm_protected}")
    return tm_whole, tm_protected

