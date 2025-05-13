import os
import pandas as pd
from .loading import extract_protected_indices, extract_sequence
from .rmsd_pymol import calc_rmsd_pymol
from .tm_score import calc_TMs
from .lddt import calc_lddt
from .loading import extract_global_metadata, extract_pdb_metadata, extract_hamdis_alphafold
from .rmsd_stef import calc_rmsd_stef
from scripts.logging_config import setup_logger
import traceback


def analyze_datasets(dataset, reference, output_dir, pdb_code, info_csv_path):
    """
    Analyze a dataset of PDB files and compute metadata, comparison metrics, and TM scores.

    Args:
        dataset (str): Path to the folder containing PDB files.
        reference (str): Path to the reference PDB file for comparisons.
        output_dir (str): Path to the folder where the output CSV file will be saved.
        pdb_code (str): PDB code of the enzyme (e.g., '1bn7') for consistent naming.    
    """

    logger = setup_logger('analysis', pdb_code)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize a list to accumulate rows
    rows = []
    
    # Extract PDB files from the dataset
    pdb_files = sorted([f for f in os.listdir(dataset) if f.endswith(".pdb")])
    num_files = len(pdb_files)


    # Extract global metadata from the first available file
    global_metadata = extract_global_metadata(os.path.join(dataset, pdb_files[0]))
    logger.info(f"Global metadata extracted: {global_metadata}")

    # Extract protected indices and PyMOL selection string
    sequence = extract_sequence(info_csv_path)
    protected_fasta_indices, protected_pdb_indices = extract_protected_indices(info_csv_path)
    logger.info(f"Protected fasta indices extracted: {protected_fasta_indices}")


    # Iterate through all PDB files in the dataset
    for i, pdb_file in enumerate(pdb_files):
        pdb_path = os.path.join(dataset, pdb_file)
        logger.info(f"Processing file {i + 1}/{num_files}: {pdb_file}")

        # Calculate RMSD
        try:
            rmsd_global_pymol, rmsd_protected_pymol = calc_rmsd_pymol(reference, pdb_path, pdb_code, protected_fasta_indices, protected_fasta_indices) #if you are using esm as a reference, use fasta twice. 
            rmsd_global_stef, rmsd_protected_stef = calc_rmsd_stef(reference, pdb_path, protected_fasta_indices, pdb_code, output_dir)
        except Exception as e:
            logger.error(f"Error calculating RMSDs for {pdb_file}: {e}")
            logger.error(f"Error calculating RMSDs for {pdb_file}:\n{traceback.format_exc()}")
            rmsd_global_pymol, rmsd_protected_pymol = None, None
            rmsd_global_stef, rmsd_protected_stef = None, None

        # Calculate TM-score
        try:
            tm_whole, tm_protected = calc_TMs(reference, pdb_path, protected_fasta_indices, sequence, pdb_code)

        except Exception as e:
            logger.error(f"Error calculating TM-score for {pdb_file}: {e}")
            tm_whole, tm_protected = None, None

        # Calculate lDDT scores
        try:
            lddt_whole, lddt_protected = calc_lddt(reference, pdb_path, protected_fasta_indices, pdb_code)

        except Exception as e:
            logger.error(f"Error calculating lDDt for {pdb_file}: {e}")
            lddt_whole, lddt_protected = None, None

        # Extract additional metadata
        iteration, delta_e, hamming_distance, emut_or_step_delta_e = extract_pdb_metadata(pdb_path)
        if hamming_distance is None:
            hamming_distance = extract_hamdis_alphafold(pdb_path)

        logger.info(f"Iteration: {iteration}, Delta E: {delta_e}, Emut/Step Delta E: {emut_or_step_delta_e}, Hamming Distance: {hamming_distance}")

        # Append data to rows
        rows.append({
            "PDB File": pdb_file,
            "Iteration": iteration,
            "Delta E": delta_e,
            "Emut": emut_or_step_delta_e,
            "Hamming Distance": hamming_distance,
            "RMSD_pymol_global": rmsd_global_pymol,
            "RMSD_pymol_local": rmsd_protected_pymol,
            "RMSD_stef_global": rmsd_global_stef,
            "RMSD_stef_local": rmsd_protected_stef,
            "TM_score_global": tm_whole,
            "TM_score_local": tm_protected,
            "lDDT_global": lddt_whole,
            "lDDT_local": lddt_protected
        })



    # Convert accumulated rows into a DataFrame
    results = pd.DataFrame(rows)

    # Prepare metadata comments
    metadata_comments = [
        f"# PDB Code: {pdb_code}",
        f"# Beta: {global_metadata['Beta']}",
        f"# Mutation Method: Point Mutation", #this is hardcoded
        f"# Embedding Difference Method: {global_metadata['Embedding Difference Method']}",
        f"# Reference File: {os.path.basename(reference)}",
        f"# Protected Indices: \"{str(global_metadata['Protected Indices'])}\""
    ]
    metadata_comments_text = "\n".join(metadata_comments) + "\n"

    # Define output CSV filename
    beta_value = global_metadata["Beta"]

    output_csv = os.path.join(output_dir, f"esmRef_{pdb_code}_b_{beta_value}.csv")

    # Write results to CSV
    with open(output_csv, "w") as f:
        f.write(metadata_comments_text)  # Write metadata as comments
    results.to_csv(output_csv, mode="a", index=False)  # Append results DataFrame

    print(f"✅ Analysis completed. Results saved to {output_csv}")
