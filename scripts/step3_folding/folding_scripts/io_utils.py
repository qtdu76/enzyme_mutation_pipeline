import pandas as pd
import numpy as np
from Bio.PDB import PDBParser
from torch.utils._pytree import tree_map

# Handles input/output operations, including loading CSV data, extracting metadata, and saving PDB files.

def load_csv_data(csv_file):
    # Dynamically load a CSV file containing sequence data, skipping non-data header rows.
    with open(csv_file, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith("Iteration"):
                skiprows = i
                break
    df = pd.read_csv(csv_file, skiprows=skiprows)
    return df


def extract_global_metadata(csv_file):
    # Extract global metadata from the CSV file.
    global_metadata = {}
    with open(csv_file, 'r') as file:
        for line in file:
            if not line.strip():  # Skip empty lines
                continue
            if line.startswith("Simulation Parameters"):  # Skip this header line
                continue
            if "Iteration" in line:  # Stop when reaching the main data section
                break
            if ',' in line:  # Only process lines with a key-value pair
                key, value = line.strip().split(',', 1)
                global_metadata[key.strip()] = value.strip()
    return global_metadata



def load_reference_coords_from_pdb(pdb_file):
    # Load atomic positions from a PDB file as reference coordinates.
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("reference", pdb_file)
    coords = [atom.coord for atom in structure.get_atoms()]
    return np.array(coords)


def load_reference_coordinates(method, global_metadata=None, model_wrapper=None, pdb_file=None):
    """
    Load reference coordinates based on the specified method.

    Args:
        method (str): Method to load coordinates ("csv" or "pdb").
        global_metadata (dict, optional): Metadata for the CSV method.
        model_wrapper (optional): Model wrapper for the CSV method.
        pdb_file (str, optional): Path to the PDB file for the PDB method.

    Returns:
        np.ndarray: Reference coordinates as a NumPy array.

    Raises:
        ValueError: If inputs are missing or the method is invalid.
    """
    if method == "csv":
        if global_metadata is None or model_wrapper is None:
            raise ValueError("Global metadata and model wrapper must be provided for the 'csv' method.")
        #logger.info("Loading reference coordinates from the CSV metadata...")
        reference_output = model_wrapper.generate_positions([global_metadata["Protein Sequence"]])
        return reference_output["positions"][0].cpu().numpy()  # Shape: [num_residues, 37, 3]

    elif method == "pdb":
        if not pdb_file:
            raise ValueError("PDB file must be provided for the 'pdb' method.")
        #logger.info("Loading reference coordinates from the PDB file...")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("reference", pdb_file)
        coords = [atom.coord for atom in structure.get_atoms()]
        return np.array(coords)

    else:
        raise ValueError("Invalid method. Use 'csv' or 'pdb'.")


def generate_pdb(model, raw_output, idx):
    """
    Generates a PDB string from the raw model output using output_to_pdb.
    Args:
        model: EsmForProteinFolding model instance.
        raw_output: Dictionary returned by the model (already computed in run()).
        idx: Index of the sequence in the batch.
    Returns:
        pdb_string: A PDB-format string for the sequence at `idx`.
    """
    # Move everything to CPU for compatibility
    raw_output_cpu = tree_map(lambda x: x.to("cpu"), raw_output)
    pdb_strings = model.output_to_pdb(raw_output_cpu)
    return pdb_strings[idx]


def save_pdb_with_metadata(output_structure, metadata, global_metadata, filename):
    # Save the ESMFold output structure as a PDB file with metadata.
    pdb_string = output_structure['pdb']
    metadata_header = "HEADER    ESMFOLD GENERATED MODEL\n"
    for key, value in global_metadata.items():
        metadata_header += f"REMARK    {key}: {value}\n"
    for key, value in metadata.items():
        metadata_header += f"REMARK    {key}: {value}\n"
    pdb_with_metadata = metadata_header.strip() + "\n" + pdb_string
    with open(filename, "w") as pdb_file:
        pdb_file.write(pdb_with_metadata)