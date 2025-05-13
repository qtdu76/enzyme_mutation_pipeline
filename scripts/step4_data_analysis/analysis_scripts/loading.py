import re
import os
import pandas as pd

def extract_sequence(csv_path):
    """
    Extracts the ESM sequence from an info.csv file.

    Args:
        csv_path (str): Path to the metadata CSV.

    Returns:
        str: Amino acid sequence (string).
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV file {csv_path} is empty!")
    
    return str(df.iloc[0]["Sequence"]).strip()


def extract_protected_indices(csv_path):
    """
    Extract protected residue indices from a metadata CSV file.

    Args:
        csv_path (str): Path to the metadata CSV.

    Returns:
        Tuple[List[int], List[int]]: (fasta_indices, pdb_indices), both 1-based.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV file {csv_path} is empty!")

    row = df.iloc[0]

    fasta_indices = list(map(int, re.findall(r"\d+", str(row["Binding Residues (FASTA)"]))))
    pdb_indices = list(map(int, re.findall(r"\d+", str(row["Binding Residues (PDB)"]))))

    return fasta_indices, pdb_indices


def format_indices_for_pymol(indices):
    """
    Convert a list of indices into a PyMOL-compatible selection string.

    Args:
        indices (list): List of indices (integers).

    Returns:
        str: PyMOL-compatible selection string.
    """
    if not indices:
        return ""
    
    # Sort indices to ensure proper range grouping
    indices = sorted(indices)

    # Group consecutive indices into ranges
    ranges = []
    start = indices[0]
    for i in range(1, len(indices)):
        if indices[i] != indices[i - 1] + 1:  # Non-consecutive
            if start == indices[i - 1]:
                ranges.append(f"resi {start}")
            else:
                ranges.append(f"resi {start}-{indices[i - 1]}")
            start = indices[i]
    # Add the last range
    if start == indices[-1]:
        ranges.append(f"resi {start}")
    else:
        ranges.append(f"resi {start}-{indices[-1]}")

    # Join ranges with " or " to create the PyMOL selection string
    return " or ".join(ranges)


import re
import os

def extract_deltae_hamming(pdb_path):
    """
    Extract Delta E, Hamming Distance, and either Emut or Step Delta E from a PDB file.

    Args:
        pdb_path (str): Path to the PDB file.

    Returns:
        dict: A dictionary containing:
            - "Delta E" (float): The Delta E value from the file.
            - "Hamming Distance" (int): The Hamming Distance value from the file.
            - "Emut or Step Delta E" (float): Emut or Step Delta E (whichever is present).
    """
    result = {"Delta E": None, "Hamming Distance": None, "Emut or Step Delta E": None}

    # Check if the file exists
    if not os.path.exists(pdb_path):
        print(f"❌ Error: File '{pdb_path}' not found.")
        return result

    try:
        with open(pdb_path, 'r') as pdb_file:
            for line in pdb_file:
                # Extract Delta E (supports negative and scientific notation)
                if "Delta E:" in line and "Step Delta E" not in line:  # Avoid "Step Delta E" confusion
                    match = re.search(r"Delta E: ([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", line)
                    if match:
                        result["Delta E"] = float(match.group(1))

                # Extract Hamming Distance
                elif "Hamming Distance:" in line:
                    match = re.search(r"Hamming Distance: (\d+)", line)
                    if match:
                        result["Hamming Distance"] = int(match.group(1))

                # Extract either Emut or Step Delta E
                elif "Emut:" in line or "Step Delta E:" in line:
                    match = re.search(r"(?:Emut|Step Delta E): ([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", line)
                    if match:
                        result["Emut or Step Delta E"] = float(match.group(1))

                # Break early if all values are found
                if (
                    result["Delta E"] is not None 
                    and result["Hamming Distance"] is not None 
                    and result["Emut or Step Delta E"] is not None
                ):
                    break

        # # Log warnings if values are missing
        # if result["Delta E"] is None:
        #     print(f"⚠️ Warning: 'Delta E' not found in {pdb_path}.")
        # if result["Hamming Distance"] is None:
        #     print(f"⚠️ Warning: 'Hamming Distance' not found in {pdb_path}.")
        # if result["Emut or Step Delta E"] is None:
        #     print(f"⚠️ Warning: Neither 'Emut' nor 'Step Delta E' found in {pdb_path}.")

    except IOError as e:
        print(f"❌ Error reading file '{pdb_path}': {e}")

    return result

import os
import re

def extract_pdb_metadata(pdb_path):
    """
    Extracts key metadata from a PDB file, including Iteration, Delta E, 
    Hamming Distance, and either Emut or Step Delta E.

    Args:
        pdb_path (str): Path to the PDB file.

    Returns:
        tuple: (iteration, delta_e, hamming_distance, emut_or_step_delta_e)
    """
    iteration = None
    delta_e = None
    hamming_distance = None
    emut_or_step_delta_e = None

    # Check if the file exists
    if not os.path.exists(pdb_path):
        print(f"❌ Error: File '{pdb_path}' not found.")
        return iteration, delta_e, hamming_distance, emut_or_step_delta_e

    try:
        with open(pdb_path, 'r') as pdb_file:
            for line in pdb_file:
                # Extract Iteration
                if "Iteration:" in line:
                    match = re.search(r"Iteration: (\d+)", line)
                    if match:
                        iteration = int(match.group(1))

                # Extract Delta E (supports negative and scientific notation)
                elif "Delta E:" in line and "Step Delta E" not in line:  # Avoid "Step Delta E" confusion
                    match = re.search(r"Delta E: ([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", line)
                    if match:
                        delta_e = float(match.group(1))

                # Extract Hamming Distance
                elif "Hamming Distance:" in line:
                    match = re.search(r"Hamming Distance: (\d+)", line)
                    if match:
                        hamming_distance = int(match.group(1))

                # Extract either Emut or Step Delta E
                elif "Emut:" in line or "Step Delta E:" in line:
                    match = re.search(r"(?:Emut|Step Delta E): ([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)", line)
                    if match:
                        emut_or_step_delta_e = float(match.group(1))

                # Break early if all values are found
                if iteration is not None and delta_e is not None and hamming_distance is not None and emut_or_step_delta_e is not None:
                    break

    except IOError as e:
        print(f"❌ Error reading file '{pdb_path}': {e}")

    return iteration, delta_e, hamming_distance, emut_or_step_delta_e



def extract_global_metadata(pdb_path):
    """
    Extract global metadata (Protein Index, Beta, Mutation Method, Embedding Difference Method, Protected Indices)
    from a PDB file.

    Args:
        pdb_path (str): Path to the PDB file.

    Returns:
        dict: A dictionary containing the extracted global metadata.
    """
    metadata = {
        "Protein Index": None,
        "Beta": None,
        "Mutation Method": None,
        "Embedding Difference Method": None,
        "Protected Indices": None,
    }
    
    with open(pdb_path, 'r') as pdb_file:
        for line in pdb_file:
            # Protein Index
            if "Protein Index:" in line:
                match = re.search(r"Protein Index: (\d+)", line)
                if match:
                    metadata["Protein Index"] = int(match.group(1))
            
            # Beta
            elif "Beta:" in line:
                match = re.search(r"Beta: ([\d.]+)", line)
                if match:
                    metadata["Beta"] = float(match.group(1))
            
            # Mutation Method
            elif "Mutation Method:" in line:
                match = re.search(r"Mutation Method: (\w+)", line)
                if match:
                    metadata["Mutation Method"] = match.group(1)
            
            # Embedding Difference Method
            elif "Embedding Difference Method:" in line:
                match = re.search(r"Embedding Difference Method: (\w+)", line)
                if match:
                    metadata["Embedding Difference Method"] = match.group(1)
            
            # Protected Indices
            elif "Protected Indices:" in line:
                match = re.search(r"Protected Indices: \"\[(.*?)\]\"", line)
                if match:
                    metadata["Protected Indices"] = list(map(int, match.group(1).split(", ")))
            
            # Break early if all fields are found
            if all(value is not None for value in metadata.values()):
                break
    
    return metadata

def extract_hamdis_alphafold(pdb_path):
    """
    Extract Hamming Distance from an alphafold PDB file, its in the filename.

    Args:
        pdb_path (str): Path to the PDB file.

    Returns:
        "Hamming Distance" (int): The Hamming Distance value from the file.
    """
    # Get the filename from the path
    filename = os.path.basename(pdb_path)
    
    # Use regex to extract the leading number (Hamming Distance)
    match = re.match(r"^(\d+)_", filename)
    
    if match:
        return int(match.group(1))  # Convert the extracted number to int
    else:
        raise ValueError(f"Could not extract Hamming Distance from filename: {filename}")