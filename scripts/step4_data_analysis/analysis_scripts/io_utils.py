import os

# Function to list datasets (subfolders)
def list_datasets(input_data_dir):
    """List all datasets (subfolders) in the INPUT_DATA directory."""
    return [f for f in os.listdir(input_data_dir) if os.path.isdir(os.path.join(input_data_dir, f))]


# Function to list reference PDB files
def list_references(ref_pdb_dir):
    """List all reference PDB files in the REF_PDBS directory."""
    return [f for f in os.listdir(ref_pdb_dir) if f.endswith('.pdb')]
