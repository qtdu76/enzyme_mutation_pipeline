import os
from pathlib import Path
from Bio.PDB import PDBParser, PDBIO

def extract_header_lines(pdb_path):
    """Extract all lines before the first ATOM line (i.e., metadata)"""
    header_lines = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                break
            header_lines.append(line)
    return header_lines

def trim_pdb_file(input_path, output_path, n_remove=28, c_remove=5):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", input_path)
    model = structure[0]
    chain = model['A']

    residues = list(chain.get_residues())
    trimmed = residues[n_remove:len(residues) - c_remove]

    for res in residues:
        if res not in trimmed:
            chain.detach_child(res.id)

    # Save trimmed ATOM lines to a temp string
    io = PDBIO()
    io.set_structure(structure)
    from io import StringIO
    atom_block = StringIO()
    io.save(atom_block)
    atom_block.seek(0)

    # Combine preserved metadata with new atom block
    header = extract_header_lines(input_path)
    with open(output_path, 'w') as out_f:
        out_f.writelines(header)
        out_f.writelines(atom_block.readlines())

def batch_trim_pdbs(input_folder, output_folder, n_remove=28, c_remove=5):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    pdb_files = list(input_folder.glob("*.pdb"))
    for pdb_file in pdb_files:
        output_path = output_folder / pdb_file.name
        trim_pdb_file(pdb_file, output_path, n_remove, c_remove)
        print(f"Trimmed {pdb_file.name} → {output_path.name}")

# # Run trimming directly when script is executed
batch_trim_pdbs(
    "/rds/general/user/qt121/home/pipeline/output/6eqe/folding/beta_20.0",
    "/rds/general/user/qt121/home/pipeline/output/6eqe/folding/beta_20.1",
    n_remove=28,
    c_remove=5
)

# Run trimming on a single file
# trim_pdb_file(
#     input_path="/rds/general/user/qt121/home/pipeline/INPUTS/esm_ref_pdbs/6eqe_fasta.pdb",
#     output_path="/rds/general/user/qt121/home/pipeline/INPUTS/esm_ref_pdbs/6eqe.pdb",
#     n_remove=28,
#     c_remove=5
# )
