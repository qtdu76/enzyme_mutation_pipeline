from Bio.PDB import PDBParser, PPBuilder
from pathlib import Path

def extract_sequence_length(pdb_path, chain_id='A'):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_path)
    model = structure[0]
    chain = model[chain_id]

    ppb = PPBuilder()
    peptides = ppb.build_peptides(chain)
    if not peptides:
        return 0

    sequence = ''.join(str(pp.get_sequence()) for pp in peptides)
    return len(sequence)

def check_sequence_lengths(folder, reference_length, chain_id='A'):
    folder = Path(folder)
    for pdb_file in folder.glob("*.pdb"):
        length = extract_sequence_length(pdb_file, chain_id)
        if length == reference_length:
            print(f"✅ {pdb_file.name} | length: {length}")
        else:
            print(f"❌ {pdb_file.name} | length: {length} (expected {reference_length})")

check_sequence_lengths("/rds/general/user/qt121/home/pipeline/output/6eqe/folding/beta_400.0_trimmed", reference_length=271)
