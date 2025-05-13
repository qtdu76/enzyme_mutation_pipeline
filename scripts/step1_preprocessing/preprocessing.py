import os
import logging
import pandas as pd
import pymol
from pymol import cmd
from Bio import PDB, SeqIO, pairwise2
from Bio.Seq import Seq
from Bio.Data import PDBData
from logging_config import setup_logger


def run_preprocessing(pdb_code):
    """
    Preprocesses the given PDB code by extracting sequence, binding sites, and alignment information.
    
    Args:
        pdb_code (str): The PDB code of the enzyme (e.g., '1bn7').
    
    Returns:
        str: Path to the generated info.csv file.
    """

    logger = setup_logger('preprocessing', pdb_code)

    pdb_code_lower = pdb_code.lower()
    pdb_code_upper = pdb_code.upper()

    # === SETUP PATHS ===
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # INPUT FILES
    fasta_path = os.path.join(script_directory, f"../../INPUTS/REF_sequences/rcsb_pdb_{pdb_code_upper}.fasta")
    pdb_path = os.path.join(script_directory, f"../../INPUTS/REF_PDBS/{pdb_code_lower}.pdb")


    # OUTPUT DIRECTORY
    output_folder = os.path.join(script_directory, f"../../output/{pdb_code_lower}/preprocessing")
    os.makedirs(output_folder, exist_ok=True)


    # OUTPUT FILES
    info_csv_path = os.path.join(output_folder, "info.csv")

    logger.info(f"Starting preprocessing for {pdb_code_upper}...")

    # === STEP 1: Extract Sequence & General Info ===
    try:
        fasta_record = SeqIO.read(fasta_path, "fasta")
        sequence = str(fasta_record.seq)
        protein_length = len(sequence)
        description = fasta_record.description

        logger.info(f"Extracted sequence from rcsb_pdb_{pdb_code_upper}.fasta - Length: {protein_length}")

    except Exception as e:
        logger.error(f"Error processing FASTA file: {e}")
        raise

    # === STEP 2: Identify Binding Sites ===
    try:
        cmd.reinitialize()
        cmd.load(pdb_path, "protein")

        cmd.select("chainA", "protein and chain A")
        cmd.select("ligands", "protein and hetatm and not resn HOH")

        ligand_atoms = cmd.get_model("ligands").atom
        ligands_found = sorted(set(atom.resn for atom in ligand_atoms))

        # Filter out irrelevant ligands
        irrelevant_ligands = {"HOH", "SO4", "ACT", "GOL", "PEG", "EDO", "NA", "CA"} #this should be tailered to a specific enzyme, check which ligands are irrelevant
        ligands_found = [lig for lig in ligands_found if lig not in irrelevant_ligands]

        binding_residues_pdb = set()

        for ligand in ligands_found:
            cmd.select(f"ligand_{ligand}", f"protein and hetatm and resn {ligand}")
            cmd.select(f"binding_site_{ligand}", f"chainA within 5.0 of ligand_{ligand} and not resn HOH")

            binding_residues = cmd.get_model(f"binding_site_{ligand}").atom
            residue_set = sorted(set((int(atom.resi), atom.resn) for atom in binding_residues if atom.resn != "HOH"))

            if residue_set:
                binding_residues_pdb.update(resi for resi, _ in residue_set)

        logger.info(f"Ligands found in {pdb_code_upper}: {', '.join(ligands_found) if ligands_found else 'None'}")
        logger.info(f"Binding site residues (PDB numbering): {sorted(binding_residues_pdb)}")

    except Exception as e:
        logger.error(f"Error processing PDB file: {e}")
        raise

    # === STEP 3: Align Binding Sites to FASTA Indices ===
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)

        pdb_sequence = []
        residue_numbers = []

        for model in structure:
            for chain in model:
                if chain.id == "A":
                    for residue in chain:
                        if PDB.is_aa(residue):
                            residue_numbers.append(residue.get_id()[1])
                            try:
                                pdb_sequence.append(PDBData.protein_letters_3to1[residue.get_resname()])
                            except KeyError:
                                pdb_sequence.append("X")  # Unrecognized residues marked as 'X'

        pdb_sequence = Seq("".join(pdb_sequence))

        alignments = pairwise2.align.globalxx(sequence, pdb_sequence)
        best_alignment = alignments[0] if alignments else None

        if best_alignment:
            aligned_fasta, aligned_pdb, _, _, _ = best_alignment

            # === LOG THE ALIGNMENT ===
            logger.info("\nAlignment between FASTA and PDB sequences:\n")
            logger.info(f"FASTA: {aligned_fasta}\n")
            logger.info(f"PDB  : {aligned_pdb}\n")

            pdb_to_fasta_map = {}
            fasta_index = 0
            pdb_index = 0

            for fasta_char, pdb_char in zip(aligned_fasta, aligned_pdb):
                if fasta_char != "-":
                    fasta_index += 1
                if pdb_char != "-":
                    if pdb_index < len(residue_numbers):
                        pdb_to_fasta_map[residue_numbers[pdb_index]] = fasta_index
                    pdb_index += 1

            binding_residues_fasta = [pdb_to_fasta_map[resi] for resi in binding_residues_pdb if resi in pdb_to_fasta_map]

            logger.info(f"Binding site residues (FASTA numbering): {sorted(binding_residues_fasta)}")
        else:
            logger.warning("No alignment found between the PDB and FASTA sequences.")

    except Exception as e:
        logger.error(f"Error aligning PDB to FASTA: {e}")
        raise


    # === STEP 4: Save Output to CSV ===
    try:
        info_df = pd.DataFrame({
            "PDB Code": [pdb_code_upper],
            "Description": [description],
            "Sequence Length": [protein_length],
            "Sequence": [sequence],  # Added the full sequence here
            "Ligands Found": [", ".join(ligands_found) if ligands_found else "None"],
            "Binding Residues (PDB)": [", ".join(map(str, sorted(binding_residues_pdb)))],
            "Binding Residues (FASTA)": [", ".join(map(str, sorted(binding_residues_fasta)))]
        })

        info_df.to_csv(info_csv_path, index=False)
        logger.info(f"Preprocessing complete. Info saved to {info_csv_path}")

    except Exception as e:
        logger.error(f"Error saving CSV: {e}")
        raise

    return info_csv_path  # Return the path to the generated info.csv
