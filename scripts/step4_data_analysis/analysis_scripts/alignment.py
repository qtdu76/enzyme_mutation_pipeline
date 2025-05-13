from Bio import SeqIO, PDB, pairwise2
from Bio.Seq import Seq
from Bio.Data import IUPACData
import os

def extract_pdb_sequence(pdb_file, chain_id="A"):
    """Extracts residue IDs and one-letter amino acids from a PDB chain."""
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    res_ids = []
    residues = []

    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for res in chain:
                    if PDB.is_aa(res):
                        res_id = res.get_id()[1]
                        try:
                            aa = IUPACData.protein_letters_3to1[res.get_resname()]
                        except KeyError:
                            aa = "X"
                        res_ids.append(res_id)
                        residues.append(aa)

    return res_ids, Seq("".join(residues))


def align_sequences(fasta_seq, pdb_seq, pdb_res_ids):
    """Aligns two sequences and returns a mapping: esm_index → (pdb_res_id, aa)."""
    alignment = pairwise2.align.globalxx(fasta_seq, pdb_seq, one_alignment_only=True)[0]
    aligned_fasta, aligned_pdb = alignment.seqA, alignment.seqB

    esm_to_pdb = {}
    fasta_index = 0
    pdb_index = 0

    for f_char, p_char in zip(aligned_fasta, aligned_pdb):
        if f_char != "-":
            fasta_index += 1
        if p_char != "-":
            if f_char == p_char and fasta_index > 0:
                esm_to_pdb[fasta_index] = (pdb_res_ids[pdb_index], f_char)
            pdb_index += 1

    return esm_to_pdb


def run_alignment(fasta_path, pdb_path, chain_id="A"):
    """Main function to align FASTA and PDB sequences and return the index mapping."""
    fasta_record = SeqIO.read(fasta_path, "fasta")
    fasta_seq = fasta_record.seq

    pdb_res_ids, pdb_seq = extract_pdb_sequence(pdb_path, chain_id=chain_id)

    esm_to_pdb = align_sequences(fasta_seq, pdb_seq, pdb_res_ids)

    return esm_to_pdb

