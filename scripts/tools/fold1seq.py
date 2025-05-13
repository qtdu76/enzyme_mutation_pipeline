import torch
from transformers import EsmForProteinFolding, AutoTokenizer
from torch.utils._pytree import tree_map
import os
import sys
import argparse


def generate_pdb(model, output, idx=0):
    output_cpu = tree_map(lambda x: x.to("cpu"), output)
    pdb_strings = model.output_to_pdb(output_cpu)
    return pdb_strings[idx]


def fold_and_save(sequence, save_path="output.pdb", model_name="facebook/esmfold_v1", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = EsmForProteinFolding.from_pretrained(model_name).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    pdb_str = generate_pdb(model, output)
    with open(save_path, "w") as f:
        f.write(pdb_str)
    print(f"[INFO] Saved PDB to {save_path}")


def find_sequence_by_code(folder, code):
    for filename in os.listdir(folder):
        if filename.endswith(".fasta") or filename.endswith(".fa"):
            with open(os.path.join(folder, filename), "r") as f:
                lines = f.readlines()
                for i in range(len(lines)):
                    if lines[i].startswith(">") and code.lower() in lines[i].lower():
                        # Join all lines after header until next header or end
                        seq_lines = []
                        for j in range(i + 1, len(lines)):
                            if lines[j].startswith(">"):
                                break
                            seq_lines.append(lines[j].strip())
                        return "".join(seq_lines)
    raise ValueError(f"Code '{code}' not found in any FASTA file in {folder}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fold a protein sequence using ESMFold")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--code", help="4-letter code to look up in FASTA files")
    group.add_argument("--sequence", help="Direct amino acid sequence")

    parser.add_argument("--folder", help="Path to folder containing FASTA files (used with --code)")
    parser.add_argument("--name", help="Name for output file if using --sequence", default="output")

    args = parser.parse_args()

    if args.code:
        if not args.folder:
            raise ValueError("You must specify --folder when using --code")
        sequence = find_sequence_by_code(args.folder, args.code)
        fold_and_save(sequence, f"{args.code.lower()}_esm.pdb")
    else:
        fold_and_save(args.sequence, f"{args.name}_esm.pdb")

