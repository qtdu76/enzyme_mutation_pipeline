import os
import sys
import torch
import argparse
import pandas as pd

from transformers import pipeline as hf_pipeline
from logging_config import setup_logger
from scripts.step2_mutation.mutation_scripts.simulation import monte_carlo_simulation  

# Constants
MODEL_PATH = "/rds/general/user/qt121/home/pipeline/scripts/step2_mutation/esm2_models/esm2_650m"

def main(pdb_code, device_choice, beta_values, num_iterations):
    pdb_code = pdb_code.lower()
    logger = setup_logger('mutation_driver', pdb_code)

    # === Load info.csv ===
    info_csv_path = os.path.join("output", pdb_code, "preprocessing", "info.csv")
    if not os.path.exists(info_csv_path):
        raise FileNotFoundError(f"info.csv not found at: {info_csv_path}")

    info_df = pd.read_csv(info_csv_path)
    sequence = info_df.at[0, 'Sequence']
    binding_residues_fasta = info_df.at[0, 'Binding Residues (FASTA)']
    protected_indices = [int(idx) for idx in binding_residues_fasta.split(",") if idx.strip().isdigit()] or None

    # === Set device ===
    if device_choice == "gpu":
        device = 0 if torch.cuda.is_available() else -1
        if device == -1:
            logger.warning("⚠️ GPU requested but not available. Falling back to CPU.")
    elif device_choice == "cpu":
        device = -1
    else:
        raise ValueError('Invalid device choice. Use "gpu" or "cpu".')

    logger.info(f"🧬 PDB: {pdb_code} | Sequence length: {len(sequence)} | Protected: {protected_indices}")
    logger.info(f"🖥️ Using device: {'GPU' if device != -1 else 'CPU'}")

    # === Load ESM2 embedding pipeline ===
    esm2_pipeline = hf_pipeline("feature-extraction", model=MODEL_PATH, device=device)

    # === Run Monte Carlo simulation for each beta ===
    for beta in beta_values:
        logger.info(f"🔁 Running simulation with beta = {beta}")
        monte_carlo_simulation(
            beta=beta,
            esm2_pipeline=esm2_pipeline,
            num_iterations=num_iterations,
            sequence=sequence,
            pdb_code=pdb_code,
            protected_indices=protected_indices
        )

    logger.info(f"✅ Mutation complete for {pdb_code} (betas: {beta_values})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Monte Carlo mutation step.")
    parser.add_argument("--pdb", required=True, help="PDB code (e.g. 1bn7)")
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu", help="Computation device")
    parser.add_argument("--betas", nargs="+", type=float, default=[10.0], help="List of beta values")
    parser.add_argument("--iterations", type=int, default=100, help="Number of Monte Carlo iterations")

    args = parser.parse_args()
    main(
        pdb_code=args.pdb,
        device_choice=args.device,
        beta_values=args.betas,
        num_iterations=args.iterations
    )
