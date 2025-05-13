import os
import torch
import pandas as pd
import time
from transformers import pipeline as hf_pipeline
from step1_preprocessing.preprocessing import run_preprocessing
from step2_mutation.mutation_scripts.simulation import monte_carlo_simulation
from step3_folding.folding_scripts.pipeline import run as run_folding
from step4_data_analysis.analysis_scripts.analyze import analyze_datasets
from logging_config import setup_logger 


def run_pipeline(pdb_code, beta_values, model_path, num_iterations, device_choice="gpu", batch_size=128):
    """
    Master pipeline script to run preprocessing, mutation, and folding steps sequentially.

    Args:
        pdb_code (str): PDB code of the enzyme to process (e.g., '1bn7').
        beta_values (list): List of beta values for the mutation simulations.
        model_path (str): Path to the ESM model.
        num_iterations (int): Number of Monte Carlo iterations for mutation.
        device_choice (str): Device to run the model on ("gpu" or "cpu").
        batch_size (int): Batch size for folding.
    """

    # === Initialize logger with pdb_code ===
    logger = setup_logger('pipeline_master', pdb_code)
    logger.info(f"🚀 Starting full pipeline for {pdb_code}...")

    start_time = time.time()

    # === STEP 1: Preprocessing ===
    logger.info("🔬 Starting Preprocessing Step...")
    info_csv_path = run_preprocessing(pdb_code)
    logger.info(f"✅ Preprocessing complete. Info saved to {info_csv_path}")

    # === STEP 2: Mutation ===
    logger.info("🔄 Starting Mutation Step...")

    # Load sequence and protected indices from info.csv
    info_df = pd.read_csv(info_csv_path)
    sequence = info_df.at[0, 'Sequence']
    binding_residues_fasta = info_df.at[0, 'Binding Residues (FASTA)']
    protected_indices = [int(idx) for idx in binding_residues_fasta.split(",") if idx.strip().isdigit()] or None

    # Set device
    if device_choice == "gpu":
        device = 0 if torch.cuda.is_available() else -1
        if device == -1:
            logger.warning("⚠️ GPU requested but not available. Falling back to CPU.")
    elif device_choice == "cpu":
        device = -1
    else:
        raise ValueError('Invalid device choice. Use "gpu" or "cpu".')

    # Load ESM2 model pipeline
    esm2_pipeline = hf_pipeline("feature-extraction", model=model_path, device=device)
    logger.info(f"🖥️ Using device: {'GPU' if device != -1 else 'CPU'}")

    # Run simulations for each beta value
    for beta in beta_values:
        logger.info(f"Running simulation with beta: {beta}")
        monte_carlo_simulation(
            beta=beta,
            esm2_pipeline=esm2_pipeline,
            num_iterations=num_iterations,
            sequence=sequence,
            pdb_code=pdb_code,
            protected_indices=protected_indices  # ✅ 1-based indices are correctly passed
        )
    logger.info(f"✅ Mutation simulations complete for betas: {beta_values}")

    logger.info("🧬 Starting Folding Step...")

    # === STEP 3: Folding ===
    # For each beta, fold the mutated sequences
    for beta in beta_values:
        beta_dir = f"beta_{beta}"
        mutation_results_dir = os.path.join("output", pdb_code, "mutation", beta_dir)
        sim_results_csv = [f for f in os.listdir(mutation_results_dir) if f.endswith("_simRes.csv")][0]
        sim_results_path = os.path.join(mutation_results_dir, sim_results_csv)

        # Define output directory for folding
        folding_output_dir = os.path.join("output", pdb_code, "folding", beta_dir)
        os.makedirs(folding_output_dir, exist_ok=True)

        # Run folding for the mutated sequences
        run_folding(
            csv_file=sim_results_path,
            output_dir=folding_output_dir,
            batch_size=batch_size,
            pdb_code=pdb_code
        )
        
        logger.info(f"✅ Folding complete for beta {beta}")

    logger.info(f"✅ Folding step complete for all betas: {beta_values}")

    logger.info("📊 Starting Data Analysis Step...")


    # === STEP 4: Analysis ===
    for beta in beta_values:
        beta_dir = f"beta_{beta}"
        
        # Define dataset and reference paths
        dataset_dir = os.path.join("output", pdb_code, "folding", beta_dir)
        reference_pdb = os.path.join("INPUTS", "esm_ref_pdbs", f"{pdb_code.lower()}.pdb")

        # Define output directory for analysis
        analysis_output_dir = os.path.join("output", pdb_code, "analysis", beta_dir)
        os.makedirs(analysis_output_dir, exist_ok=True)

        # Run data analysis
        analyze_datasets(
            dataset=dataset_dir,
            reference=reference_pdb,
            output_dir=analysis_output_dir, 
            pdb_code=pdb_code,
            info_csv_path=info_csv_path
        )
        logger.info(f"✅ Analysis complete for beta {beta}")

    logger.info("🎉 Pipeline complete! All preprocessing, mutation, folding, and analysis steps are done.")

    end_time = time.time()
    elapsed_walltime = end_time - start_time


    # === Log Summary ===
    logger.info("\n" + "="*35)
    logger.info(f"Walltime usage: {time.strftime('%H:%M:%S', time.gmtime(elapsed_walltime))}")
    logger.info("="*35 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the full protein mutation, folding, and analysis pipeline.")
    parser.add_argument("--pdb_code", type=str, required=True, help="PDB code of the enzyme (e.g., '1bn7').")
    parser.add_argument("--beta_values", type=str, required=True, help="Comma-separated list of beta values (e.g., '50.0,100.0,250.0').")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the ESM model.")
    parser.add_argument("--num_iterations", type=int, default=10000, help="Number of Monte Carlo iterations for mutation.")
    parser.add_argument("--device_choice", type=str, choices=["gpu", "cpu"], default="gpu", help="Device to run the model on ('gpu' or 'cpu').")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for folding.")

    args = parser.parse_args()
    beta_values = [float(beta) for beta in args.beta_values.split(",")]

    run_pipeline(
        pdb_code=args.pdb_code,
        beta_values=beta_values,
        model_path=args.model_path,
        num_iterations=args.num_iterations,
        device_choice=args.device_choice,
        batch_size=args.batch_size
    )