import os
import argparse
from scripts.logging_config import setup_logger
from scripts.step3_folding.folding_scripts.pipeline import run 


def main(pdb_code, batch_size, beta_values=None):
    pdb_code = pdb_code.lower()
    logger = setup_logger("folding_driver", pdb_code)

    mutation_base_dir = os.path.join("output", pdb_code, "mutation")
    folding_base_dir = os.path.join("output", pdb_code, "folding")

    if not os.path.exists(mutation_base_dir):
        logger.error(f"Mutation folder not found: {mutation_base_dir}")
        return

    if beta_values:
        beta_dirs = [f"beta_{b}" for b in beta_values]
    else:
        beta_dirs = [d for d in os.listdir(mutation_base_dir) if d.startswith("beta_")]


    if not beta_dirs:
        logger.error(f"No beta_* folders found in {mutation_base_dir}")
        return

    for beta_dir in sorted(beta_dirs):
        beta_value = beta_dir.replace("beta_", "")
        mutation_dir = os.path.join(mutation_base_dir, beta_dir)
        folding_output_dir = os.path.join(folding_base_dir, beta_dir)

        try:
            sim_results_csv = [f for f in os.listdir(mutation_dir) if f.endswith("_simRes.csv")][0]
        except IndexError:
            logger.warning(f"No _simRes.csv found in {mutation_dir}. Skipping.")
            continue

        sim_results_path = os.path.join(mutation_dir, sim_results_csv)
        os.makedirs(folding_output_dir, exist_ok=True)

        logger.info(f"🧬 Folding: {pdb_code} | Beta: {beta_value}")
        logger.info(f"Input: {sim_results_path}")
        logger.info(f"Output: {folding_output_dir}")

        run(
            csv_file=sim_results_path,
            output_dir=folding_output_dir,
            batch_size=batch_size,
            pdb_code=pdb_code
        )

    logger.info(f"✅ Folding step complete for all betas in {mutation_base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run folding for all beta folders.")
    parser.add_argument("--pdb", required=True, help="PDB code (e.g. 1bn7)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--betas", nargs="*", help="List of beta values to process (e.g. 20.0 50.0)")
    
    args = parser.parse_args()
    main(pdb_code=args.pdb, batch_size=args.batch_size)
