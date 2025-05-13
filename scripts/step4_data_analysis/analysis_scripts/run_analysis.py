import os
import argparse
from .analyze import analyze_datasets
from scripts.logging_config import setup_logger

def main(pdb_code, beta, ref_type):
    pdb_code = pdb_code.lower()

    # Set paths
    base_dir = os.path.join("output", pdb_code)
    dataset_dir = os.path.join(base_dir, "folding", f"beta_{beta}")
    ref_folder = "esm_ref_pdbs" if ref_type == "esm" else "REF_PDBS"
    reference_file = os.path.join("INPUTS", ref_folder, f"{pdb_code}.pdb")
    output_dir = os.path.join(base_dir, "analysis", f"beta_{beta}")
    info_csv_path = os.path.join(base_dir, "preprocessing", "info.csv")

    logger = setup_logger("analysis_driver", pdb_code)
    logger.info(f"🔍 Starting analysis for {pdb_code} at beta={beta}")
    logger.info(f"Dataset: {dataset_dir}")
    logger.info(f"Reference: {reference_file}")

    if not os.path.exists(reference_file):
        logger.error(f"Reference file not found: {reference_file}")
        return
    if not os.path.exists(dataset_dir):
        logger.error(f"Dataset directory not found: {dataset_dir}")
        return

    # Run analysis
    analyze_datasets(
        dataset=dataset_dir,
        reference=reference_file,
        output_dir=output_dir,
        pdb_code=pdb_code,
        info_csv_path=info_csv_path
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the analysis step of the pipeline.")
    parser.add_argument("--pdb", required=True, help="PDB code (e.g. 1bn7)")
    parser.add_argument("--beta", type=float, required=True, help="Beta value (e.g. 10.0)")
    parser.add_argument("--ref_type", choices=["pdb", "esm"], default="pdb", help="Reference structure type (pdb or esm)")

    args = parser.parse_args()
    main(pdb_code=args.pdb, beta=args.beta, ref_type=args.ref_type)
