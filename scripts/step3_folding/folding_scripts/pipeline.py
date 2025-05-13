import os
import torch
import numpy as np
import logging
import pandas as pd
from torch.utils.data import DataLoader
from .data_management import SequenceDataset, custom_collate_fn
from .model_wrapper import ESMModelWrapper
from .io_utils import load_csv_data, extract_global_metadata, generate_pdb, save_pdb_with_metadata
from scripts.logging_config import setup_logger


def log_gpu_memory_usage():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024 ** 3)
        used_percentage = (allocated_memory / total_memory) * 100

        logging.info(
            f"GPU Memory Usage: Allocated: {allocated_memory:.2f} GB, Reserved: {reserved_memory:.2f} GB, "
            f"Total: {total_memory:.2f} GB, Used: {used_percentage:.2f}%"
        )
    else:
        logging.info("No GPU detected. Running on CPU.")

def run(csv_file, output_dir, batch_size, pdb_code):
    """
    Orchestrates the entire folding pipeline: loading data, running the model, and saving results.

    Args:
        csv_file (str): Path to the input CSV file from mutation results.
        output_dir (str): Directory to save the output PDB files.
        batch_size (int): Batch size for processing sequences.
    """

    logger = setup_logger('folding', pdb_code)

    logger.info("Starting the folding pipeline...")

    # === EXTRACT GLOBAL METADATA ===
    global_metadata = extract_global_metadata(csv_file)
    pdb_code = global_metadata.get("PDB Code", "unknown")
    beta_value = global_metadata.get("Beta", "unknown")
    logger.info(f"Extracted global metadata: {global_metadata}")

    # === LOAD DATA ===
    df = load_csv_data(csv_file)
    logger.info(f"Loaded {len(df)} sequences from the mutation CSV.")

    os.makedirs(output_dir, exist_ok=True)

    log_gpu_memory_usage()

    # === PREPARE DATASET AND LOADER ===
    dataset = SequenceDataset(df)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=4)
    logger.info(f"Data loader prepared with batch size: {batch_size}")

    # === INITIALIZE MODEL ===
    model_wrapper = ESMModelWrapper()

    # === PROCESS BATCHES ===
    for i, (batch_rows, batch_indices) in enumerate(data_loader):
        logger.info(f"Processing batch {i + 1}/{len(data_loader)}...")
        sequences = [row['Sequence'] for row in batch_rows]

        assert len(batch_indices) == len(sequences), "Mismatch between batch_indices and sequences!"

        log_gpu_memory_usage()

        # Generate positions for the current batch
        outputs = model_wrapper.generate_positions(sequences)
        logger.info(f"Generated model outputs for batch {i + 1}.")

        # === PROCESS EACH SEQUENCE IN THE BATCH ===
        for idx, _ in enumerate(sequences):
            # Define a clear and descriptive PDB file name
            pdb_filename = os.path.join(output_dir, f"msk_{pdb_code}_beta_{beta_value}_seq_{batch_indices[idx]}.pdb")

            # Generate PDB string
            pdb_string = generate_pdb(model_wrapper.model, outputs, idx)

            # Save the PDB file with metadata
            output_structure = {"pdb": pdb_string}
            save_pdb_with_metadata(output_structure, batch_rows[idx], global_metadata, pdb_filename)
            logger.info(f"Saved PDB file: {pdb_filename}")


    logger.info("Folding pipeline complete.")
