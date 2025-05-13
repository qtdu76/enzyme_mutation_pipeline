from .sequence_utils import hamming_distance, generate_sequence_pairs
from .embedding_utils import get_embeddings, calc_embedding_diff2ref, SequenceEmbeddingPair
from .mutation_utils import mutate_sequence_pair
import math
import random
import csv
import inspect
import os
import logging
from logging_config import setup_logger

logger = setup_logger('preprocessing', 'pipeline_execution.log')

def calculate_acceptance_probability(prev_pair, mut_pair, ref_pair, protected_indices_0, beta):
    """
    Calculates the acceptance probability for the Monte Carlo simulation.

    Parameters: 
    - prev_pair: SequenceEmbeddingPair of the original sequence.
    - mut_pair: SequenceEmbeddingPair of the mutated sequence.
    - ref_pair: SequenceEmbeddingPair of the reference/original sequence.
    - protected_indices_0 (list[int]): 0-based indices of protected residues.
    - beta: Beta parameter for the probability calculation (default is 0.5).

    Returns:
    - A tuple containing:
        1. Acceptance probability
        2. deltaE (difference in similarity score)
        3. Emut (similarity score of mutated sequence)
    """

    # ✅ Compute ΔE using the standard embedding difference function
    dE, Emut = calc_embedding_diff2ref(prev_pair.embedding, mut_pair.embedding, ref_pair.embedding, protected_indices_0)
    print("delta_E is:", dE)

    # ✅ Step 2: Calculate the acceptance probability using the Monte Carlo formula
    probability = min(1, math.exp(-beta * dE))

    return probability, dE, Emut



import random

def monte_carlo_step(prev_pair, mut_pair, ref_pair, esm2_pipeline, beta, protected_indices_0):
    """
    Perform a single Monte Carlo step in the mutation simulation.

    Parameters:
    - prev_pair: SequenceEmbeddingPair of the original sequence.
    - mut_pair: SequenceEmbeddingPair of the mutated sequence.
    - ref_pair: SequenceEmbeddingPair of the reference/original sequence.
    - esm2_pipeline: Model or pipeline used to generate embeddings.
    - beta: Beta parameter for the probability calculation (default is 0.5).
    - protected_indices_0: List of 0-based indices that must not be mutated.

    Returns:
    - prev_pair: Updated SequenceEmbeddingPair after mutation.
    - new_mut_pair: SequenceEmbeddingPair of the new mutated sequence.
    - delta_E: Energy difference between prev and mut sequence.
    - Emut: Similarity score of the mutated sequence.
    """

    # ✅ Step 1: Calculate the acceptance probability
    probability, delta_E, Emut = calculate_acceptance_probability(
        prev_pair=prev_pair,
        mut_pair=mut_pair,
        ref_pair=ref_pair,
        protected_indices_0=protected_indices_0,
        beta=beta
    )

    # ✅ Debug: Print probability and random value
    rand_value = random.uniform(0, 1)
    print(f"Calculated Probability: {probability}, Random Value: {rand_value}")
    
    # ✅ Step 2: Determine if the mutation is accepted
    accepted = (rand_value < probability)
    print(f"Acceptance Decision: {'Accepted' if accepted else 'Rejected'}")

    if accepted:
        # If accepted, update prev_pair to mut_pair
        prev_pair = mut_pair

    # ✅ Step 3: Generate a new mutated pair based on the updated prev_pair
    new_mut_pair, _ = mutate_sequence_pair(
        prev_pair=prev_pair,
        esm2_pipeline=esm2_pipeline,
        protected_indices_0=protected_indices_0
    )

    return prev_pair, new_mut_pair, delta_E, Emut

def monte_carlo_simulation(beta, esm2_pipeline, num_iterations, sequence, pdb_code, protected_indices):
    """
    Run a Monte Carlo simulation for protein sequence mutation.

    Parameters:
    - beta: Beta parameter for the probability calculation.
    - esm2_pipeline: Model or pipeline used to generate embeddings.
    - num_iterations: Number of Monte Carlo iterations.
    - sequence: The initial protein sequence.
    - pdb_code: The PDB code for the protein.
    - protected_indices: List of 1-based indices that must not be mutated. the input is a fasta sequence, so these should be fasta indices.

    Outputs:
    - Stores results in a CSV file.
    """

    logger = setup_logger('mutation', pdb_code)

    logger.info(f"Beta: {beta}")
    logger.info("Starting the mutation simulation...")

    # ✅ Convert 1-based protected indices to 0-based for internal use
    protected_indices = protected_indices if protected_indices else []
    protected_indices_0 = [idx - 1 for idx in protected_indices]

    # 🔍 Debugging print
    print(f"Protected Indices (1-based): {protected_indices}")
    print(f"Protected Indices (0-based): {protected_indices_0}")

    record_interval = 10  # Record data every 'n' iterations

    # Step 1: Initialize the reference sequence and embedding
    og_sequence = sequence
    logger.info(f"Reference sequence: {og_sequence}")

    og_embedding = get_embeddings(og_sequence, esm2_pipeline)
    ref_pair = SequenceEmbeddingPair(og_sequence, og_embedding, protected_indices_0)

    # Step 2: Generate initial sequence pairs (original and first mutation)
    prev_pair, mut_pair = generate_sequence_pairs(
        sequence=og_sequence,
        esm2_pipeline=esm2_pipeline,
        protected_indices_0=protected_indices_0
    )

    # ✅ ASSERTION: Check if protected indices are conserved in the first mutation
    assert all(mut_pair.sequence[idx] == og_sequence[idx] for idx in protected_indices_0), (
        f"❌ ERROR: Protected residues were altered in first mutation!\n"
        f"Original:  {og_sequence}\n"
        f"Mutated:   {mut_pair.sequence}\n"
        f"Protected Indices (1-based): {protected_indices}"
    )

    # Step 3: Define output directory
    output_dir = os.path.join("output", pdb_code.lower(), "mutation", f"beta_{beta}")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, f"msk_{pdb_code.lower()}_beta_{beta}_simRes.csv")

    logger.info(f"Output file: {output_file}")

    # Write header to the file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Simulation Parameters"])
        writer.writerow(["PDB Code", pdb_code])
        writer.writerow(["Protein Sequence", og_sequence])
        writer.writerow(["Beta", beta])
        writer.writerow(["Protected Indices", protected_indices])
        writer.writerow([])
        writer.writerow(["Iteration", "Delta E", "Emut", "Hamming Distance", "Sequence"])

    # ✅ Monte Carlo simulation loop
    for i in range(num_iterations):
        print(f"\n{'-' * 10} Step {i + 1} {'-' * 10}")
        logger.info(f"\n{'-' * 10} Step {i + 1} {'-' * 10}")

        prev_pair, mut_pair, dE, Emut = monte_carlo_step(
            prev_pair=prev_pair,
            mut_pair=mut_pair,
            ref_pair=ref_pair,
            esm2_pipeline=esm2_pipeline,
            beta=beta,
            protected_indices_0=protected_indices_0
        )

        # ✅ ASSERTION: Check if protected indices are conserved after mutation
        assert all(prev_pair.sequence[idx] == og_sequence[idx] for idx in protected_indices_0), (
            f"❌ ERROR: Protected residues were altered at step {i + 1}!\n"
            f"Original:  {og_sequence}\n"
            f"Mutated:   {prev_pair.sequence}\n"
            f"Protected Indices (1-based): {protected_indices}"
        )

        logger.info(f"dE = {dE}")
        logger.info(f"Emut = {Emut}")

        # ✅ Record data every `record_interval` steps
        if (i + 1) % record_interval == 0:
            # Calculate Hamming distance
            ham_diff = hamming_distance(og_sequence, prev_pair.sequence)

            # Write to the CSV file
            with open(output_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i + 1, dE, Emut, ham_diff, prev_pair.sequence])

    print(f"\n{'-' * 10} Simulation Complete {'-' * 10}")
    print("Original sequence:", og_sequence)
    print("Final accepted sequence:", prev_pair.sequence)
    ham_diff = hamming_distance(og_sequence, mut_pair.sequence)
    print("Hamming difference is:", ham_diff)

    logger.info(f"\n{'-' * 10} Simulation Complete {'-' * 10}")
    logger.info(f"Original sequence: {og_sequence}")
    logger.info(f"Final accepted sequence: {prev_pair.sequence}")
    logger.info(f"Hamming difference is: {ham_diff}")
