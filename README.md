# Enzyme Mutation Pipeline (ESM-2 + Monte Carlo + ESMFold)

This repository implements a computational pipeline for generating enzyme variants while preserving the structural environment of key residues (for example catalytic-site residues).

Starting from a reference sequence and structure, the pipeline:

1. identifies protected residues from the reference structure,
2. proposes random point mutations outside those protected positions,
3. scores mutations using ESM-2 embeddings and a Metropolis acceptance rule,
4. folds accepted sequences with ESMFold,
5. evaluates structural preservation with RMSD and lDDT-based metrics.

The workflow is designed to explore distant regions of sequence space while constraining changes around functionally important sites.

## Pipeline Overview

### Step 1: Preprocessing

- Input: reference FASTA + reference PDB.
- Uses PyMOL to detect ligands and residues within 5 Å of ligand atoms (chain A).
- Aligns PDB sequence to FASTA sequence and maps protected residues from PDB numbering to FASTA numbering.
- Writes `output/<pdb_code>/preprocessing/info.csv`.

### Step 2: Mutation (Monte Carlo)

- Mutates one random non-protected position per proposal.
- Computes ESM-2 embeddings for the current and proposed sequences.
- Energy is based on protected-residue embedding similarity to the reference:
  - `E = 1 - mean(cosine_similarity(protected_residue_embeddings, reference_embeddings))`
  - `ΔE = E_mut - E_prev`
- Accepts/rejects with Metropolis criterion:
  - `p_accept = min(1, exp(-beta * ΔE))`
- Writes sampled trajectory rows (every 10 iterations) to:
  - `output/<pdb_code>/mutation/beta_<beta>/msk_<pdb>_beta_<beta>_simRes.csv`

### Step 3: Folding

- Reads mutation CSVs and folds sampled sequences with `facebook/esmfold_v1`.
- Saves one PDB per sampled sequence, with metadata headers (beta, ΔE, Hamming distance, sequence, etc.).
- Output:
  - `output/<pdb_code>/folding/beta_<beta>/msk_<pdb>_beta_<beta>_seq_<idx>.pdb`

### Step 4: Structural Analysis

For each folded structure vs a reference structure, computes:

- PyMOL RMSD (global + protected)
- custom RMSD (`rmsd_stef`, global + protected)
- lDDT (global + protected)
- TM-score columns are included in output schema

Outputs:

- `output/<pdb_code>/analysis/beta_<beta>/esmRef_<pdb>_b_<beta>.csv`

## Repository Layout

```text
INPUTS/
  REF_sequences/     # reference FASTA files (rcsb_pdb_<PDB>.fasta)
  REF_PDBS/          # reference experimental structures
  esm_ref_pdbs/      # ESM-folded reference structures used in analysis

scripts/
  script_daddy.py                     # full pipeline driver
  step1_preprocessing/
  step2_mutation/
  step3_folding/
  step4_data_analysis/
  tools/                              # one-off utility scripts

queueing_scripts/                     # PBS scripts used on HPC
output/                               # pipeline outputs (includes example for 1ua7)
```

## Environment Setup

```bash
conda env create -f environment.yml --name enzyme-mutation
conda activate enzyme-mutation
```

If `conda` complains about the exported `prefix:` in `environment.yml`, remove the last `prefix:` line and retry.

## Inputs Required for a New Target

For `pdb_code=xxxx`, preprocessing expects:

- FASTA: `INPUTS/REF_sequences/rcsb_pdb_XXXX.fasta` (uppercase code in filename)
- PDB: `INPUTS/REF_PDBS/xxxx.pdb` (lowercase code in filename)

The current preprocessing/analysis code assumes chain `A`.

## Run the Full Pipeline

From repository root:

```bash
python scripts/script_daddy.py \
  --pdb_code 1ua7 \
  --beta_values "1.0,100.0,500.0" \
  --model_path facebook/esm2_t33_650M_UR50D \
  --num_iterations 10000 \
  --device_choice gpu \
  --batch_size 1
```

Notes:

- `--model_path` accepts a Hugging Face model ID or a local model directory.
- If GPU is unavailable, the code falls back to CPU.

## Run Steps Individually

### 1) Preprocessing

```bash
python scripts/step1_preprocessing/run_preprocessing.py 1ua7
```

### 2) Mutation

`run_mutation.py` uses a hardcoded `MODEL_PATH` constant in the script. Update it first if needed.

```bash
PYTHONPATH="$(pwd)/scripts:${PYTHONPATH}" \
python scripts/step2_mutation/mutation_scripts/run_mutation.py \
  --pdb 1ua7 --device gpu --betas 100.0 500.0 --iterations 10000
```

### 3) Folding

```bash
python -m scripts.step3_folding.folding_scripts.run_folding \
  --pdb 1ua7 --batch_size 1
```

### 4) Analysis

```bash
python -m scripts.step4_data_analysis.analysis_scripts.run_analysis \
  --pdb 1ua7 --beta 500.0 --ref_type esm
```

## Logs

Logs are written to:

- `pipeline_logs/pipeline_<pdb_code>.log`

## Included Example Outputs

The repository includes a full example run for `1ua7` under:

- `output/1ua7/preprocessing/`
- `output/1ua7/mutation/`
- `output/1ua7/folding/`
- `output/1ua7/analysis/`

## Current Caveats

- Protected residue handling is central to the method; ensure preprocessing finds meaningful ligand-adjacent residues.
- Indexing is mixed across the codebase (FASTA vs PDB numbering), especially during analysis.
- In `run_folding.py`, the `--betas` CLI argument is parsed but not passed into `main()`.
- TM-score fields may appear empty in analysis outputs depending on alignment/parsing path.
- Several utility scripts in `scripts/tools/` use hardcoded file paths and are intended as ad-hoc helpers.

## HPC Usage

Cluster submission examples are provided in `queueing_scripts/` (`run_pipeline.pbs`, `run_mutation.pbs`, `run_analysis.pbs`, etc.).

## Citation

If you use this code, cite your thesis/project and the underlying models:

- ESM-2 protein language models
- ESMFold
