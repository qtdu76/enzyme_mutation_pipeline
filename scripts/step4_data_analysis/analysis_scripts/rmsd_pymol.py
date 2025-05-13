from pymol import cmd
from scripts.logging_config import setup_logger
from .loading import format_indices_for_pymol  
import traceback

def calc_rmsd_pymol(reference_file, target_file, pdb_code, protected_indices_fasta, protected_indices_pdb):
    """
    Calculate the RMSD between reference and target structures using PyMOL.

    Args:
        reference_file (str): Path to the reference PDB (PDB numbering).
        target_file (str): Path to the target PDB (FASTA numbering).
        pdb_code (str): Used for logging.
        protected_indices_fasta (list[int]): 1-based FASTA residue indices (for target).
        protected_indices_pdb (list[int]): 1-based PDB residue indices (for reference).

    Returns:
        tuple: (global RMSD, protected RMSD)
    """
    logger = setup_logger('Rmsd', pdb_code)

    try:

        # Load structures
        cmd.load(reference_file, "reference")
        cmd.load(target_file, "target")
        logger.info("Structures loaded successfully.")

        # Global RMSD
        rmsd_global = cmd.pair_fit("reference and name CA", "target and name CA")
        logger.info(f"Global RMSD: {rmsd_global:.4f}")

        # Format PyMOL selections
        selection_ref = format_indices_for_pymol(protected_indices_pdb)
        selection_target = format_indices_for_pymol(protected_indices_fasta)

        logger.info(f"Protected selection — reference: {selection_ref}")
        logger.info(f"Protected selection — target: {selection_target}")

        # === 🔍 NEW: Check that residue names match at selected indices ===
        def get_residue_info(structure_name, selection):
            mapping = {}
            model = cmd.get_model(f"{structure_name} and ({selection})")
            for atom in model.atom:
                resi = int(atom.resi)
                resn = atom.resn
                if resi not in mapping:
                    mapping[resi] = resn
            return mapping

        ref_residues = get_residue_info("reference", selection_ref)
        target_residues = get_residue_info("target", selection_target)

        mismatch_log = []
        for ref_resi, target_resi in zip(sorted(ref_residues), sorted(target_residues)):
            ref_resn = ref_residues[ref_resi]
            target_resn = target_residues[target_resi]
            if ref_resn != target_resn:
                mismatch_log.append(f"ref: {ref_resi}:{ref_resn} ≠ tar: {target_resi}:{target_resn}")

        if mismatch_log:
            log_text = "❌ Residue mismatch at protected indices:\n" + "\n".join(mismatch_log)
            logger.error(log_text)
            raise ValueError("Residue mismatch between reference and target protected selections.")
        else:
            logger.info("✅ Residue names match at all protected indices.")

        # Select protected regions
        cmd.select("protected_ref", f"reference and ({selection_ref})")
        cmd.select("protected_target", f"target and ({selection_target})")

        num_ref = cmd.count_atoms("protected_ref")
        num_target = cmd.count_atoms("protected_target")
        logger.info(f"Selected atoms — reference: {num_ref}, target: {num_target}")

        # Protected RMSD
        # rmsd_results = cmd.super("protected_ref", "protected_target", cycles=0)
        # rmsd_protected = rmsd_results[0]

        rmsd_protected = cmd.pair_fit("protected_ref", "protected_target")

        logger.info(f"Protected RMSD: {rmsd_protected:.4f}")

        # Cleanup
        cmd.delete("all")

        return rmsd_global, rmsd_protected
    
    except Exception as e:
        logger.error(f"❌ PyMOL RMSD calculation failed for {pdb_code}:\n{traceback.format_exc()}")
        raise  # Re-raise so the outer try-except also catches it
