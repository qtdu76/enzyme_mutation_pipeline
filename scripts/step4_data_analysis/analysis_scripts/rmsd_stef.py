from Bio.PDB import PDBParser
from biotite.structure import superimpose, rmsd
from biotite.structure import AtomArray

import biotite.structure.io as strucio

import biotite.structure.io.pdb as pdb
import biotite.structure as struc
import numpy as np

from Bio import PDB
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def pre_process_pdb_structure( ref_pdb, mutant_pdb, active_residues_list:list, backbone_atoms_types : list = ["N", "CA", "C", "O"] )-> tuple:
    """This function takes a reference pdb and a mutant pdb and returns two
    pdbs that have been cleaned and ordered to make a correct RMSD.
    Cleaning and ordering means: 
    For residues id in active residues, it make sure the whole residue is
    present and with atoms in the same order in the reference and mutant. 
    For residues not in active residues, it does the same operation but
    only taking backbone atoms, not all atoms, since these might simply 
    not be present in both"""

    ref_structure = ref_pdb.get_structure()[0]
    mutant_structure = mutant_pdb.get_structure()[0]

    # need to re-order atoms in mutant_structure to match ref_structure
    new_ref_structure = [] 
    new_mutant_structure = []
    
    # get all residue ids and order them in growing number. 
    ref_residue_ids = np.unique([a.res_id for a in ref_structure])
    mutant_residue_ids = np.unique([a.res_id for a in mutant_structure])

    # This part of the algorithm re-orders atoms inside a residue to make sure
    # they have the same order in the reference vs mutant structure, otherwise 
    # rmsd calculation makes little sense!

    # This operation is done fully only for active_residues, for other residues
    # we only do this and store backbone atoms - other atoms might even not be 
    # present in both the reference and the mutant.
     
    # Additional assertion check make sure that no to atoms have the same atom_name attribute,
    # something which should never happen in a correct pdb but is not necessarily
    # enforced - and this situation would make calculation of RMSD ill-defined
    count = 0
    #print( ref_residue_ids, mutant_residue_ids )
    for ref_res_id, mutant_res_id in zip(ref_residue_ids, mutant_residue_ids):
        # Select /extract all atoms belonging to a residue
        ref_atoms = ref_structure[ref_structure.res_id == ref_res_id]
        mutant_atoms = mutant_structure[mutant_structure.res_id == mutant_res_id]
        previous_atom_names = set() 
        error = "Multiple atoms with same atom_name attribute in the same residue should not occur or RMSD is not clear" 
        if ref_res_id in active_residues_list:
            for ref_atom in ref_atoms:
                  # Check inside mutat_atoms the matching one. Once found, stop searching and just record it in ordered structure
                  for mutant_atom in mutant_atoms:
                      # The following assertion should always be true since active residues should always match!
                      assert ref_atom.res_name == mutant_atom.res_name, \
                        AssertionError(f"Active residues not matching: ref[{ref_res_id}]={ref_atom.res_name} vs mut[{mutant_res_id}]={mutant_atom.res_name}")

                      if ref_atom.atom_name == mutant_atom.atom_name:
                          assert ref_atom.atom_name not in previous_atom_names, AssertionError( error + f"{ref_atom.atom_name} already in {previous_atom_names}" )
                          previous_atom_names.add( ref_atom.atom_name )
                          new_ref_structure.append( ref_atom )
                          new_mutant_structure.append( mutant_atom )
                          #print( f"REF" )
                          #print( f"{ref_atom}" )
                          #print( f"MUTANT" )
                          #print( f"{mutant_atom}" )
                          count += 1
                          break
        else:
            for ref_atom in ref_atoms:
                  if ref_atom.atom_name in backbone_atoms_types:
                      for mutant_atom in mutant_atoms:
                          if ref_atom.atom_name == mutant_atom.atom_name:
                              assert ref_atom.atom_name not in previous_atom_names, AssertionError( error + f"{ref_atom.atom_name} already in {previous_atom_names}" )
                              previous_atom_names.add( ref_atom.atom_name )
                              new_ref_structure.append( ref_atom )
                              new_mutant_structure.append( mutant_atom )
                              #print( f"REF" )
                              #print( f"{ref_atom}" )
                              #print( f"MUTANT" )
                              #print( f"{mutant_atom}" )
                              count += 1
                              break
            
        #print( f"{count} {new_ref_structure[count-1]}" )
        #print( f"{count} {new_mutant_structure[count-1]}" )
        
    # assert same number of atoms
    assert len( new_mutant_structure) == len( new_ref_structure), f"number of atoms: {len(new_mutant_structure)} != {len(new_ref_structure)}"

    # Now build AtomArray object from list of atoms objects
    ref_structure_final = AtomArray(len( new_ref_structure ) )
    mutant_structure_final = AtomArray(len( new_ref_structure ) )

    # another sanity check just in case
    non_matching_atoms = []
    for count, ( a1, a2 ) in enumerate( zip(new_ref_structure, new_mutant_structure) ):
        if a1.element != a2.element:
            non_matching_atoms.append((a1, a2))
            print( "Found non matching pair:" )
            print( f"AAA {a1}" )
            print( f"BBB {a2}" )
        assert len(non_matching_atoms) == 0, f"Number of non-matching atoms: {len(non_matching_atoms)}"
        ref_structure_final[ count ] = a1
        mutant_structure_final[ count ] = a2

    return ( ref_structure_final, mutant_structure_final )

def get_coordinates( 
    pdb_file, 
    active_residues_list, 
    add_inactive = False, 
    backbone_only = False,
    backbone_atoms_types = ["N", "CA", "C", "O"] 
    ):
    """
    - pdb_file: name of pdb file from which to extract atomic positions
    - active_residues_list: residues ID of active site
    - add_inactive: whether to also calculate RMSD including non-active residues 
    - backbone_only: whether to only count backbone atoms EVEN for active site. 
                     when False, all atoms are included in active site, backbone
                     atoms for all the others
    - backbone_atoms_types: which atom types are counted as backbone atoms
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", pdb_file)
    atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                res_id = residue.get_id()[1]  # Get numerical residue ID
                if res_id in active_residues_list:
                    if not backbone_only:
                        atoms.extend(residue.get_atoms())  # All atoms
                    else:
                        for atom in residue:
                            if atom.get_name() in backbone_atoms_types: 
                                atoms.append(atom)  # Backbone atoms
                elif add_inactive:
                    for atom in residue:
                        if atom.get_name() in backbone_atoms_types: 
                            atoms.append(atom)  # Backbone atoms
                
    coords = np.array([atom.get_coord() for atom in atoms])
    
    return coords

def kabsch_rotation( 
    coords1, 
    coords2, 
    verbose = False 
    ):
    """Aligns two sets of coordinates using the Kabsch algorithm.
    In practice, this algorithm used SVD to find the best rotation that minimizes
    RMSD between two groups of atoms
    """
    # Compute centroids
    coords1 = centre( coords1 )
    coords2 = centre( coords2 )

    # Compute covariance matrix
    H = np.dot(coords1.T, coords2)

    # Singular Value Decomposition
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation (det(R) should be 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)
   
    if verbose: 
        print( "Rotation matrix:" )
        print( f"{R[0]}" )
        print( f"{R[1]}" )
        print( f"{R[2]}" )
 
    return R

def centre( coords ):
    centroid = np.mean(coords, axis=0)
    coords -= centroid
    return coords

def rotate( coords, R ):
    """Aligns two sets of coordinates using the Kabsch algorithm.
    In practice, this algorithm used SVD to find the best rotation that minimizes
    RMSD between two groups of atoms
    """
    # Rotate coords1
    coords = np.dot( R, coords.T ).T

    return coords

def plot_histo( series, file_name ):
    """Take series as input and plot the histogram of the distribution of values from it"""
    for i, ser in enumerate( series ):
        if not isinstance( ser, pd.Series ):
            ser = pd.Series( ser )
            series[ i ] = ser


    # Create a PDF file to save the plot
    with PdfPages( file_name ) as pdf:
        # Plot the histogram
        alphas = np.linspace( 0.5, 1, len( series ) )
        colors = [ "red", "green", "blue" ]
        names = [ "active", "all", "unknown" ]
        for i, ser in enumerate( series ):
            color = colors[ i ] if i < len( series )-1 else colors[ -1 ]
            name = names[ i ] if i < len( series )-1 else names[ -1 ]
            ser.plot(kind='hist', bins=10, density = True, label = name, edgecolor='black', alpha=0.5)

        # Labels and title
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Data')
        plt.legend()

        # Save the current figure to the PDF
        pdf.savefig()
        plt.close()  # Close the figure to free memory

    return

def compute_rmsd(
    pdb1: str, 
    pdb2: str, 
    active_residues_list: set, 
    temp_dir: str,
    add_inactive : bool = True, 
    verbose = False,
    backbone_atoms_types = ["N", "CA", "C", "O"] 
    )-> list:
    """
    Computes RMSD between two PDBs using active residues as alignment reference.
    Temporary PDBs are saved in a specified directory.

    """

    import os

    # if add_inactive:
    #     print( "NOTE 1: Calculate both RMSD of active sites AND other atoms" ) 
    #     print( "NOTE 2: For INACTIVE residues, only N, C, C-alpha, O are included, as other atoms differ across systems" )
    # else:
    #     print( "NOTE: Calculate RMSD of atoms in active site only" )
    # print( "IMPORTANT: Best rotation is taken to be the one which best aligns **atoms in active site only**" )

    # === Prepare temp folder ===
    os.makedirs(temp_dir, exist_ok=True)
    ref_path = os.path.join(temp_dir, "curr_ref.pdb")
    mut_path = os.path.join(temp_dir, "curr_mut.pdb")
    
    # First clean up PDB and make sure that atoms in active residues are properly aligned 
    ref_pdb = pdb.PDBFile.read( pdb1 )
    mut_pdb = pdb.PDBFile.read( pdb2 )
    ref_structure, mut_structure = pre_process_pdb_structure( ref_pdb, mut_pdb, active_residues_list, backbone_atoms_types = backbone_atoms_types )

    # Save cleaned-up coordinates in new .pdb
    strucio.save_structure(ref_path, ref_structure )
    strucio.save_structure(mut_path, mut_structure )

    # First calculate best rotation including active site only
    coords1 = get_coordinates( ref_path, active_residues_list, add_inactive = False, backbone_atoms_types = backbone_atoms_types ) 
    coords2 = get_coordinates( mut_path, active_residues_list, add_inactive = False, backbone_atoms_types = backbone_atoms_types )   
    assert len( coords1) == len( coords2 )
    #print( f"Total number of atoms part of active sites: in coords1 = {len( coords1)}, coords2 = {len(coords2)}" )

    # Centre and calculate best rotation to overlap coords1 to coords2, stores in R
    coords1 = centre(coords1)
    coords2 = centre(coords2)
    R = kabsch_rotation(coords1, coords2, verbose = False )

    # Now decide what to include in the calculation of the RMSD, if only active site or not.
    # NOTE: the rotation matrix is always calculated to get the best alignment of active site only!
    coords1 = get_coordinates( ref_path, active_residues_list, add_inactive = add_inactive, backbone_atoms_types = backbone_atoms_types )   
    coords2 = get_coordinates( mut_path, active_residues_list, add_inactive = add_inactive, backbone_atoms_types = backbone_atoms_types )   

    # Now centre new coordinates and rotate those of 1 to align to 2
    coords1 = centre(coords1) # Move centre of mass of coords1 to 0 0 0 
    coords2 = centre(coords2) # Move centre of mass of coords2 to 0 0 0 
    coords1 = rotate( coords1, R ) # Rotate coords1 to get best alignment to coords2

    if verbose:
        print( f"Aligned coords1" )
        print( coords1 )
        print( f"Aligned coords2" )
        print( coords2 )
        print( "ROTATION matrix" )
        print( f"{R[0]}" )
        print( f"{R[1]}" )
        print( f"{R[2]}" )

    # Compute RMSD, also store the x atom rmsd in delta so that you can plot the histogram 
    rmsd = np.sqrt(np.mean(np.sum((coords1 - coords2) ** 2, axis=1)))
    delta = np.sqrt(np.sum((coords1 - coords2) ** 2, axis=1))

    return rmsd, delta

from scripts.logging_config import setup_logger


def calc_rmsd_stef(reference_file, target_file, protected_indices, pdb_code, temp_dir):
    """
    Wrapper for compute_rmsd to match pipeline interface.

    Args:
        reference_file (str): Path to the reference PDB.
        target_file (str): Path to the predicted PDB.
        protected_indices (list[int]): 1-based residue indices (FASTA numbering).
        pdb_code (str): For logging.

    Returns:
        tuple: (global RMSD, protected RMSD)
    """
    logger = setup_logger("StefRMSD", pdb_code)
    logger.info("Starting Stef RMSD computation")

    # === Create a dedicated temp folder inside analysis output ===
    temp_dir = os.path.join(temp_dir, "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    active_residues_set = set(protected_indices)

    try:
        # Global RMSD: include all residues (protected + backbone-only inactive)
        rmsd_global, _ = compute_rmsd(
            reference_file,
            target_file,
            active_residues_set,
            temp_dir=temp_dir,
            add_inactive=True,
            verbose=False
        )

        # Local RMSD: only protected residues
        rmsd_protected, _ = compute_rmsd(
            reference_file,
            target_file,
            active_residues_set,
            temp_dir=temp_dir,
            add_inactive=False,
            verbose=False
        )

        logger.info(f"Stef RMSD — Global: {rmsd_global:.4f}, Protected: {rmsd_protected:.4f}")
        return rmsd_global, rmsd_protected

    except Exception as e:
        logger.error(f"Stef RMSD computation failed: {e}")
        raise


