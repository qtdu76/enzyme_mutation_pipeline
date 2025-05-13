# run_preprocessing_driver.py

import sys
from preprocessing import run_preprocessing  # replace with actual filename (no .py)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_preprocessing_driver.py <PDB_CODE>")
        sys.exit(1)

    pdb_code = sys.argv[1].lower()
    run_preprocessing(pdb_code)
