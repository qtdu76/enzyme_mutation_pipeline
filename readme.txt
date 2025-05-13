the nameing of the analysis output files needs to be changed if using esm or pdb as ref
the protected indices as well in the anlysis script for the same reason. 
more specifically, in the anlyze.py script, if you are using esm, you need to use fasta indices twice, if you are using pdb, using fasta once and pdb for the other. 
each step can either be run individually or using the pipeline script, although more likely to get erros

 