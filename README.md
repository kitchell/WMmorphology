# WMmorphology
code for white matter morphology project

Steps for studying white matter morphology

1. Preprocess diffusion files
2. run AFQ and Dan's fibers
3. Using the output from AFQ, run script to create volumes from each fiber group - need to get from Dan
4. run script to create surfaces from each volume - createsurfallfibers.m
5. run sript to get shapeDNA of each surface file - get_shapeDNA
6. read .ev (shapeDNA) files into python - parse_ev_files.py
7. do some ML on the data


# Preprocessing steps for HCP data

1. download the subjects from HCP
2. the T1 is already ACPC aligned, just going to use that for now, otherwise do ACPC alignment
3. split the diffusion data into separate shells, going to use shell 2000 for now
    run batch_split.m
4. do Dti init
