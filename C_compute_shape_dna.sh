#!/bin/bash

# This script will run Shape DNA of a series of meshes (.off files) geenrated with A_* and B_*
#
# The result is a series of stats saved into a *.ev (Eigen Values) file
# This needs the software Shape-DNA, which can be obtained from:
#  http://reuter.mit.edu/software/shapedna/
#
# Lindsey Kitchell, Brain-Life Team, Indiana University, Copyright 2017 

export SHAPEDNA_HOME=/Users/lindseykitchell/Documents/Github_repos/shapeDNA-tria
for f in *.off;
do
    $SHAPEDNA_HOME/shapeDNA-tria --mesh $f --num 50 --normfirst

done
