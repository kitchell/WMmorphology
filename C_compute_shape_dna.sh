#!/bin/bash

export SHAPEDNA_HOME=/Users/lindseykitchell/Documents/Github_repos/shapeDNA-tria

for f in *.off;
do
    $SHAPEDNA_HOME/shapeDNA-tria --mesh $f --num 50 --normfirst

done
