#!/bin/bash
# This script destroys the conda environment for this pacakage and reinstalls it.

# Create and activate new conda environment
NAME=pmace
if [ "$CONDA_DEFAULT_ENV"==$NAME ]; then
    conda deactivate
fi

conda remove env --name $NAME --all
conda create --name $NAME python=3.8
conda activate $NAME

echo
echo "Use 'conda activate" $NAME "' to activate this environment."
echo

# Install pmace along with all requirements for the package, demos and docs.
cd ..
pip install -r requirements.txt
pip install -e .
pip install -r docs/requirements.txt
cd dev_scripts
