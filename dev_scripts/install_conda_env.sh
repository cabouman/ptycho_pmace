#!/bin/bash
# Remove and create conda environment, install package locally

cd ..
conda deactivate
conda remove --name pmace --all
conda create --name pmace python=3.8
conda activate pmace
pip install -r requirements.txt
pip install -e .
cd dev_scripts