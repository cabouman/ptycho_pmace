#!/bin/bash
# This script installs the documentation.
# You can view documentation pages from ptycho_pmace/docs/build/index.html .

# Build documentation
cd ../docs
pip install -r requirements.txt
make clean html
make html
cd ../dev_scripts
