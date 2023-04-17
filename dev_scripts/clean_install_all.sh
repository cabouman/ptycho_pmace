#!/bin/bash

# Clean out old installation
source clean_ptycho_pmace.sh

# Create conda environment and install package
source install_conda_env.sh

# Build documentation
source install_docs.sh