# PMACE: Projected Multi-Agent Consensus Equilibrium

[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue.svg)](https://ptycho-pmace.readthedocs.io/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

*The documentation for this package is available here:*
> https://ptycho-pmace.readthedocs.io/

## Overview

PMACE (Projected Multi-Agent Consensus Equilibrium) is a powerful framework for distributed reconstruction, particularly designed for ptychographic image reconstruction. This Python package implements the PMACE approach with advanced features for multi-mode probe reconstruction and position refinement.

![PMACE Pipeline](images/PMACE_flow.png)

<div align="center">
  Conceptual rendering of the PMACE pipeline.
</div>

## Key Features

- **Projected Multi-Agent Consensus Equilibrium**: Implements the core PMACE algorithm for distributed reconstruction
- **Multi-Mode Probe Reconstruction**: Support for complex probe modes with automatic mode detection
- **Position Refinement**: Advanced scan position refinement with parallel grid search
- **Regularization**: Built-in BM3D/BM4D denoising for improved reconstruction quality
- **Flexible Configuration**: YAML-based configuration system for easy parameter tuning
- **Comprehensive Testing**: Multiple test datasets including synthetic and real-world data

## Installation
1. Download the source code:

   Move to a directory of your choice and run the following commands.

   ```console
   git clone https://github.com/cabouman/ptycho_pmace.git
   cd ptycho_pmace
   ```
	

2. Installation:

   It is recommended that you install the package to a virtual environment. Install Anaconda and follow any of the two methods.

* 2.1. Easy installation: If you have Anaconda installed, run the following commands.
           
    ```console
    cd dev_scripts
    source ./clean_install_all.sh
    cd ..
    ```
    
* 2.2. Manual installation: Note the ``pmace`` environment needs to be activated every time you use the package.

	 - 2.2.1 Create a Virtual Environment:

		```console
		conda create --name pmace python=3.8
		conda activate pmace
		```

	 - 2.2.2 Install the dependencies:

		```console
		pip install -r requirements.txt
		```

	 - 2.2.3 Install the package:

		```console
		pip install .
		```

	 - 2.2.4 Install the documentation:
		```console
		cd docs/
		pip install -r requirements.txt
		make clean html
		cd ..
		```

3. Validate installation:

   Once the installation is done, you can validate the installation by running the demo script.
   
   ```console
   cd demo/
   pyhon demo_pmace.py
   ```


## Related publications

For more information on the methods implemented in this code, please see:

**[PMACE for Ptychography (2021)](https://ieeexplore.ieee.org/document/9723357)**
Zhai, Qiuchen, Brendt Wohlberg, Gregery T. Buzzard, and Charles A. Bouman. "Projected multi-agent consensus equilibrium for ptychographic image reconstruction." In 2021 55th Asilomar Conference on Signals, Systems, and Computers, pp. 1694-1698. IEEE, 2021, doi: 10.1109/IEEECONF53345.2021.9723357.

**[PMACE for Distributed Reconstruction (2023)](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10301493)**
Zhai, Qiuchen, Gregery T. Buzzard, Kevin Mertes, Brendt Wohlberg, and Charles A. Bouman. "Projected multi-agent consensus equilibrium (PMACE) with application to ptychography." IEEE Transactions on Computational Imaging 9 (2023): 1058-1070..

**[Blind Multi-Mode PMACE for Ptychography (2025)](https://arxiv.org/pdf/2501.06470)**
Zhai, Qiuchen, Gregery T. Buzzard, Kevin Mertes, Brendt Wohlberg, and Charles A. Bouman. "Ptychography using Blind Multi-Mode PMACE." arXiv preprint arXiv:2501.06470 (2025).

## Contact

- **Maintainer**: Qiuchen Zhai (qzhai@alumni.purdue.edu)
- **Repository**: https://github.com/cabouman/ptycho_pmace
