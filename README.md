# PMACE: Projected Multi-Agent Consensus Equilibrium 
*The documentation for this package is available here:*
>https://ptycho-pmace.readthedocs.io/

## Overview

PMACE is a framework for distributed reconstruction such as ptychographic image reconstruction.

This python package implements PMACE approach in python.

![](images/PMACE_flow.png)

<div align="center">
  Conceptual rendering of the PMACE pipeline.
</div>


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

[paper link](https://ieeexplore.ieee.org/document/9723357)
Zhai, Qiuchen, Brendt Wohlberg, Gregery T. Buzzard, and Charles A. Bouman. "Projected multi-agent consensus equilibrium for ptychographic image reconstruction." In 2021 55th Asilomar Conference on Signals, Systems, and Computers, pp. 1694-1698. IEEE, 2021, doi: 10.1109/IEEECONF53345.2021.9723357.

[paper link](https://arxiv.org/pdf/2303.15679.pdf)
Zhai, Qiuchen, Gregery T. Buzzard, Kevin Mertes, Brendt Wohlberg, and Charles A. Bouman. "Projected Multi-Agent Consensus Equilibrium (PMACE) for Distributed Reconstruction with Application to Ptychography." arXiv preprint arXiv:2303.15679 (2023).
