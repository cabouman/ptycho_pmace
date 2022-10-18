# Projected Multi-Agent Consensus Equilibrium (PMACE) for Ptychographic Image Reconstruction

## Overview
This python package implements PMACE approach for solving ptychographic 
reconstruction problem. 

## Requirements
The required packages are included in requirements.txt. Please build virtual environment first,
```console
conda create -n ptycho python=3.8
conda activate ptycho
```
Then install the required packages through
```console
python setup.py install
```

## Files
#### 1. demo <br/>
This folder contains the demo to demonstrate PMACE reconstruction of complex object by processing the simulated data.
#### 2. utils <br/>
It contains functions needed for the package, etc. 
#### 3. ptycho <br/>
It contains specific functions of PMACE and other algorithms for differnet use. 
#### 4. experiment <br/>
It contains experiment files that reproduce the reconstruction results on both synthetic data and real data.


## Runing the demo
1. Download ground truth images and synthetic data at: [Google Drive](https://drive.google.com/drive/folders/1feA5LdkEjVJhqhyFRu7ErgqwKa9Nbkxp?usp=sharing).
2. Specify data directory and output directory in configuration file /demo/config/demo_pmace.yaml
3. Run /demo/demo_pmace.py.
```console
cd demo/
python demo_pmace.py
```

## Related publications
For more information on the methods implemented in this code, please see:

[paper link](https://ieeexplore.ieee.org/document/9723357)
Q. Zhai, B. Wohlberg, G. T. Buzzard and C. A. Bouman, "Projected Multi-Agent Consensus Equilibrium for Ptychographic Image Reconstruction," 2021 55th Asilomar Conference on Signals, Systems, and Computers, 2021, pp. 1694-1698, doi: 10.1109/IEEECONF53345.2021.9723357.

## Other
Constructing this repository is a work in progress. The details and documentation remain to be improved, and the code needs to be cleaned and improved. Please let me know if you have any suggestions or questions.

