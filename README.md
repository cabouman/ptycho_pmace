# Projected Multi-Agent Consensus Equilibrium (PMACE) for Ptychographic Image Reconstruction

## Overview
This python package implements PMACE and other competing algorithms including WF, AWF, and SHARP for solving ptychographic 
reconstruction problem. 

## Requirements
The required packages are included in requirements.txt. Please build virtual environment first,
```console
conda create -n ptycho python=3.8
conda activate ptycho
```
Then install the required packages through
```console
pip install -r requirements.txt
```

## Files
#### 1. demo <br/>
This folder contains the demo to demonstrate the generation of synthetic data (phaseless measurements) and 
PMACE reconstruction of complex object by processing the simulated data.
#### 2. utils <br/>
It contains functions such as error calculation, etc. 
#### 3. ptycho <br/>
It contains specific functions of PMACE and other algorithms for differnet use. 
#### 4. configs <br/>
It contains configuration files for generating synthetic data and reconstruction demo.


## Runing the demo
1. Download ground truth images and synthetic data at: [Google Drive](https://drive.google.com/drive/folders/1zB9ANOWs3e_mDc7_hputzLKOpG8ThqNo?usp=sharing).
2. In /configs/demo_pmace.yaml, specify data directory for input and output directory for saving reconstruction results.
3. Run /demo/demo_pmace.py.


## Other
Constructing this repository is a work in progress. The details and documentation remain to be improved, and the code needs to be cleared and modified. Please let me know if you have any suggestions or questions.

