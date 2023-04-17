from setuptools import setup, find_packages


setup(
   name='ptycho_pmace',
   description='Python code for PMACE (Projected Multi-Agent Consensus Equilibrium)',
   maintainer='Qiuchen Zhai',
   maintainer_email='qzhai@purdue.edu',
   url="https://github.com/cabouman/ptycho_pmace",
   license="BSD-3-Clause",
   packages=find_packages(),
   install_requires=['numpy==1.22.*', 'matplotlib>=3.5', 'scipy==1.8.0', 'pandas==1.4.2',
                     'tifffile==2022.5.4', 'pyfftw==0.13.0', 'PyYAML==6.0',
                     'imagecodecs==2022.2.22', 'scico', 'imageio==2.19.2', 'h5py==3.7.0'],  #external packages as dependencies
)


