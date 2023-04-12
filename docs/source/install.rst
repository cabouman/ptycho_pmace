============
Installation 
============

The ``ptycho_pmace`` package currently is only available to download and install from source through GitHub.


**Downloading and installing from source**

1. Download the source code:

  Move to a directory of your choice and run the following commands.

	| ``git clone https://github.com/cabouman/ptycho_pmace.git``
	| ``cd ptycho_pmace``
	
  Alternatively, you can directly clone from GitHub and then enter the repository.

2. Installation:
  
	2.1. Create a Virtual Environment:

	  It recommended that you install the package to a virtual environment.
	  If you have Anaconda installed, you can run the following.
      
		| ``conda create --name pmace python=3.8``
		| ``conda activate pmace``
		
	2.2. Install the dependencies:
    
	  In order to install the dependencies, use the following command.
	  
		| ``python setup.py install``
		

  The installation is done. The ``pmace`` environment needs to be activated every time you use the package.

3. Validate installation:

  You can validate the installation by running a demo script.
  
	| ``cd demo``
	| ``python demo_pmace.py``

