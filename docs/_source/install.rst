============
Installation
============

The ``ptycho_pmace`` package is currently only available to download and install from source available through GitHub at `ptycho_pmace <https://github.com/cabouman/ptycho_pmace>`_.



Downloading and installing from source
--------------------------------------

1. Download the source code:

  Move to a directory of your choice and run the following commands.

	| ``git clone https://github.com/cabouman/ptycho_pmace.git``
	| ``cd ptycho_pmace``

  Alternatively, you can directly clone from GitHub and then enter the repository.

2. Follow any of the two methods.

	2.1. Easy installation:

	  If you have Anaconda installed, run the following commands.

		| ``cd dev_scripts``
		| ``source ./clean_install_all.sh``
		| ``cd ..``

	2.2. Manual installation:

		2.2.1 Create a Virtual Environment:

		  It is recommended that you install the package to a virtual environment.
		  If you have Anaconda installed, you can run the following.

			| ``conda create --name pmace python=3.8``
			| ``conda activate pmace``

		2.2.2 Install the dependencies:

		  In order to install the dependencies, use the following command.

			``pip install -r requirements.txt``

		2.2.3 Install the documentation:

		  Use the following command to install the documentation.

			| ``cd docs``
			| ``pip install -r requirements.txt``
			| ``make clean html``
			| ``make html``
			| ``cd ..``

  The installation is done. The ``pmace`` environment needs to be activated every time you use the package.

3. Validate installation:

  You can validate the installation by running a demo script.

	| ``cd demo/``
	| ``python demo.py``
