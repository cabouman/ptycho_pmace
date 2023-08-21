#!/bin/bash
# This script purges the docs and rebuilds them.
# You can view documentation pages from ptycho_pmace/docs/build/index.html .

# Build documentation
cd ../docs
/bin/rm -r _build

make clean html
make html

echo ""
echo "*** The html documentation is at ptycho_pmace/docs/build/html/index.html ***"
echo ""

cd ../dev_scripts
