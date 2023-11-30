#!/bin/bash

# Downloads and installs the correct version of PyTorch based on Python and CUDA versions.

# Print executed commands.
set -x

# Configure Python executable.
PYTHON_BINARY="python"

# This is an annoyance of running this script from within a Conda environment. You have to provide the path with which to find this script.
SCRIPT_DIR=$HOME

# Get the CUDA version.
echo "Finding CUDA version"
CUDA_VERSION="$(${PYTHON_BINARY} ${SCRIPT_DIR}/nvcc_version_parser.py)"
echo "Found CUDA version: ${CUDA_VERSION}"

# Install correct version of PyTorch.
# "$(${PYTHON_BINARY} -m pip install torch --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION})"
echo "Installing PyTorch"
pip install torch --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
