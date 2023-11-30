#!/bin/bash

# Downloads and installs the correct version of PyTorch based on Python and CUDA versions.

# Print executed commands.
set -x

# Configure Python executable.
# PYTHON_BINARY="python"

# Get the CUDA version.
CUDA_VERSION="$(${PYTHON_BINARY} nvcc_version_parser.py)"

# Install correct version of PyTorch.
# "$(${PYTHON_BINARY} -m pip install torch --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION})"
$(pip install torch --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION})
