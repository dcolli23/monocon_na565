#!/bin/bash

# Turn on printing of commands.
set -x

# Setup Constants
ANACONDA_ENVIRONMENT_NAME="rob535_monocon"
PYTHON_VERSION="3.8"

# Get the script directory so we don't have to worry about relative file paths.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "${SCRIPT_DIR}"

## Get and extract data.
## ---------------------
./get_extract_project_data.sh

## Clone the repository.
## ---------------------
git clone https://github.com/dcolli23/monocon_na565.git
cd monocon_na565

## Setup Anaconda environment.
## ---------------------------
# This was adapted from the repo readme
# [Step 1]: Create new conda environment and activate.
#           Set [ENV_NAME] freely to any name you want. (Please exclude the brackets.)
conda create --name "${ANACONDA_ENVIRONMENT_NAME}" "python=${PYTHON_VERSION}"
# conda activate "${ANACONDA_ENVIRONMENT_NAME}"

conda run -n "${ANACONDA_ENVIRONMENT_NAME}" "${SCRIPT_DIR}/install_pytorch.sh"

# [Step 4]: Install some packages using 'requirements.txt' in the repository.
#           The version of numpy must be 1.22.4.
conda run -n "${ANACONDA_ENVIRONMENT_NAME}" pip install -r "${SCRIPT_DIR}/monocon_na565/requirements.txt"

# [Step 5]
conda run -n "${ANACONDA_ENVIRONMENT_NAME}" install cudatoolkit


## Completion Message
## ------------------
echo "Dev environment setup complete!"
echo "You can now activate your conda environment and begin running jobs"
