#!/bin/bash

# Downloads and extracts the final project data to your home directory.
# NOTE: This will lead to some duplication if multiple users are using the same computer/VM instance
#       but it should be fine since the data won't be modified.

# NOTE: Assuming that we've changed the working directory to where we want the data to be stored and
# called this script from there.
# TEST_DATA_ROOT_DIR="~/code/school/navarch565_self_driving_cars/"

# Turn on printing of commands as executed.
set -x;

# cd "${TEST_DATA_ROOT_DIR}"
mkdir final_project_data
cd final_project_data
curl -O https://curly-dataset-public.s3.us-east-2.amazonaws.com/NA_565/Final/FinalData.zip
unzip FinalData.zip
rm FinalData.zip

# Turn off printing of commands.
# set +x;