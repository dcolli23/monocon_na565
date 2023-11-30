import subprocess

out = subprocess.check_output(["nvcc", "--version"], text=True)

pre_version_str = "release "
version_loc = out.find(pre_version_str) + len(pre_version_str)

# Version info is always 2 numbers for major version and 1 number for minor version (at least for
# versions I've seen).
version_str = out[version_loc:version_loc + 4].replace('.', '')
# print("Found nvcc (CUDA) version:", version_str)
print(version_str)
