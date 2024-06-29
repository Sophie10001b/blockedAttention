import os
import glob

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = "0.0.1"

def getExtension():
    rootDir = os.path.dirname(os.path.realpath(__file__))

    cppFile = glob.glob(os.path.join(rootDir, "csrc/**/*.cpp"), recursive=True)
    cudaFile = glob.glob(os.path.join(rootDir, "csrc/**/*.cu"), recursive=True)

    extension = CUDAExtension(
        "blockAttention",
        cppFile + cudaFile,
        include_dirs=[os.path.join(rootDir, "csrc"), os.path.join(rootDir, "csrc/cuda")]
    )
    return [extension]

setup(
    name="blockedAttention",
    version=__version__,
    python_requires=">=3.7",
    install_requires=[],
    ext_modules=getExtension(),
    cmdclass={
        "build_ext": BuildExtension
    },
    packages=find_packages()
)