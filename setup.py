from setuptools import setup, find_packages

setup(
    name="pylidar_leaf",
    version="1.0.0",
    description="A Python toolset for processing laser scanner CSV files to LAZ format",
    author="PyLidar Leaf Scanner Contributors",
    author_email="your.email@example.com",
    url="https://github.com/username/pylidar_leaf",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "laspy>=2.0",
        "glob2",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
)
