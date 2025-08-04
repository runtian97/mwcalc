from setuptools import setup, find_packages
setup(
    name="mwcalc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "rdkit>=2022.3.1",
        "tqdm>=4.60.0",
        "ase>=3.22.0",
        "pyyaml>=6.0",
        "fairchem-core"
    ],
    entry_points={"console_scripts": ["mwcalc=mwcalc.pipeline.workflow:main"]},
)
