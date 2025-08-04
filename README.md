# MW Calc

A Python package for Si-N molecular generation, deduplication, reorganization, and optimization.

## Installation

```bash
# Install OpenBabel via conda
conda install conda-forge::openbabel

pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121  # for CUDA 12.1


# Install the package
git clone https://github.com/runtian97/mwcalc.git
cd mwcalc
pip install .
```

## Usage

### Complete Workflow
```python
from mwcalc import MolecularWorkflowPipeline

pipeline = MolecularWorkflowPipeline()
results = pipeline.run_complete_workflow(
    target_mw=473.352,
    output_dir="molecules",
    num_mols_per_combo=2000,
    tolerance=0,
    random_seed=0
)
```

### Step-by-Step
```python
from mwcalc.generation import generate_sin_only_molecules
from mwcalc.processing import remove_duplicates, reorganize_output
from mwcalc.optimization import optimize_folder_xyz

# 1. Generate molecules
generate_sin_only_molecules(
    target_mw=473.352,
    output_dir="molecules",
    num_mols_per_combo=2000,
    tolerance=0,
    random_seed=0
)

# 2. Remove duplicates  
remove_duplicates("molecules", dry_run=False)

# 3. Reorganize
reorganize_output("molecules")

# 4. Optimize (optional)
optimize_folder_xyz("molecules", "/path/to/model.pt")
```

### Command Line
```bash
mw-calculator run --target-mw 473.352 --output-dir molecules --num-mols 2000 --tolerance 0 --random-seed 0
```
