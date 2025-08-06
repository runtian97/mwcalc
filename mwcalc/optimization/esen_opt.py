import os
import numpy as np
import pandas as pd
from ase.io import read
from ase.optimize.lbfgs import LBFGS
from fairchem.core.units.mlip_unit import load_predict_unit
from fairchem.core import FAIRChemCalculator
from openbabel import pybel
import torch

EV_TO_KCAL = 23.0605

def ESEN_ASE(mol_path, model_path, device="cpu", fmax=0.02, steps=500):
    # 1) Load geometry
    mol_atoms = read(mol_path)

    # 2) Capture charge & spin via Pybel
    mol_py = next(pybel.readfile("xyz", mol_path))
    charge, mult = mol_py.charge, mol_py.spin
    mol_atoms.info = {"charge": charge, "spin": mult}

    # 3) Assign calculator
    predictor = load_predict_unit(path=model_path, device=device)
    calc = FAIRChemCalculator(predictor)
    mol_atoms.set_calculator(calc)

    # 4) Optimize
    base = os.path.splitext(os.path.basename(mol_path))[0]
    traj = f"esen_{base}_opt.traj"
    log  = f"{base}.log"
    LBFGS(mol_atoms, trajectory=traj, logfile=log).run(fmax=fmax, steps=steps)

    # 5) Extract energy & force
    energy = mol_atoms.get_potential_energy()
    forces = mol_atoms.get_forces()
    max_force = np.max(np.linalg.norm(forces, axis=1))

    # 6) Update coordinates & metadata
    for atom, pos in zip(mol_py.atoms, mol_atoms.get_positions()):
        atom.OBAtom.SetVector(*pos)
    mol_py.data.clear()
    mol_py.data.update({
        "Energy(eV)":      f"{energy:.6f}",
        "Max force(eV/Å)": f"{max_force:.6f}"
    })

    # 7) Overwrite original file
    temp = f"{base}_temp.xyz"
    mol_py.write("xyz", temp, overwrite=True)
    os.replace(temp, mol_path)

    # 8) Cleanup
    if os.path.exists(traj):
        os.remove(traj)

    return energy

def optimize_folder_xyz(root_dir, model_path, device="cuda" if torch.cuda.is_available() else "cpu", fmax=0.02):
    records = []
    for dp, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".xyz"):
                path = os.path.join(dp, fn)
                try:
                    e = ESEN_ASE(path, model_path, device=device, fmax=fmax)
                    rel = os.path.relpath(path, root_dir)
                    records.append((rel, e))
                except Exception as exc:
                    print(f"❌ Failed {path}: {exc}")

    # build DataFrame
    df = pd.DataFrame(records, columns=["path", "absolute_eV"])
    df["relative_eV"]   = df["absolute_eV"] - df["absolute_eV"].min()
    df["relative_kcal"] = df["relative_eV"] * EV_TO_KCAL

    # sort and write CSV
    df = df.sort_values("absolute_eV").reset_index(drop=True)
    csv_file = os.path.join(root_dir, "optimized_energies_summary.csv")
    df.to_csv(csv_file, index=False)

    # also write text summary
    txt = os.path.join(root_dir, "optimized_energies_summary.txt")
    with open(txt, "w") as out:
        for p, e in df[["path","absolute_eV"]].values:
            out.write(f"{p}\t{e:.6f} eV\n")

    print(f"\n✅ CSV summary written to: {csv_file}")
    print(f"📄 Text summary written to: {txt}")

if __name__ == "__main__":
    ROOT_FOLDER = "/Users/nickgao/Desktop/pythonProject/Merck/Mw_calculator/generated_molecules_473_copy"
    MODEL_FILE  = "/Users/nickgao/Desktop/pythonProject/FAIRchem/esen_sm_conserving_all.pt"

    optimize_folder_xyz(ROOT_FOLDER, MODEL_FILE, device="cpu", fmax=0.02)
