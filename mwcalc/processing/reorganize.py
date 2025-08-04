import re
import os
import shutil
import math

def reorganize_output(output_dir):
    """
    Reorganize all combo_* folders under output_dir into:
      output_dir/
        H_<h_ratio>_N_<n_ratio>/
          N_<n>Si_<si>H_<h>/
            <disproportionation>/
              combo_<…>/
    """
    # Species in order of decreasing name-length, so prefixes match correctly
    species_keys = ['SiH3','SiH2','SiH','NH2','NH','Si','N']
    # How many H atoms each species carries
    species_h_count = {'N':0, 'NH':1, 'NH2':2, 'Si':0, 'SiH':1, 'SiH2':2, 'SiH3':3}

    for entry in os.listdir(output_dir):
        src = os.path.join(output_dir, entry)
        if not os.path.isdir(src) or not entry.startswith('combo_'):
            continue

        # Parse counts from folder name
        tail = entry[len('combo_'):]  # e.g. "N8_SiH1_SiH210_SiH31"
        tokens = tail.split('_')
        counts = {}
        for token in tokens:
            for sp in species_keys:
                if token.startswith(sp):
                    num = token[len(sp):]
                    if num.isdigit():
                        counts[sp] = int(num)
                        break
            else:
                print(f"  ⚠️  Unrecognized token in {entry}: {token}")

        # Sum up N, Si, H atoms
        n_atoms  = counts.get('N',0) + counts.get('NH',0) + counts.get('NH2',0)
        si_atoms = counts.get('Si',0) + counts.get('SiH',0) + counts.get('SiH2',0) + counts.get('SiH3',0)
        h_atoms  = sum(species_h_count[sp] * counts.get(sp,0) for sp in species_keys)

        # Compute lowest‐term H:N ratio
        if n_atoms > 0 and h_atoms > 0:
            g = math.gcd(h_atoms, n_atoms)
        else:
            g = 1
        h_ratio = h_atoms // g
        n_ratio = n_atoms // g

        # Degree of disproportionation = Si + SiH3
        disp = counts.get('Si',0) + counts.get('SiH3',0)

        # Build new path
        ratio_folder   = f"H_{h_ratio}_N_{n_ratio}"
        formula_folder = f"N_{n_atoms}Si_{si_atoms}H_{h_atoms}"
        disp_folder    = str(disp)

        dst = os.path.join(output_dir, ratio_folder, formula_folder, disp_folder)
        os.makedirs(dst, exist_ok=True)

        dst_path = os.path.join(dst, entry)
        print(f"Moving {entry} → {ratio_folder}/{formula_folder}/{disp_folder}/{entry}")
        shutil.move(src, dst_path)

if __name__ == '__main__':
    out_dir = 'test_molecules'
    # ... (any prior validation steps) ...
    reorganize_output(out_dir)
    print("Reorganization complete!")
