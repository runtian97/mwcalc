import os
import shutil
from collections import defaultdict
from rdkit import Chem


def canonicalize_smi(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol:
        return Chem.MolToSmiles(mol, canonical=True)
    return None


def read_smi_file(smi_path):
    """Return set of canonical SMILES from a .smi file."""
    smiles = set()
    with open(smi_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            smi = line.strip().split()[0]
            canon = canonicalize_smi(smi)
            if canon:
                smiles.add(canon)
    return smiles


def find_smi_in_subdir(subdir):
    """Find the first .smi file in a directory."""
    for f in os.listdir(subdir):
        if f.endswith('.smi'):
            return os.path.join(subdir, f)
    return None


def collect_mol_smiles_in_small_folder(small_folder_path):
    """Collect SMILES from all mol folders within a small folder."""
    mol_to_smiles = {}
    smiles_to_mols = defaultdict(set)

    if not os.path.isdir(small_folder_path):
        return mol_to_smiles, smiles_to_mols

    for mol_dir in sorted(os.listdir(small_folder_path)):
        mol_path = os.path.join(small_folder_path, mol_dir)
        if not os.path.isdir(mol_path):
            continue

        smi_path = find_smi_in_subdir(mol_path)
        if not smi_path:
            continue

        smi_set = read_smi_file(smi_path)
        if smi_set:  # Only add if there are valid SMILES
            mol_to_smiles[mol_dir] = smi_set
            for smi in smi_set:
                smiles_to_mols[smi].add(mol_dir)

    return mol_to_smiles, smiles_to_mols


def find_duplicates_in_small_folder(small_folder_path):
    """Find duplicate mol folders within a small folder based on SMILES content."""
    mol_to_smiles, smiles_to_mols = collect_mol_smiles_in_small_folder(small_folder_path)

    # Group folders by their SMILES content signature
    smiles_signature_to_folders = defaultdict(list)
    for mol_dir, smiles_set in mol_to_smiles.items():
        # Create a signature from the sorted SMILES to identify identical sets
        signature = tuple(sorted(smiles_set))
        smiles_signature_to_folders[signature].append(mol_dir)

    duplicates = []
    for signature, folders in smiles_signature_to_folders.items():
        if len(folders) > 1:
            # Keep the first folder (alphabetically), mark the rest as duplicates
            folders_sorted = sorted(folders)
            keep_folder = folders_sorted[0]
            remove_folders = folders_sorted[1:]

            duplicates.append({
                'smiles_signature': signature,
                'keep': keep_folder,
                'remove': remove_folders,
                'all_folders': folders_sorted
            })

    return duplicates, mol_to_smiles, smiles_to_mols


def remove_duplicates_in_small_folder(small_folder_path, small_folder_name, dry_run=True):
    """Remove duplicate mol folders within a small folder."""
    print(f"\n📁 Processing small folder: {small_folder_name}")

    duplicates, mol_to_smiles, smiles_to_mols = find_duplicates_in_small_folder(small_folder_path)

    if not duplicates:
        print(f"  ✅ No duplicates found in {small_folder_name}")
        return []

    print(f"  🔍 Found {len(duplicates)} duplicate group(s):")

    folders_to_remove = []
    for i, dup_info in enumerate(duplicates, 1):
        print(f"\n    🔄 Duplicate group {i}:")
        print(f"       SMILES: {list(dup_info['smiles_signature'])}")
        print(f"       📁 Keeping: {dup_info['keep']}")
        print(f"       🗑️  Removing: {dup_info['remove']}")

        folders_to_remove.extend(dup_info['remove'])

    if dry_run:
        print(f"\n  ⚠️  DRY RUN: Would remove {len(folders_to_remove)} mol folders from {small_folder_name}")
        return folders_to_remove
    else:
        print(f"\n  🗑️  Removing {len(folders_to_remove)} duplicate mol folders...")
        removed_count = 0
        for folder_name in folders_to_remove:
            folder_path_full = os.path.join(small_folder_path, folder_name)
            try:
                shutil.rmtree(folder_path_full)
                print(f"       ✅ Removed: {folder_name}")
                removed_count += 1
            except Exception as e:
                print(f"       ❌ Failed to remove {folder_name}: {e}")

        print(f"  ✅ Successfully removed {removed_count} mol folders from {small_folder_name}")
        return folders_to_remove


def process_main_folder(main_folder_path, dry_run=True):
    """Process all small folders within the main folder to remove duplicates."""
    print("=" * 80)
    print(f"🔍 HIERARCHICAL SMILES DEDUPLICATION")
    print(f"📂 Main folder: {main_folder_path}")
    print(f"🔧 Mode: {'DRY RUN' if dry_run else 'LIVE REMOVAL'}")
    print("=" * 80)

    if not os.path.isdir(main_folder_path):
        print(f"❌ Error: Main folder does not exist: {main_folder_path}")
        return

    small_folders = [f for f in os.listdir(main_folder_path)
                     if os.path.isdir(os.path.join(main_folder_path, f))]

    if not small_folders:
        print("❌ No small folders found in the main directory")
        return

    print(f"📊 Found {len(small_folders)} small folders to process\n")

    total_removed = []
    summary = {}

    for small_folder_name in sorted(small_folders):
        small_folder_path = os.path.join(main_folder_path, small_folder_name)
        removed_folders = remove_duplicates_in_small_folder(
            small_folder_path, small_folder_name, dry_run=dry_run
        )

        if removed_folders:
            total_removed.extend(removed_folders)
            summary[small_folder_name] = removed_folders

    # Final summary
    print("\n" + "=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)

    if not summary:
        print("✅ No duplicates found in any small folder!")
    else:
        print(f"📊 Total small folders with duplicates: {len(summary)}")
        print(f"🗑️  Total mol folders {'to be removed' if dry_run else 'removed'}: {len(total_removed)}")

        print(f"\n📝 Detailed breakdown:")
        for small_folder, removed_list in summary.items():
            print(f"  📁 {small_folder}: {len(removed_list)} mol folders")
            for mol_folder in removed_list:
                print(f"    - {mol_folder}")

    if dry_run and total_removed:
        print(f"\n⚠️  To actually remove the duplicates, run with dry_run=False")

    return summary


def analyze_small_folder_details(main_folder_path, small_folder_name):
    """Detailed analysis of a specific small folder - useful for debugging."""
    small_folder_path = os.path.join(main_folder_path, small_folder_name)

    print(f"\n🔍 DETAILED ANALYSIS: {small_folder_name}")
    print("=" * 60)

    mol_to_smiles, smiles_to_mols = collect_mol_smiles_in_small_folder(small_folder_path)

    if not mol_to_smiles:
        print("❌ No valid mol folders with .smi files found")
        return

    print(f"📊 Found {len(mol_to_smiles)} mol folders with SMILES data:")

    for mol_dir, smiles_set in sorted(mol_to_smiles.items()):
        print(f"\n📂 {mol_dir}:")
        for smi in sorted(smiles_set):
            print(f"  🧪 {smi}")

    print(f"\n🔍 SMILES sharing analysis:")
    shared_found = False
    for smi, mol_dirs in smiles_to_mols.items():
        if len(mol_dirs) > 1:
            shared_found = True
            print(f"  🔁 SMILES: {smi}")
            print(f"     Found in: {sorted(mol_dirs)}")

    if not shared_found:
        print("  ✅ No shared SMILES found - all mol folders are unique")


# === Usage Examples ===

# Example usage:

# 1. Dry run - see what would be removed (SAFE)
# process_main_folder(main_folder, dry_run=True)

# 2. Actually remove duplicates (CAUTION!)
# process_main_folder(main_folder, dry_run=False)

# 3. Analyze a specific small folder in detail
# analyze_small_folder_details(main_folder, "small_folder_A")

# Default: Dry run for safety
if __name__ == "__main__":
    # Update this path to your actual main folder
    main_folder = "test"

    print("🚀 Starting hierarchical SMILES deduplication...")
    process_main_folder(main_folder, dry_run=False)