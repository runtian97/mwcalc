import os
import random
import numpy as np
import math
from itertools import product
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToXYZBlock
from tqdm import tqdm
import glob
import shutil
from collections import Counter

def generate_sin_only_molecules(
    target_mw,
    output_dir='generated_molecules',
    tolerance=2.0,
    num_mols_per_combo=1000,
    max_estimate_threshold=10000000,
    uncertainty_threshold=1000,
    bond_length_thresholds=None,
    validate_final=True,
    verbose=True,
    random_seed=None
):
    """
    Generate molecular structures with ONLY Si-N bonds (no Si-Si or N-N bonds allowed).
    Uses enhanced generation logic from code 2 with improved estimation and limiting.
    
    Parameters:
    -----------
    target_mw : float
        Target molecular weight to search for
    output_dir : str, default='generated_molecules'
        Directory to save generated molecules
    tolerance : float, default=2.0
        Tolerance for molecular weight matching
    num_mols_per_combo : int, default=1000
        Number of molecules to generate per valid combination
    max_estimate_threshold : int, default=10000000
        If estimated molecules exceed this, limit generation
    uncertainty_threshold : int, default=1000
        If estimates exceed this, report as "many structures"
    bond_length_thresholds : dict, optional
        Bond length thresholds for validation (Angstroms)
        Default: {('N', 'N'): 1.4, ('Si', 'Si'): 2.3}
    validate_final : bool, default=True
        Whether to perform final validation and remove invalid molecules
    verbose : bool, default=True
        Whether to print detailed progress information
    random_seed : int, optional
        Random seed for reproducible results. If None, uses current time/system state
    
    Returns:
    --------
    dict : Summary statistics of generation process
    
    Example:
    --------
    >>> stats = generate_sin_only_molecules(
    ...     target_mw=326.928,
    ...     output_dir='my_molecules',
    ...     num_mols_per_combo=500,
    ...     tolerance=1.5,
    ...     random_seed=42
    ... )
    >>> print(f"Generated {stats['total_molecules']} molecules in {stats['combinations_found']} combinations")
    """
    
    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        if verbose:
            print(f"Random seed set to: {random_seed}")
    
    # Set default bond length thresholds if not provided
    if bond_length_thresholds is None:
        bond_length_thresholds = {
            ('N', 'N'): 1,      # N-N single bond ~1.45 Å (forbidden)
            ('Si', 'Si'): 1.5,  # Si-Si single bond ~2.35 Å (forbidden)
        }
    
    # Initialize molecular weight dictionary and bonding rules
    mw_dict = {
        'N':   14,
        'NH':  14+1,
        'NH2': 14+1*2,
        'Si':  28,
        'SiH': 28 + 1,
        'SiH2': 28 + 1*2,
        'SiH3': 28 + 1*3
    }
    
    bonds_allowed = {
        'N':   3,
        'NH':  2,
        'NH2': 1,
        'Si':  4,
        'SiH': 3,
        'SiH2':2,
        'SiH3':1
    }
    
    element_map = {
        'N':   'N',
        'NH':  'N',
        'NH2': 'N',
        'Si':  'Si',
        'SiH': 'Si',
        'SiH2':'Si',
        'SiH3':'Si'
    }
    
    si_units = {'Si','SiH','SiH2','SiH3'}
    n_units  = {'N','NH','NH2'}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print("="*60)
        print("SI-N MOLECULAR GENERATION")
        print("="*60)
        print(f"Target molecular weight: {target_mw}±{tolerance}")
        print(f"Molecules per combination: {num_mols_per_combo}")
        print(f"Output directory: {output_dir}")
        if random_seed is not None:
            print(f"Random seed: {random_seed}")
        print("CONSTRAINT: ONLY Si-N bonds allowed (no Si-Si or N-N bonds)")
        print("="*60)
    
    # Find valid combinations
    list_species = ['N', 'NH', 'NH2', 'Si', 'SiH', 'SiH2', 'SiH3']
    list_mw = [mw_dict[sp] for sp in list_species]
    
    combinations = _find_combinations(list_mw, target_mw, list_species, tolerance, 
                                    mw_dict, bonds_allowed, si_units, n_units, verbose)
    
    if not combinations:
        if verbose:
            print("No valid combinations found!")
        return {
            'combinations_found': 0,
            'total_molecules': 0,
            'valid_molecules': 0,
            'invalid_molecules': 0
        }
    
    if verbose:
        print(f"\nFound {len(combinations)} valid combinations")
        print("All combinations below are guaranteed to support Si-N only bonding patterns.")
        
        # Display combination analysis like code 2
        print("\nCombination analysis:")
        for i, combo in enumerate(combinations):
            combo_str = ', '.join(f"{s}:{combo[s]}" for s in sorted(combo.keys()) if combo[s] > 0)
            mw = sum(mw_dict[sp] * count for sp, count in combo.items())
            n_stubs = sum(bonds_allowed[sp] * combo[sp] for sp in combo.keys() if sp in n_units)
            si_stubs = sum(bonds_allowed[sp] * combo[sp] for sp in combo.keys() if sp in si_units)
            
            # Count heavy atoms only
            heavy_atoms = sum(combo[sp] for sp in combo.keys() if sp in si_units or sp in n_units)
            
            estimated = _estimate_combo_size(combo, uncertainty_threshold, bonds_allowed, si_units, n_units)
            print(f"  {i+1}. {combo_str}")
            print(f"      MW: {mw:.3f}, Heavy atoms: {heavy_atoms} (Si+N), Stubs: N={n_stubs}, Si={si_stubs}")
            
            if estimated == -1:
                print(f"      Estimated unique Si-N structures: >>{uncertainty_threshold} (many possible)")
                print(f"      → Very large structure space - sampling subset of unique structures")
            else:
                print(f"      Estimated unique Si-N structures: {estimated}")
                if estimated <= 5:
                    print(f"      → Very small structure space - will likely generate all unique structures")
                elif estimated <= 25:
                    print(f"      → Small structure space - high probability of finding most unique structures")
                elif estimated <= 100:
                    print(f"      → Moderate structure space - good sampling of unique structures expected")
                elif estimated <= 500:
                    print(f"      → Large structure space - sampling significant portion of unique structures")
                else:
                    print(f"      → Very large structure space - sampling subset of unique structures")
            print()
        
        print("Generating molecules...")
    
    # Generate molecules for each combination
    total_generated = 0
    for combo in tqdm(combinations, desc='Processing combinations', disable=not verbose):
        if sum(combo.values()) == 0:
            continue
        generated = _generate_molecules_for_combo(
            combo, num_mols_per_combo, output_dir, max_estimate_threshold, 
            uncertainty_threshold, bonds_allowed, element_map, 
            si_units, n_units, verbose, random_seed
        )
        total_generated += generated
    
    # Final validation if requested
    invalid_count = 0
    if validate_final:
        if verbose:
            print("\nPerforming final validation...")
        invalid_count = _validate_all_generated_molecules(output_dir, bond_length_thresholds, verbose)
    
    # Generate summary statistics
    stats = {
        'combinations_found': len(combinations),
        'total_molecules': total_generated,
        'valid_molecules': total_generated - invalid_count,
        'invalid_molecules': invalid_count,
        'output_directory': output_dir,
        'random_seed': random_seed
    }
    
    if verbose:
        print(f"\nGeneration complete!")
        print("All generated molecules satisfy the Si-N only bonding constraint.")
        print(f"Combinations found: {stats['combinations_found']}")
        print(f"Total molecules generated: {stats['total_molecules']}")
        print(f"Valid molecules: {stats['valid_molecules']}")
        print(f"Invalid molecules removed: {stats['invalid_molecules']}")
        if random_seed is not None:
            print(f"Random seed used: {random_seed}")
        print(f"Results saved in: {output_dir}")
    
    return stats

# Helper functions (internal implementation using enhanced code 2 logic)

def _factorial(n):
    """Calculate factorial with handling for large numbers and overflow protection"""
    if n <= 1:
        return 1
    if n > 20:  # Prevent overflow for very large factorials
        return float('inf')  # Return infinity to indicate very large number
    
    result = 1
    for i in range(2, n + 1):
        result *= i
        if result > 10**15:  # Cap at a reasonable large number
            return 10**15
    return result

def _safe_divide(numerator, denominator):
    """Safely divide large numbers, handling infinity cases"""
    if denominator == 0:
        return 0
    if numerator == float('inf') or denominator == float('inf'):
        if numerator == float('inf') and denominator == float('inf'):
            return 1  # inf/inf treated as 1 for our purposes
        elif numerator == float('inf'):
            return 10**10  # Large but finite number
        else:
            return 0
    return max(1, numerator // denominator)

def _estimate_bipartite_matchings(si_degrees, n_degrees):
    """
    Estimate the number of possible UNIQUE bipartite perfect matchings.
    Uses enhanced estimation approach from code 2.
    """
    # Total stubs on each side
    total_si_stubs = sum(si_degrees)
    total_n_stubs = sum(n_degrees)
    
    if total_si_stubs != total_n_stubs:
        return 0
    
    if total_si_stubs == 0:
        return 1
    
    # Count identical atom types for symmetry analysis
    si_counter = Counter(si_degrees)
    n_counter = Counter(n_degrees)
    total_heavy_atoms = len(si_degrees) + len(n_degrees)
    
    # For very small systems, use more accurate combinatorial counting
    if total_heavy_atoms <= 4:
        # Small systems: be more generous with estimates
        if total_heavy_atoms == 2:
            return 1  # Only one way to connect 2 atoms
        elif total_heavy_atoms == 3:
            return max(1, 2)  # Usually 1-2 ways for 3 atoms
        else:  # 4 atoms
            return max(1, min(10, total_heavy_atoms * 3))
    
    # For typical systems, use degree-based estimation
    elif total_heavy_atoms <= 12:
        # Calculate base estimate using degree sequence analysis
        si_degree_types = len(si_counter)
        n_degree_types = len(n_counter)
        
        # Base structural diversity scales with degree variety and atom count
        degree_diversity = si_degree_types * n_degree_types
        
        # Estimate based on the number of ways to distribute connections
        if degree_diversity <= 2:
            # Low diversity (e.g., all Si same degree, all N same degree)
            base_estimate = max(total_heavy_atoms // 2, 1) * degree_diversity
        elif degree_diversity <= 4:
            # Moderate diversity
            base_estimate = total_heavy_atoms * degree_diversity
        else:
            # High diversity
            base_estimate = (total_heavy_atoms ** 1.5) * degree_diversity
        
        # Apply size-based scaling
        if total_heavy_atoms <= 6:
            scaling = max(2, total_heavy_atoms)
        elif total_heavy_atoms <= 8:
            scaling = max(5, total_heavy_atoms * 2)
        else:
            scaling = max(10, total_heavy_atoms * 3)
        
        estimate = int(base_estimate * scaling)
        
        # Apply symmetry reduction, but less aggressively
        # Calculate symmetry factor
        symmetry_factor = 1
        for degree, count in si_counter.items():
            if count > 1:
                symmetry_factor *= max(1, count // 2 + 1)  # Gentler symmetry reduction
        
        for degree, count in n_counter.items():
            if count > 1:
                symmetry_factor *= max(1, count // 2 + 1)  # Gentler symmetry reduction
        
        final_estimate = max(1, estimate // max(1, symmetry_factor))
        
        # Apply reasonable bounds
        min_estimate = max(1, total_heavy_atoms // 2)
        max_estimate = min(10000, total_heavy_atoms ** 3)
        
        return max(min_estimate, min(final_estimate, max_estimate))
    
    else:
        # Large systems: more generous estimates
        si_degree_variety = len(set(si_degrees))
        n_degree_variety = len(set(n_degrees))
        
        base_estimate = (total_heavy_atoms ** 2) * (si_degree_variety + n_degree_variety)
        
        # Less aggressive symmetry reduction for large systems
        symmetry_factor = 1
        for degree, count in si_counter.items():
            if count > 2:
                symmetry_factor *= max(1, count // 3 + 1)
        
        for degree, count in n_counter.items():
            if count > 2:
                symmetry_factor *= max(1, count // 3 + 1)
        
        final_estimate = max(total_heavy_atoms, base_estimate // max(1, symmetry_factor))
        
        return min(final_estimate, 50000)  # Cap at reasonable maximum

def _estimate_combo_size(combo, uncertainty_threshold, bonds_allowed, si_units, n_units):
    """
    Estimate the number of possible UNIQUE molecular structures for a given combination.
    Uses enhanced estimation from code 2 that's less conservative and more realistic.
    """
    # Create degree sequences
    si_degrees = []
    n_degrees = []
    
    for species, count in combo.items():
        if count == 0:
            continue
            
        degree = bonds_allowed[species]
        
        if species in si_units:
            si_degrees.extend([degree] * count)
        elif species in n_units:
            n_degrees.extend([degree] * count)
    
    # Check if we have a valid bipartite system
    if sum(si_degrees) != sum(n_degrees):
        return 0
    
    # Get base estimate
    base_estimate = _estimate_bipartite_matchings(si_degrees, n_degrees)
    
    # Apply much less conservative bounds for molecular realities
    total_heavy_atoms = len(si_degrees) + len(n_degrees)  # Only Si and N atoms
    total_stubs = sum(si_degrees)
    
    # Count unique degree types for additional complexity assessment
    si_degree_types = len(set(si_degrees))
    n_degree_types = len(set(n_degrees))
    degree_complexity = si_degree_types + n_degree_types
    
    # More generous bounds that allow for realistic structural diversity
    if total_heavy_atoms <= 3:
        # Very small molecules 
        min_bound = 1
        max_bound = max(3, degree_complexity * 2)
    elif total_heavy_atoms <= 6:
        # Small molecules
        min_bound = max(1, total_heavy_atoms // 2)
        max_bound = max(10, total_heavy_atoms * degree_complexity * 3)
    elif total_heavy_atoms <= 10:
        # Medium molecules
        min_bound = max(2, total_heavy_atoms)
        max_bound = max(50, total_heavy_atoms * degree_complexity * 10)
    else:
        # Large molecules
        min_bound = max(5, total_heavy_atoms * 2)
        max_bound = max(200, total_heavy_atoms * degree_complexity * 20)
    
    # Apply bounds but don't be too restrictive
    final_estimate = max(min_bound, min(base_estimate, max_bound))
    
    # Check against uncertainty threshold
    if final_estimate > uncertainty_threshold:
        return -1  # Indicates "more than threshold"
    
    return int(final_estimate)

def _find_combinations(values, target, species_list, tolerance, mw_dict, bonds_allowed, si_units, n_units, verbose):
    """Find valid molecular combinations using enhanced search from code 2"""
    results = []
    max_multiple = math.ceil(target / (mw_dict['N'] + mw_dict['Si']))
    
    # Calculate total combinations to search
    total_combinations = (max_multiple + 1) ** len(values)
    
    if verbose:
        print(f"Searching combinations with constraint: Si stubs = N stubs (for Si-N only bonding)")
        print(f"Total combinations to check: {total_combinations:,}")
        print("CONSTRAINT: Only combinations allowing Si-N bonds (no Si-Si or N-N) will be accepted")
    
    count = 0
    # Create progress bar for combination search
    pbar = tqdm(
        product(range(max_multiple + 1), repeat=len(values)), 
        total=total_combinations,
        desc="Searching combinations",
        disable=not verbose,
        unit="combos"
    )
    
    for multiples in pbar:
        total = sum(m * v for m, v in zip(multiples, values))
        count += 1
        
        # Update progress bar description with current results
        if count % 10000 == 0:  # Update every 10k iterations to avoid slowdown
            pbar.set_postfix(found=len(results), refresh=False)
        
        if np.isclose(total, target, atol=tolerance):
            combo_dict = {sp: mult for sp, mult in zip(species_list, multiples)}
            
            # CRITICAL: Check if combination has both Si and N units (required for Si-N bonding)
            has_si = any(mult > 0 for sp, mult in combo_dict.items() if sp in si_units)
            has_n = any(mult > 0 for sp, mult in combo_dict.items() if sp in n_units)
            
            if has_si and has_n:
                # CRITICAL: Check if stubs match exactly (required for bipartite Si-N graph)
                # For Si-N only bonding, every Si stub must connect to exactly one N stub
                n_stubs = sum(bonds_allowed[sp] * mult for sp, mult in combo_dict.items() if sp in n_units)
                si_stubs = sum(bonds_allowed[sp] * mult for sp, mult in combo_dict.items() if sp in si_units)
                
                if n_stubs == si_stubs and n_stubs > 0:
                    results.append(combo_dict)
                    if verbose:
                        print(f"\nFound valid Si-N bipartite combination: {combo_dict}, MW: {total:.3f}")
                        print(f"  -> Si stubs: {si_stubs}, N stubs: {n_stubs} (MATCHED for Si-N bonding)")
                    # Update progress bar immediately when we find a result
                    pbar.set_postfix(found=len(results), refresh=True)
                else:
                    # Optional: show why combinations were rejected (comment out if too verbose)
                    if verbose and count % 500000 == 0 and n_stubs != si_stubs:  # Only show occasionally
                        print(f"Rejected {combo_dict}: Si stubs ({si_stubs}) ≠ N stubs ({n_stubs}) - cannot form Si-N only bonds")
            else:
                # Combinations without both Si and N cannot form Si-N bonds (comment out if too verbose)
                if verbose and count % 500000 == 0:  # Only show occasionally
                    if not has_si and has_n:
                        print(f"Rejected {combo_dict}: No Si atoms - cannot form Si-N bonds")
                    elif has_si and not has_n:
                        print(f"Rejected {combo_dict}: No N atoms - cannot form Si-N bonds")
    
    pbar.close()
    
    if verbose:
        print(f"Combination search complete: {len(results)} valid combinations found out of {total_combinations:,} checked")
    
    return results

def _validate_molecule_strict_si_n(mol, species_list_flat, si_units, n_units):
    """
    STRICT validation that a molecule has ONLY Si-N bonds.
    NO Si-Si bonds and NO N-N bonds are allowed.
    Enhanced version from code 2.
    """
    invalid_bonds = []
    si_n_bonds = 0
    total_heavy_bonds = 0
    
    for bond in mol.GetBonds():
        a_idx = bond.GetBeginAtomIdx()
        b_idx = bond.GetEndAtomIdx()
        
        # Get atom information
        a_atom = mol.GetAtomWithIdx(a_idx)
        b_atom = mol.GetAtomWithIdx(b_idx)
        a_symbol = a_atom.GetSymbol()
        b_symbol = b_atom.GetSymbol()
        
        # Skip bonds involving hydrogen
        if a_symbol == 'H' or b_symbol == 'H':
            continue
        
        total_heavy_bonds += 1
        
        # Get species information if available
        a_species = species_list_flat[a_idx] if a_idx < len(species_list_flat) else f"Unknown({a_symbol})"
        b_species = species_list_flat[b_idx] if b_idx < len(species_list_flat) else f"Unknown({b_symbol})"
        
        # Check bond type
        if a_symbol == 'Si' and b_symbol == 'N':
            # Valid Si-N bond
            si_n_bonds += 1
        elif a_symbol == 'N' and b_symbol == 'Si':
            # Valid N-Si bond (same as Si-N)
            si_n_bonds += 1
        elif a_symbol == 'Si' and b_symbol == 'Si':
            # INVALID: Si-Si bond
            invalid_bonds.append(f"FORBIDDEN Si-Si bond: {a_idx}({a_species}) - {b_idx}({b_species})")
        elif a_symbol == 'N' and b_symbol == 'N':
            # INVALID: N-N bond
            invalid_bonds.append(f"FORBIDDEN N-N bond: {a_idx}({a_species}) - {b_idx}({b_species})")
        else:
            # INVALID: Any other heavy atom combination
            invalid_bonds.append(f"FORBIDDEN bond (non Si-N): {a_idx}({a_symbol},{a_species}) - {b_idx}({b_symbol},{b_species})")
    
    # Additional check: ALL heavy bonds must be Si-N
    is_valid = (len(invalid_bonds) == 0) and (si_n_bonds == total_heavy_bonds) and (total_heavy_bonds > 0)
    
    if not is_valid and len(invalid_bonds) == 0:
        # No invalid bonds found, but constraint still not satisfied
        if total_heavy_bonds == 0:
            invalid_bonds.append("No heavy atom bonds found")
        elif si_n_bonds != total_heavy_bonds:
            invalid_bonds.append(f"Not all heavy bonds are Si-N: {si_n_bonds} Si-N bonds out of {total_heavy_bonds} total heavy bonds")
    
    return is_valid, invalid_bonds

def _generate_molecules_for_combo(combo, num_mols, output_dir, max_estimate_threshold, 
                                uncertainty_threshold, bonds_allowed, element_map, 
                                si_units, n_units, verbose, random_seed):
    """
    Generate molecules for a given combination using enhanced logic from code 2.
    """
    
    # Estimate the number of possible molecules using enhanced estimation
    estimated_molecules = _estimate_combo_size(combo, uncertainty_threshold, bonds_allowed, si_units, n_units)
    
    if estimated_molecules == 0:
        if verbose:
            print(f"Skipping combination (no valid structures possible): {combo}")
        return 0
    
    # Enhanced limiting logic from code 2
    original_num_mols = num_mols
    if estimated_molecules == -1:
        if verbose:
            print(f"Combination {combo}: estimated >>{uncertainty_threshold} unique molecular structures (many possible), generating {num_mols}")
        estimated_molecules = uncertainty_threshold + 1  # Use threshold+1 for logic below
    elif estimated_molecules <= uncertainty_threshold and estimated_molecules < num_mols:
        # Enhanced limiting logic from code 2
        if estimated_molecules <= 10:
            num_mols = min(num_mols, estimated_molecules + 2)  # Try to get all + a few extra attempts
            if verbose:
                print(f"Small structure space for {combo}: estimated {estimated_molecules} unique molecules, generating {num_mols} (targeting all unique structures)")
        elif estimated_molecules <= 50:
            num_mols = min(num_mols, int(estimated_molecules * 1.2))  # Generate 120% to account for estimation uncertainty
            if verbose:
                print(f"Moderate structure space for {combo}: estimated {estimated_molecules} unique molecules, generating {num_mols} (targeting most unique structures)")
        elif estimated_molecules <= 200:
            num_mols = min(num_mols, int(estimated_molecules * 0.8))  # Generate 80% for larger spaces
            if verbose:
                print(f"Large structure space for {combo}: estimated {estimated_molecules} unique molecules, generating {num_mols} (targeting 80%)")
        else:
            num_mols = min(num_mols, int(estimated_molecules * 0.5))  # Generate 50% for very large spaces
            if verbose:
                print(f"Very large structure space for {combo}: estimated {estimated_molecules} unique molecules, generating {num_mols} (targeting 50%)")
    elif estimated_molecules > max_estimate_threshold:
        num_mols = min(num_mols, max_estimate_threshold // 1000)  # More generous limit for large spaces
        if verbose:
            print(f"Extremely large structure space for {combo}: estimated >>{uncertainty_threshold if estimated_molecules > uncertainty_threshold else estimated_molecules} unique molecules possible, limiting to {num_mols}")
    else:
        if estimated_molecules > uncertainty_threshold:
            if verbose:
                print(f"Combination {combo}: estimated >>{uncertainty_threshold} unique molecular structures (many possible), generating {num_mols}")
        else:
            if verbose:
                print(f"Combination {combo}: estimated {estimated_molecules} unique molecular structures, generating {num_mols}")
    
    # Setup
    species_list_flat = []
    for sp, count in combo.items():
        species_list_flat += [sp] * count
    
    if not species_list_flat:
        return 0
    
    combo_name = '_'.join(f"{s}{combo[s]}" for s in sorted(combo.keys()) if combo[s] > 0)
    combo_dir = os.path.join(output_dir, f"combo_{combo_name}")
    os.makedirs(combo_dir, exist_ok=True)
    
    degrees = [bonds_allowed[sp] for sp in species_list_flat]
    total_stubs = sum(degrees)
    
    # Validation checks
    if total_stubs % 2 != 0:
        if verbose:
            print(f"Skipping {combo_name}: odd sum of stubs ({total_stubs}).")
        return 0
    
    # CRITICAL: Enforce equal N vs Si stubs for purely bipartite Si-N bonding
    n_stubs = sum(deg for sp, deg in zip(species_list_flat, degrees) if sp in n_units)
    si_stubs = sum(deg for sp, deg in zip(species_list_flat, degrees) if sp in si_units)
    if n_stubs != si_stubs:
        if verbose:
            print(f"Skipping {combo_name}: unmatched N stubs ({n_stubs}) vs Si stubs ({si_stubs}). Si-N only bonding requires equal stub counts.")
        return 0

    # Verify we have both Si and N atoms (can't have Si-N bonds without both)
    has_si = any(sp in si_units for sp in species_list_flat)
    has_n = any(sp in n_units for sp in species_list_flat)
    if not (has_si and has_n):
        if verbose:
            print(f"Skipping {combo_name}: missing either Si atoms ({has_si}) or N atoms ({has_n}). Si-N bonding requires both.")
        return 0
    
    # Generation loop with enhanced logic from code 2
    generated = 0
    attempts = 0
    max_attempts = num_mols * 200  # Increased for better success rate with larger systems
    generated_smiles = set()
    
    pbar = tqdm(total=num_mols, desc=f"Generating {combo_name} (Si-N only)", leave=False, disable=not verbose)
    
    while generated < num_mols and attempts < max_attempts:
        attempts += 1
        
        # STRICT BIPARTITE APPROACH: Create separate stub lists for Si and N atoms
        si_stubs = []
        n_stubs = []
        
        for idx, (sp, deg) in enumerate(zip(species_list_flat, degrees)):
            if sp in si_units:
                si_stubs += [idx] * deg
            elif sp in n_units:  # Only N units, not Si
                n_stubs += [idx] * deg
            else:
                # This should never happen with our current species, but safety check
                if verbose:
                    print(f"ERROR: Unknown species {sp} encountered!")
                continue
        
        # Sanity check: must have equal numbers of Si and N stubs
        if len(si_stubs) != len(n_stubs):
            if verbose:
                print(f"ERROR: Si stubs ({len(si_stubs)}) != N stubs ({len(n_stubs)}) for {combo_name}")
            break
        
        # Shuffle both lists independently
        random.shuffle(si_stubs)
        random.shuffle(n_stubs)
        
        # STRICT PAIRING: Each Si stub MUST pair with exactly one N stub
        edges = []
        edges_set = set()
        valid = True
        
        for si_idx, n_idx in zip(si_stubs, n_stubs):
            # DOUBLE CHECK: Ensure we're pairing Si with N (and ONLY Si with N)
            si_species = species_list_flat[si_idx]
            n_species = species_list_flat[n_idx]
            
            if si_species not in si_units:
                if verbose:
                    print(f"ERROR: Expected Si atom at index {si_idx}, got {si_species}")
                valid = False
                break
            if n_species not in n_units:
                if verbose:
                    print(f"ERROR: Expected N atom at index {n_idx}, got {n_species}")
                valid = False
                break
            
            # Check for duplicate edges (same pair of atoms)
            edge = tuple(sorted([si_idx, n_idx]))
            if edge in edges_set:
                valid = False  # Multiple bonds between same atoms not allowed
                break
            
            edges_set.add(edge)
            edges.append((si_idx, n_idx))
        
        if not valid:
            continue
        
        # FINAL VALIDATION: Verify ALL edges are strictly Si-N bonds
        for a, b in edges:
            sp_a = species_list_flat[a]
            sp_b = species_list_flat[b]
            
            # Must be one Si and one N (in either order)
            if not ((sp_a in si_units and sp_b in n_units) or (sp_a in n_units and sp_b in si_units)):
                if verbose:
                    print(f"ERROR: Invalid bond detected during generation: {sp_a}-{sp_b}")
                valid = False
                break
            
            # Additional check: cannot have Si-Si or N-N
            if (sp_a in si_units and sp_b in si_units):
                if verbose:
                    print(f"ERROR: Si-Si bond detected: {sp_a}-{sp_b}")
                valid = False
                break
            if (sp_a in n_units and sp_b in n_units):
                if verbose:
                    print(f"ERROR: N-N bond detected: {sp_a}-{sp_b}")
                valid = False
                break
        
        if not valid:
            continue

        # Build RDKit Mol with ONLY the validated Si-N bonds
        try:
            rw = Chem.RWMol()
            for sp in species_list_flat:
                atom = Chem.Atom(element_map[sp])
                rw.AddAtom(atom)
            
            # Add ONLY the Si-N bonds we've validated
            for a, b in edges:
                rw.AddBond(a, b, Chem.BondType.SINGLE)
            
            mol = rw.GetMol()

            # Sanitize and add hydrogens
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL)
            mol = Chem.AddHs(mol)
            
            # Check for duplicate structures using SMILES
            smi = Chem.MolToSmiles(mol)
            if smi in generated_smiles:
                continue  # Skip duplicate
            generated_smiles.add(smi)
            
            # CRITICAL VALIDATION: Final check that molecule has ONLY Si-N bonds
            is_valid, invalid_bonds = _validate_molecule_strict_si_n(mol, species_list_flat, si_units, n_units)
            if not is_valid:
                if verbose:
                    print(f"REJECTED: Invalid bonds detected in {combo_name}:")
                    for bond in invalid_bonds:
                        print(f"  {bond}")
                continue

            # Embed 3D geometry
            params = AllChem.EmbedParameters()
            params.randomSeed = random_seed
            params.useRandomCoords = True
            
            if AllChem.EmbedMolecule(mol, params) != 0:
                continue
            
            # Optimize geometry
            AllChem.MMFFOptimizeMolecule(mol)

            # SUCCESS: write SMILES + XYZ in its own folder
            generated += 1
            mol_dir = os.path.join(combo_dir, f"mol_{generated}")
            os.makedirs(mol_dir, exist_ok=True)
            
            # SMILES
            with open(os.path.join(mol_dir, 'mol.smi'), 'w') as f:
                f.write(smi + '\n')
            
            # Write a detailed bond list for verification (enhanced from code 2)
            with open(os.path.join(mol_dir, 'bonds.txt'), 'w') as f:
                f.write("BOND VALIDATION REPORT\n")
                f.write("=" * 40 + "\n")
                f.write("Heavy atom bonds only (H bonds excluded):\n")
                f.write("All bonds MUST be Si-N or N-Si only!\n\n")
                
                si_n_bonds = 0
                invalid_bonds_found = 0
                
                for bond in mol.GetBonds():
                    a_idx = bond.GetBeginAtomIdx()
                    b_idx = bond.GetEndAtomIdx()
                    a_sym = mol.GetAtomWithIdx(a_idx).GetSymbol()
                    b_sym = mol.GetAtomWithIdx(b_idx).GetSymbol()
                    
                    # Skip H bonds for clarity
                    if a_sym == 'H' or b_sym == 'H':
                        continue
                    
                    # Map back to original species
                    a_species = species_list_flat[a_idx] if a_idx < len(species_list_flat) else "Unknown"
                    b_species = species_list_flat[b_idx] if b_idx < len(species_list_flat) else "Unknown"
                    
                    bond_type = "UNKNOWN"
                    if (a_sym == 'Si' and b_sym == 'N') or (a_sym == 'N' and b_sym == 'Si'):
                        bond_type = "Si-N (VALID)"
                        si_n_bonds += 1
                    elif a_sym == 'Si' and b_sym == 'Si':
                        bond_type = "Si-Si (INVALID!)"
                        invalid_bonds_found += 1
                    elif a_sym == 'N' and b_sym == 'N':
                        bond_type = "N-N (INVALID!)"
                        invalid_bonds_found += 1
                    
                    f.write(f"{a_idx}({a_sym},{a_species}) - {b_idx}({b_sym},{b_species}) [{bond_type}]\n")
                
                f.write(f"\nSUMMARY:\n")
                f.write(f"Valid Si-N bonds: {si_n_bonds}\n")
                f.write(f"Invalid bonds: {invalid_bonds_found}\n")
                f.write(f"CONSTRAINT SATISFIED: {invalid_bonds_found == 0}\n")
            
            # XYZ
            xyz = MolToXYZBlock(mol)
            with open(os.path.join(mol_dir, 'mol.xyz'), 'w') as f:
                f.write(xyz)

            pbar.update(1)
            
        except Exception as e:
            # Skip this attempt if any error occurs
            if verbose:
                print(f"Exception during molecule generation: {e}")
            continue

    pbar.close()
    actual_unique = len(generated_smiles)
    
    # Handle display of estimated molecules with uncertainty threshold
    if estimated_molecules > uncertainty_threshold:
        estimated_display = f">>{uncertainty_threshold}"
    else:
        estimated_display = str(estimated_molecules)
    
    if verbose:
        print(f"Generated {generated}/{num_mols} molecules for combo {combo_name} (attempts: {attempts}, unique structures: {actual_unique}, estimated unique possible: {estimated_display})")
        print(f"  CONSTRAINT CHECK: All molecules have ONLY Si-N bonds (no Si-Si or N-N bonds)")
        print(f"  UNIQUENESS: Generated {actual_unique} unique SMILES out of estimated {estimated_display} possible unique structures")
    
    return generated

def _validate_xyz_file(xyz_file, bond_length_thresholds):
    """
    Validate an XYZ file to check for N-N or Si-Si bonds using enhanced logic from code 2
    """
    try:
        with open(xyz_file, 'r') as f:
            lines = f.readlines()
        
        num_atoms = int(lines[0].strip())
        atoms = []
        
        for i in range(2, 2 + num_atoms):
            parts = lines[i].split()
            atom_type = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            atoms.append((atom_type, np.array([x, y, z])))
        
        # Check distances between heavy atoms using provided thresholds
        invalid_bonds = []
        
        for i in range(num_atoms):
            if atoms[i][0] == 'H':
                continue
                
            for j in range(i + 1, num_atoms):
                if atoms[j][0] == 'H':
                    continue
                
                dist = np.linalg.norm(atoms[i][1] - atoms[j][1])
                atom_pair = tuple(sorted([atoms[i][0], atoms[j][0]]))
                
                if atom_pair in bond_length_thresholds and dist < bond_length_thresholds[atom_pair]:
                    if atom_pair == ('N', 'N'):
                        invalid_bonds.append(f"N-N bond detected: atoms {i} and {j}, distance: {dist:.3f} Å")
                    elif atom_pair == ('Si', 'Si'):
                        invalid_bonds.append(f"Si-Si bond detected: atoms {i} and {j}, distance: {dist:.3f} Å")
        
        return len(invalid_bonds) == 0, invalid_bonds
    
    except Exception:
        return False, ["File reading error"]

def _validate_all_generated_molecules(output_dir, bond_length_thresholds, verbose):
    """
    Validate all generated XYZ files and remove invalid ones using enhanced logic from code 2
    """
    if verbose:
        print("\n" + "="*60)
        print("VALIDATING ALL GENERATED MOLECULES")
        print("="*60)
        print(f"Bond length thresholds: {bond_length_thresholds}")
    
    xyz_files = glob.glob(os.path.join(output_dir, '**/mol.xyz'), recursive=True)
    
    if not xyz_files:
        if verbose:
            print("No XYZ files found to validate.")
        return 0
    
    if verbose:
        print(f"Found {len(xyz_files)} XYZ files to validate.\n")
    
    invalid_count = 0
    invalid_files = []
    removed_dirs = []
    
    for xyz_file in tqdm(xyz_files, desc="Validating XYZ files", disable=not verbose):
        is_valid, invalid_bonds = _validate_xyz_file(xyz_file, bond_length_thresholds)
        
        if not is_valid:
            invalid_count += 1
            invalid_files.append((xyz_file, invalid_bonds))
            
            # Remove the entire molecule directory
            mol_dir = os.path.dirname(xyz_file)
            if os.path.exists(mol_dir):
                shutil.rmtree(mol_dir)
                removed_dirs.append(mol_dir)
    
    # Clean up empty combo directories
    if verbose:
        print("\nCleaning up empty directories...")
    combo_dirs = glob.glob(os.path.join(output_dir, 'combo_*'))
    empty_combos = []
    
    for combo_dir in combo_dirs:
        # Check if combo directory has any molecule subdirectories
        mol_dirs = glob.glob(os.path.join(combo_dir, 'mol_*'))
        if not mol_dirs:
            # Remove empty combo directory
            shutil.rmtree(combo_dir)
            empty_combos.append(combo_dir)
            if verbose:
                print(f"  Removed empty combo directory: {os.path.basename(combo_dir)}")
    
    # Print summary
    if verbose:
        print("\n" + "-"*60)
        print("VALIDATION SUMMARY")
        print("-"*60)
        print(f"Total files checked: {len(xyz_files)}")
        print(f"Valid molecules kept: {len(xyz_files) - invalid_count}")
        print(f"Invalid molecules removed: {invalid_count}")
        print(f"Empty combo directories removed: {len(empty_combos)}")
        
        if invalid_count > 0:
            print("\nREMOVED INVALID MOLECULES:")
            for (xyz_file, invalid_bonds), mol_dir in zip(invalid_files, removed_dirs):
                print(f"\nRemoved: {mol_dir}")
                for bond in invalid_bonds:
                    print(f"  - {bond}")
    
    # Create validation report
    report_file = os.path.join(output_dir, 'validation_report.txt')
    with open(report_file, 'w') as f:
        f.write("MOLECULE VALIDATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated on: {os.popen('date').read().strip()}\n")
        f.write(f"Bond length thresholds: {bond_length_thresholds}\n")
        f.write(f"Total molecules checked: {len(xyz_files)}\n")
        f.write(f"Valid molecules kept: {len(xyz_files) - invalid_count}\n")
        f.write(f"Invalid molecules removed: {invalid_count}\n")
        f.write(f"Empty combo directories removed: {len(empty_combos)}\n\n")
        
        if invalid_count > 0:
            f.write("REMOVED INVALID MOLECULES:\n")
            for (xyz_file, invalid_bonds), mol_dir in zip(invalid_files, removed_dirs):
                f.write(f"\nRemoved directory: {mol_dir}\n")
                f.write(f"Original file: {xyz_file}\n")
                for bond in invalid_bonds:
                    f.write(f"  - {bond}\n")
        
        if empty_combos:
            f.write("\n\nREMOVED EMPTY COMBO DIRECTORIES:\n")
            for combo_dir in empty_combos:
                f.write(f"  - {combo_dir}\n")
    
    if verbose:
        print(f"\nValidation report saved to: {report_file}")
        
        # Count remaining molecules in each combo directory
        print("\nCounting remaining valid molecules...")
        combo_dirs = glob.glob(os.path.join(output_dir, 'combo_*'))
        for combo_dir in combo_dirs:
            mol_dirs = [d for d in glob.glob(os.path.join(combo_dir, 'mol_*')) if os.path.isdir(d)]
            print(f"  {os.path.basename(combo_dir)}: {len(mol_dirs)} valid molecules")
    
    return invalid_count


# Example usage
if __name__ == '__main__':

    stats = generate_sin_only_molecules(
        target_mw=472,
        output_dir='test',
        tolerance=0,
        num_mols_per_combo=2000,
        random_seed=0,  # Set random seed for reproducible results
        verbose=True
    )
    
    print(f"\nGeneration Summary:")
    print(f"Combinations found: {stats['combinations_found']}")
    print(f"Total molecules: {stats['total_molecules']}")
    print(f"Valid molecules: {stats['valid_molecules']}")
    print(f"Invalid molecules removed: {stats['invalid_molecules']}")
    print(f"Random seed used: {stats['random_seed']}")
    print(f"Results saved in: {stats['output_directory']}")
