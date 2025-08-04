import os
import time
import argparse
import yaml
from typing import Dict, Any, Optional
from ..generation import generate_sin_only_molecules
from ..processing import remove_duplicates, reorganize_output
from ..optimization import optimize_folder_xyz

class MolecularWorkflowPipeline:
    def __init__(self, config_path: Optional[str] = None):
        self.config = {'generation': {'tolerance': 2.0, 'num_mols_per_combo': 1000, 'validate_final': True, 'random_seed': None, 'verbose': True}, 'processing': {'dry_run': False}}
    
    def run_generation(self, target_mw: float, output_dir: str, **kwargs) -> Dict[str, Any]:
        print("="*60 + "\nSTEP 1: MOLECULAR GENERATION\n" + "="*60)
        gen_params = self.config['generation'].copy()
        gen_params.update(kwargs)
        start_time = time.time()
        stats = generate_sin_only_molecules(target_mw=target_mw, output_dir=output_dir, **gen_params)
        stats['generation_time'] = time.time() - start_time
        print(f"\n✅ Generation completed in {stats['generation_time']:.1f} seconds")
        return stats
    
    def run_deduplication(self, output_dir: str, **kwargs) -> Dict[str, Any]:
        print("\n" + "="*60 + "\nSTEP 2: DEDUPLICATION\n" + "="*60)
        start_time = time.time()
        dup_stats = remove_duplicates(output_dir, dry_run=kwargs.get('dry_run', False))
        dedup_time = time.time() - start_time
        total_removed = sum(len(folders) for folders in dup_stats.values())
        print(f"\n✅ Deduplication completed in {dedup_time:.1f} seconds")
        return {'deduplication_time': dedup_time, 'total_molecules_removed': total_removed}
    
    def run_reorganization(self, output_dir: str) -> Dict[str, Any]:
        print("\n" + "="*60 + "\nSTEP 3: REORGANIZATION\n" + "="*60)
        start_time = time.time()
        reorganize_output(output_dir)
        reorg_time = time.time() - start_time
        print(f"\n✅ Reorganization completed in {reorg_time:.1f} seconds")
        return {'reorganization_time': reorg_time}
    
    def run_optimization(self, output_dir: str, model_path: str, **kwargs) -> Dict[str, Any]:
        print("\n" + "="*60 + "\nSTEP 4: ENERGY OPTIMIZATION\n" + "="*60)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        start_time = time.time()
        optimize_folder_xyz(output_dir, model_path, **kwargs)
        opt_time = time.time() - start_time
        print(f"\n✅ Optimization completed in {opt_time:.1f} seconds")
        return {'optimization_time': opt_time}
    
    def run_complete_workflow(self, target_mw: float, output_dir: str, model_path: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        print("🚀 STARTING COMPLETE MOLECULAR WORKFLOW\n" + "="*80)
        total_start_time = time.time()
        
        gen_stats = self.run_generation(target_mw, output_dir, **kwargs)
        dup_stats = self.run_deduplication(output_dir, **kwargs)
        reorg_stats = self.run_reorganization(output_dir)
        
        opt_stats = None
        if model_path:
            opt_stats = self.run_optimization(output_dir, model_path, **kwargs)
        
        total_time = time.time() - total_start_time
        final_stats = {'target_mw': target_mw, 'total_workflow_time': total_time, 'generation': gen_stats, 'deduplication': dup_stats, 'reorganization': reorg_stats, 'optimization': opt_stats}
        
        print(f"\n🎉 WORKFLOW COMPLETED in {total_time:.1f} seconds!")
        return final_stats

def main():
    parser = argparse.ArgumentParser(description="Molecular Workflow")
    subparsers = parser.add_subparsers(dest='command')
    run_parser = subparsers.add_parser('run', help='Run complete workflow')
    run_parser.add_argument('--target-mw', type=float, required=True)
    run_parser.add_argument('--output-dir', type=str, required=True)
    run_parser.add_argument('--model-path', type=str, help='ESEN model path')
    run_parser.add_argument('--num-mols', type=int, default=1000)
    run_parser.add_argument('--tolerance', type=float, default=2.0)
    run_parser.add_argument('--random-seed', type=int)
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return
    
    pipeline = MolecularWorkflowPipeline()
    if args.command == 'run':
        kwargs = {}
        if hasattr(args, 'num_mols'): kwargs['num_mols_per_combo'] = args.num_mols
        if hasattr(args, 'tolerance'): kwargs['tolerance'] = args.tolerance
        if hasattr(args, 'random_seed') and args.random_seed is not None: kwargs['random_seed'] = args.random_seed
        pipeline.run_complete_workflow(args.target_mw, args.output_dir, getattr(args, 'model_path', None), **kwargs)

if __name__ == '__main__': main()
