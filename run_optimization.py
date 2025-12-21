#!/usr/bin/env python3
"""
Quick runner for basal ganglia network optimization.

Usage:
    python run_optimization.py --trials 100 --name test_run
    python run_optimization.py --trials 1000 --name production --gpu

Author: Kavin Nakkeeran
Date: December 2025
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optimization.optuna_driver import run_optimization


def main():
    parser = argparse.ArgumentParser(
        description='Run JAX-Optuna parameter optimization for basal ganglia network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test (10 trials)
  python run_optimization.py --trials 10 --name quick_test
  
  # Medium run (100 trials)
  python run_optimization.py --trials 100 --name medium_run
  
  # Full optimization (1000 trials)
  python run_optimization.py --trials 1000 --name full_optimization
  
  # GPU-accelerated
  python run_optimization.py --trials 5000 --name gpu_run --gpu
        """
    )
    
    parser.add_argument(
        '--trials', 
        type=int, 
        default=100,
        help='Number of Optuna trials to run (default: 100)'
    )
    
    parser.add_argument(
        '--name', 
        type=str, 
        default='optimization',
        help='Study name for Optuna (default: optimization)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Enable GPU acceleration (requires jax[cuda])'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Check GPU
    if args.gpu:
        try:
            import jax
            devices = jax.devices()
            if len(devices) > 0 and 'gpu' in str(devices[0]).lower():
                print(f"✓ GPU detected: {devices[0]}")
            else:
                print("⚠ Warning: --gpu specified but no GPU found. Using CPU.")
        except Exception as e:
            print(f"⚠ Warning: Could not check GPU status: {e}")
    
    # Run optimization
    print(f"\nStarting optimization:")
    print(f"  Trials: {args.trials}")
    print(f"  Study name: {args.name}")
    print(f"  Seed: {args.seed}\n")
    
    study = run_optimization(
        n_trials=args.trials, 
        study_name=args.name
    )
    
    print(f"\n✓ Optimization complete!")
    print(f"  Best score: {study.best_value:.4f}")
    print(f"  Results saved to study: {args.name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
