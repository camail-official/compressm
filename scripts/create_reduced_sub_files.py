#!/usr/bin/env python3

import os
import json

# Configuration
dataset = "scifar"  # Change this to match your dataset

# Read the base configuration to get the SSM dimension
with open(f'experiment_configs/repeats/lru/{dataset}.json', 'r') as f:
    base_config = json.load(f)

ssm_dim = base_config['ssm_dim']
seeds = [8, 42, 123, 456, 789]
tolerance_values = [1.5e-1, 1e-1, 7e-2, 5e-2, 3e-2, 2e-2]

# Create cluster directory if it doesn't exist
cluster_dir = f"cluster/{dataset}"
os.makedirs(cluster_dir, exist_ok=True)

print(f'Creating HTCondor submission files for {dataset} reduced experiments...')

# Create the reduced submission file
sub_content = f"""executable = /home/pnazari/cluster/htcondor/empty_wrapper.sh
initialdir = /home/pnazari/workspace/compressm
arguments = /home/pnazari/.local/bin/uv run --no-sync -m run_experiment --dataset_name {dataset}/reduced/{dataset}-{ssm_dim}-$(seed)-$(tol)
error = outputs/$(Cluster)_$(Process).err
output = outputs/$(Cluster)_$(Process).out
log = outputs/$(Cluster)_$(Process).log
request_memory = 64000
request_cpus = 8
request_gpus = 1
requirements = TARGET.CUDAGlobalMemoryMb > 64000
MaxTime = 288000
periodic_remove = (JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))

queue tol, seed from (
"""

# Add all tolerance-seed combinations
for tol in tolerance_values:
    for seed in seeds:
        # Format tolerance to match the existing format (scientific notation)
        if tol == 0.15:
            tol_str = "1.5e-1"
        elif tol == 0.1:
            tol_str = "1e-1"
        elif tol == 0.07:
            tol_str = "7e-2"
        elif tol == 0.05:
            tol_str = "5e-2"
        elif tol == 0.03:
            tol_str = "3e-2"
        elif tol == 0.02:
            tol_str = "2e-2"
        else:
            tol_str = f"{tol:.0e}".replace('+', '').replace('-0', '-')
        sub_content += f"{tol_str} {seed}\n"

sub_content += ")"

# Write the submission file
sub_filename = f"{cluster_dir}/{dataset}-reduced.sub"
with open(sub_filename, 'w') as f:
    f.write(sub_content)

print(f'Created: {sub_filename}')

print(f'\nCreated HTCondor submission files:')
print(f'- Batch file: {sub_filename} ({len(seeds) * len(tolerance_values)} jobs)')
print(f'SSM dimension: {ssm_dim}')
print(f'Tolerance values: {tolerance_values}')
print(f'Seeds: {seeds}')
