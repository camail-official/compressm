#!/usr/bin/env python3

import os

# Configuration
model = "imdb"  # Change this to match your model
seeds = [8, 42, 123, 456, 789]
new_ssm_dims = [192, 165, 150, 136, 119, 95]  # Change this to match your SSM dimensions

# Create cluster directory if it doesn't exist
cluster_dir = f"cluster/{model}"
os.makedirs(cluster_dir, exist_ok=True)

print(f'Creating HTCondor submission files for {model}...')

# Create the non-reduced-new submission file
sub_content = f"""executable = /home/pnazari/cluster/htcondor/empty_wrapper.sh
initialdir = /home/pnazari/workspace/compressm
arguments = /home/pnazari/.local/bin/uv run --no-sync -m run_experiment --dataset_name {model}/non-reduced-adapted/{model}-$(seed)-$(dim)
error = outputs/$(Cluster)_$(Process).err
output = outputs/$(Cluster)_$(Process).out
log = outputs/$(Cluster)_$(Process).log
request_memory = 64000
request_cpus = 8
request_gpus = 1
requirements = TARGET.CUDAGlobalMemoryMb > 64000
MaxTime = 288000
periodic_remove = (JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))

queue dim, seed from (
"""

# Add all dimension-seed combinations
for ssm_dim in new_ssm_dims:
    for seed in seeds:
        sub_content += f"{ssm_dim} {seed}\n"

sub_content += ")"

# Write the submission file
sub_filename = f"{cluster_dir}/{model}-non-reduced-adapted.sub"
with open(sub_filename, 'w') as f:
    f.write(sub_content)

print(f'Created: {sub_filename}')

print(f'\nCreated HTCondor submission files:')
print(f'- Batch file: {sub_filename} ({len(seeds) * len(new_ssm_dims)} jobs)')
print(f'SSM dimensions: {new_ssm_dims}')
print(f'Seeds: {seeds}')
