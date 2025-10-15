import json
import os


dataset = "scifar"

# Read the base configuration from imdb.json
with open(f'experiment_configs/repeats/lru/{dataset}.json', 'r') as f:
    base_config = json.load(f)

# Seeds and tolerance values for reduced experiments
seeds = [8, 42, 123, 456, 789]
tolerance_values = [1.5e-1, 1e-1, 7e-2, 5e-2, 3e-2, 2e-2]

print('Creating new reduced configuration files with custom tolerance values...')

# Create the output directory if it doesn't exist
output_dir = f'experiment_configs/repeats/lru/{dataset}/reduced'
os.makedirs(output_dir, exist_ok=True)

# Create configuration files for each seed and tolerance combination
for seed in seeds:
    for tol in tolerance_values:
        config = base_config.copy()
        config['seeds'] = [seed]
        config['tol'] = tol  # reduced case with specific tolerance

        ssm_dim = config['ssm_dim']
        
        # Format tolerance for filename (replace scientific notation)
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
        filename = f'experiment_configs/repeats/lru/{dataset}/reduced/{dataset}-{ssm_dim}-{seed}-{tol_str}.json'
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f'Created: {dataset}-{ssm_dim}-{seed}-{tol_str}.json')

print(f'\nTotal files created: {len(seeds) * len(tolerance_values)} files')
print(f'Tolerance values used: {tolerance_values}')
print(f'Seeds used: {seeds}')
