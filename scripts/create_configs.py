import json


model = "imdb"

# Read the base configuration from listops.json
with open(f'experiment_configs/repeats/lru/{model}.json', 'r') as f:
    base_config = json.load(f)

# Seeds and new SSM dimensions
seeds = [8, 42, 123, 456, 789]
new_ssm_dims = [192, 165, 150, 136, 119, 95]

print('Creating new listops configuration files with custom SSM dimensions...')

# Create configuration files for each seed and SSM dimension combination
for seed in seeds:
    for ssm_dim in new_ssm_dims:
        config = base_config.copy()
        config['seeds'] = [seed]
        config['ssm_dim'] = str(ssm_dim)
        config['tol'] = -1  # non-reduced case
        
        filename = f'experiment_configs/repeats/lru/{model}/non-reduced-adapted/{model}-{seed}-{ssm_dim}.json'
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f'Created: {model}-{seed}-{ssm_dim}.json')

print(f'\nTotal files created: {len(seeds) * len(new_ssm_dims)} files')
print(f'SSM dimensions used: {new_ssm_dims}')
print(f'Seeds used: {seeds}')
