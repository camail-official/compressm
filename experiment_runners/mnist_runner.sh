#!/bin/bash
# filepath: /home/choukram/repos/LinOSS/run_ssm_dim_experiment.sh

# Define the path to the configuration file
CONFIG_FILE="experiment_configs/repeats/lru/mnist.json"

# Define the different ssm_dim values to test
SSM_DIMS=(16 32 64 128 256)

# shuffle the ssmdims
SSM_DIMS=($(shuf -e "${SSM_DIMS[@]}"))

# Define the different num_blocks values to test
NUM_BLOCKS=(1)

# Define the different learning rates to test
LEARNING_RATES=(0.0004)

# Define the tolerance values to test with each dimension
TOLERANCES=(-1)

# Define GPU assignments for each experiment (modify as needed)
GPU_ASSIGNMENTS=(0 1 2 3)  # Use GPU 0 for first experiment, GPU 1 for second, etc.
# If you have fewer GPUs than experiments, they will cycle through the available GPUs

# Function to update the JSON configuration file
update_config() {
    local ssm_dim=$1
    local num_blocks=$2
    local lr=$3
    local tol=$4
    
    # Use jq to update the configuration AND set single seed to prevent multiple runs
    jq ".ssm_dim = \"$ssm_dim\" | .num_blocks = \"$num_blocks\" | .lr = \"$lr\" | .tol = $tol" $CONFIG_FILE > temp.json
    mv temp.json $CONFIG_FILE
    
    echo "Updated configuration with ssm_dim=$ssm_dim, num_blocks=$num_blocks, lr=$lr, tol=$tol"
}

# Function to run a group of experiments sequentially on one GPU
run_experiment_group() {
    local gpu_id=$1
    local group_num=$2
    shift 2
    local experiments=("$@")
    
    local group_window_name="gpu${gpu_id}_group${group_num}"
    
    echo "=========================================="
    echo "Starting experiment group $group_num on GPU $gpu_id"
    echo "Group contains ${#experiments[@]} experiments"
    echo "Window: $group_window_name"
    echo "=========================================="
    
    # Build the command string for sequential execution
    local cmd="cd $(pwd) && echo 'Starting experiment group $group_num on GPU $gpu_id'"
    
    # Add each experiment to the command chain
    local exp_count=0
    for exp in "${experiments[@]}"; do
        IFS=',' read -r ssm_dim num_blocks lr tol <<< "$exp"
        exp_count=$((exp_count + 1))
        cmd="$cmd && echo '--- Running experiment $exp_count/${#experiments[@]} in group $group_num: ssm_dim=$ssm_dim, num_blocks=$num_blocks, lr=$lr, tol=$tol ---'"
        cmd="$cmd && jq \".ssm_dim = \\\"$ssm_dim\\\" | .num_blocks = \\\"$num_blocks\\\" | .lr = \\\"$lr\\\" | .tol = $tol\" $CONFIG_FILE > temp.json && mv temp.json $CONFIG_FILE"
        cmd="$cmd && CUDA_VISIBLE_DEVICES=$gpu_id python run_experiment.py --dataset_name mnist"
        cmd="$cmd && echo 'Completed experiment $exp_count/${#experiments[@]} in group $group_num: ssm_dim=$ssm_dim, num_blocks=$num_blocks, lr=$lr, tol=$tol'"
    done
    
    cmd="$cmd && echo 'All experiments in group $group_num completed on GPU $gpu_id' && bash"
    
    # Create tmux window and run the sequential experiments
    tmux new-window -n "$group_window_name" -d "$cmd"
    
    echo "Experiment group $group_num launched on GPU $gpu_id"
}

# Main execution
echo "Starting SSM dimension experiments"

# Check if tmux session exists, if not create one
if ! tmux has-session -t ssm_experiments 2>/dev/null; then
    echo "Creating new tmux session: ssm_experiments"
    tmux new-session -d -s ssm_experiments -c $(pwd)
else
    echo "Using existing tmux session: ssm_experiments"
fi

# Calculate total number of experiments
total_experiments=$((${#SSM_DIMS[@]} * ${#NUM_BLOCKS[@]} * ${#LEARNING_RATES[@]} * ${#TOLERANCES[@]}))
experiments_per_gpu=$((total_experiments / ${#GPU_ASSIGNMENTS[@]}))
remaining_experiments=$((total_experiments % ${#GPU_ASSIGNMENTS[@]}))

echo "=== PHASE 1: Running grid search experiments in groups (sequential per GPU) ==="
echo "Total experiments: $total_experiments"
echo "GPUs available: ${#GPU_ASSIGNMENTS[@]}"
echo "Experiments per GPU: $experiments_per_gpu (plus $remaining_experiments extra for first GPUs)"
echo "Parameters: SSM_DIMS(${#SSM_DIMS[@]}), NUM_BLOCKS(${#NUM_BLOCKS[@]}), LEARNING_RATES(${#LEARNING_RATES[@]}), TOLERANCES(${#TOLERANCES[@]})"

# Create array of all experiment configurations
all_experiments=()
for ssm_dim in "${SSM_DIMS[@]}"; do
    for num_blocks in "${NUM_BLOCKS[@]}"; do
        for lr in "${LEARNING_RATES[@]}"; do
            for tol in "${TOLERANCES[@]}"; do
                all_experiments+=("$ssm_dim,$num_blocks,$lr,$tol")
            done
        done
    done
done

echo "Generated ${#all_experiments[@]} experiment configurations"

# Split experiments into groups for each GPU
for gpu_index in "${!GPU_ASSIGNMENTS[@]}"; do
    gpu_id=${GPU_ASSIGNMENTS[$gpu_index]}
    group_num=$((gpu_index + 1))
    
    # FIXED: Calculate start and end indices correctly
    if [ $gpu_index -lt $remaining_experiments ]; then
        # First few GPUs get one extra experiment
        start_idx=$((gpu_index * (experiments_per_gpu + 1)))
        end_idx=$(((gpu_index + 1) * (experiments_per_gpu + 1)))
    else
        # Later GPUs get normal amount, but offset by the extra experiments
        start_idx=$((remaining_experiments * (experiments_per_gpu + 1) + (gpu_index - remaining_experiments) * experiments_per_gpu))
        end_idx=$((start_idx + experiments_per_gpu))
    fi
    
    # Extract experiments for this GPU
    gpu_experiments=("${all_experiments[@]:$start_idx:$((end_idx - start_idx))}")
    
    echo ""
    echo "GPU $gpu_id (Group $group_num): ${#gpu_experiments[@]} experiments (indices $start_idx to $((end_idx-1)))"
    
    # Run this group of experiments sequentially on the assigned GPU
    run_experiment_group $gpu_id $group_num "${gpu_experiments[@]}"
    
    # Small delay between starting different GPU groups
    sleep 2
done

echo ""
echo "All experiment groups launched!"
echo "Total experiments: $total_experiments distributed across ${#GPU_ASSIGNMENTS[@]} GPUs"
echo "To monitor experiments, use: tmux attach -t ssm_experiments"
echo "To list all windows: tmux list-windows -t ssm_experiments"
echo ""
echo "Experiment summary:"
echo "- SSM dimensions: ${SSM_DIMS[*]}"
echo "- Number of blocks: ${NUM_BLOCKS[*]}"
echo "- Learning rates: ${LEARNING_RATES[*]}"
echo "- Tolerances: ${TOLERANCES[*]}"
echo "- GPUs used: ${GPU_ASSIGNMENTS[*]}"

# Helper commands for monitoring:
echo ""
echo "=== MONITORING COMMANDS ==="
echo "1. Attach to tmux session:    tmux attach -t ssm_experiments"
echo "2. List all windows:          tmux list-windows -t ssm_experiments"
echo "3. Switch to GPU group N:     tmux select-window -t ssm_experiments:gpu<N>_group<N>"
echo "4. Kill all experiments:      tmux kill-session -t ssm_experiments"
echo "5. Monitor GPU usage:         watch -n 2 nvidia-smi"
echo ""
echo "Window naming: gpu<GPU_ID>_group<GROUP_NUM>"
echo ""
echo "=== EXPERIMENT GRID DETAILS ==="
echo "Parameter combinations being tested:"
echo "  SSM Dimensions: ${#SSM_DIMS[@]} values (${SSM_DIMS[*]})"
echo "  Number of Blocks: ${#NUM_BLOCKS[@]} values (${NUM_BLOCKS[*]})"
echo "  Learning Rates: ${#LEARNING_RATES[@]} values (${LEARNING_RATES[*]})"
echo "  Tolerances: ${#TOLERANCES[@]} values (${TOLERANCES[*]})"
echo "  Total combinations: ${#SSM_DIMS[@]} × ${#NUM_BLOCKS[@]} × ${#LEARNING_RATES[@]} × ${#TOLERANCES[@]} = $total_experiments"
echo ""
echo "Group distribution (CORRECTED):"
for gpu_index in "${!GPU_ASSIGNMENTS[@]}"; do
    gpu_id=${GPU_ASSIGNMENTS[$gpu_index]}
    group_num=$((gpu_index + 1))
    
    # Use the SAME calculation as above for consistency
    if [ $gpu_index -lt $remaining_experiments ]; then
        start_idx=$((gpu_index * (experiments_per_gpu + 1)))
        end_idx=$(((gpu_index + 1) * (experiments_per_gpu + 1)))
    else
        start_idx=$((remaining_experiments * (experiments_per_gpu + 1) + (gpu_index - remaining_experiments) * experiments_per_gpu))
        end_idx=$((start_idx + experiments_per_gpu))
    fi
    
    echo "- GPU $gpu_id (Group $group_num): $((end_idx - start_idx)) experiments (indices $start_idx to $((end_idx-1)))"
done