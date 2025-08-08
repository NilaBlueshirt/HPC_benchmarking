import sys
import random
import os
import subprocess

def read_file(file_path):
    with open(file_path, 'r') as file:
        return [line.strip() for line in file]

def shuffle_and_filter_nodes(nodes):
    random.shuffle(nodes)
    
    # If odd number of nodes, randomly exclude one
    if len(nodes) % 2 != 0:
        random_index = random.randint(0, len(nodes) - 1)
        print(f"Odd number of nodes detected, randomly excluding node: {nodes[random_index]}")
        nodes.pop(random_index)
    
    return nodes

def create_pairs(nodes):
    pairs = []
    
    # Shuffle nodes and create pairs
    while len(nodes) > 1:
        # Randomly select two distinct nodes
        node1 = random.choice(nodes)
        node2 = random.choice(nodes)
        
        # Ensure node1 and node2 are different
        while node1 == node2:
            node2 = random.choice(nodes)
        
        # Add the pair and remove both nodes from the list
        pair = f"{node1},{node2}"
        pairs.append(pair)
        nodes.remove(node1)
        nodes.remove(node2)

    return pairs

def generate_job_script(i, pair, omb_name, mpi_cmd, sanitized_omb_name):
    node1, node2 = pair.split(',')
    
    # Create job script content
    job_script = f"""#!/bin/bash
#SBATCH --job-name=bibw_{omb_name}_{node1}_{node2}
#SBATCH --reservation=maint
#SBATCH -p htc
#SBATCH -q public
#SBATCH -t 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -N 2
#SBATCH -n 2
#SBATCH -c 1
#SBATCH --nodelist={node1},{node2}
#SBATCH --output=log/{i}_{sanitized_omb_name}_{node1}_{node2}_%j.out
#SBATCH --export=NONE

echo "Running on nodes {node1} and {node2} with OMB {omb_name}"
module load {omb_name}
{mpi_cmd} "osu_bibw"
"""
    return job_script

def main(nodelist_file, omb_name, mpi_cmd):
    # Read node list
    nodes = read_file(nodelist_file)
    
    # Shuffle nodes and filter out if the number of nodes is odd
    shuffled_nodes = shuffle_and_filter_nodes(nodes)
    
    # Create pairs (each node is only used once, no node pairs with itself)
    pairs = create_pairs(shuffled_nodes)
    
    # Sanitize OMB name for use in filenames
    sanitized_omb_name = omb_name.replace('/', '-')  # Sanitize name
    
    # Iterate through each pair and create job script
    for i, pair in enumerate(pairs):
        # Generate job script
        job_script = generate_job_script(i, pair, omb_name, mpi_cmd, sanitized_omb_name)
        
        # Create script file
        # Check if log dir exists in the current working dir
        log_dir = "log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        script_name = f"log/bibw_{i}_{sanitized_omb_name}_{pair.replace(',', '_')}.sh"
        with open(script_name, 'w') as script_file:
            script_file.write(job_script)
        os.chmod(script_name, 0o755)
        
        # Submit job using sbatch
        subprocess.run(['sbatch', script_name])

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 script.py <nodelist> <omb_name> <mpi_cmd>")
        sys.exit(1)
    
    nodelist_file = sys.argv[1]
    omb_name = sys.argv[2]
    mpi_cmd = sys.argv[3]
    
    main(nodelist_file, omb_name, mpi_cmd)

