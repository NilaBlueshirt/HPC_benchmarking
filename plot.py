import os
import re
import sys
import matplotlib.pyplot as plt
import pandas as pd
from itertools import cycle
import numpy as np

def print_help():
    """Displays the usage instructions."""
    print("""
Usage: python script.py <directory> <plot_type> [output_file]

Arguments:
  <directory>   Directory where the .out files are stored.
  <plot_type>   Type of plot: "bibw" for bandwidth or "latency".
  [output_file] Optional output file name. If not provided, defaults to "plot.png".

Examples:
  python script.py /path/to/log bibw bandwidth_plot.png
  python script.py /path/to/log latency latency_plot.png
  python script.py /path/to/log bibw

The last example will save the plot as "plot.png" by default.
    """)

def parse_out_file(file_path, plot_type):
    """Parses the output file and extracts the 'Size' and the 'Bandwidth' or 'Latency' columns based on the plot_type."""
    size = []
    values = []
    
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'(\d+)\s+([\d\.]+)', line)
            if match:
                size.append(int(match.group(1)))
                values.append(float(match.group(2)))
    
    if plot_type == "bibw":
        return pd.DataFrame({'Size': size, 'Bandwidth': values})
    elif plot_type == "latency":
        return pd.DataFrame({'Size': size, 'Latency': values})

def extract_module_info(file_name):
    """Extracts the module information for grouping files based on the library, version, and optional CUDA flag."""
    # Ensure the module name ends at "7.4-cuda" if "cuda" is present in the file name
    match = re.match(r'\d+_microOSU-([a-zA-Z\-]+-[\d\.]+-7.4-cuda)', file_name)
    
    if not match:
        # If no "cuda" is found, extract module name as usual without forcing the end at "7.4"
        match = re.match(r'\d+_microOSU-([a-zA-Z\-]+-[\d\.]+(?:-cuda)?)', file_name)
    
    if match:
        return match.group(1)
    return None

def plot_data(log_dir, output_file, plot_type):
    """Generates either bandwidth or latency plot from all .out files in the log directory."""
    
    # Define a custom set of at least 16 distinguishable colors
    color_list = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
        "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
        "#98df8a", "#ff9896", "#c5b0d5", "#c49c94"
    ]
    color_cycle = cycle(color_list)  # Cycle through the 16 custom colors
    
    module_colors = {}  # Dictionary to store assigned colors for each module
    modules_plotted = set()  # To track modules that have been added to the legend
    
    plt.figure(figsize=(10, 6))

    # Iterate through all .out files in the log directory
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.endswith(".out"):
                file_path = os.path.join(root, file)
                
                # Parse the file and extract data based on plot_type (bibw or latency)
                data = parse_out_file(file_path, plot_type)
                
                # Replace any non-positive values with a small positive number (to avoid log scale issues)
                data = data.replace(0, 1e-3)
                data = data[data > 0]  # Remove any negative values

                # Extract module info from the filename
                module = extract_module_info(file)
                if module is None:
                    continue
                
                # Assign a color to the module if it's not already in the dictionary
                if module not in module_colors:
                    module_colors[module] = next(color_cycle)  # Assign next color
                
                # Plot with the appropriate color based on the module
                color = module_colors[module]
                
                # If the module hasn't been plotted yet, add it to the legend
                label = None
                if module not in modules_plotted:
                    label = f'{module}'  # Use only the module name for the legend
                    modules_plotted.add(module)
                
                if plot_type == "bibw":
                    plt.plot(data['Size'], data['Bandwidth'], label=label, color=color)
                elif plot_type == "latency":
                    plt.plot(data['Size'], data['Latency'], label=label, color=color)

    # Customize plot
    if plot_type == "bibw":
        plt.title('Bandwidth vs. Message Size')
        plt.ylabel('Bandwidth (MB/s)')
    elif plot_type == "latency":
        plt.title('Latency vs. Message Size')
        plt.ylabel('Latency (us)')
        
    plt.xlabel('Size (Bytes)')

    # Set log scale for both the x-axis and y-axis
    plt.xscale('log')
    plt.yscale('log')

    # Set the limits manually to avoid display issues if needed
    plt.xlim(1, None)  # Ensure x-axis starts at 1 or greater
    plt.ylim(1e-3, None)  # Ensure y-axis starts at a small positive number

    # Customize grid lines for x and y axes
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.minorticks_on()  # Enable minor ticks
    
    # Major ticks for both axes
    plt.grid(True, which='major', linestyle='-', linewidth=0.75)

    # Minor ticks with finer grid lines
    plt.grid(True, which='minor', linestyle=':', linewidth=0.5)
    
    # Place legend outside the plot
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    
    plt.grid(True)  # Enable the grid
    
    # Save the plot as a PNG file
    plt.savefig(output_file, bbox_inches='tight')  # Ensure the plot fits within the output file
    plt.close()

if __name__ == "__main__":
    # Check if correct number of arguments is passed
    if len(sys.argv) < 3:
        print("Error: Insufficient arguments provided.")
        print_help()
        sys.exit(1)

    # Read arguments from sys.argv
    log_dir = sys.argv[1]  # Directory where the .out files are stored
    plot_type = sys.argv[2]  # Type of plot: "bibw" for bandwidth or "latency"

    # Check if the plot type is valid
    if plot_type not in ['bibw', 'latency']:
        print('Error: Invalid plot type. Use "bibw" for bandwidth or "latency".')
        print_help()
        sys.exit(1)

    # Optional output file name, default is 'plot.png'
    output_file = sys.argv[3] if len(sys.argv) > 3 else 'plot.png'

    # Generate the plot (either bandwidth or latency) from all .out files in the specified directory
    plot_data(log_dir, output_file, plot_type)

