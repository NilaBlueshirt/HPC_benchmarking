#!/bin/bash

# Check if the required arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <base_directory> <line_offset, use 23 for bibw and latency>"
    exit 1
fi

# Get the base directory and line offset from the arguments
base_dir="${1%/}" # Remove trailing slash if present
line_offset="$2"

# Check if the base directory exists
if [ ! -d "$base_dir" ]; then
    echo "Error: Directory '$base_dir' does not exist."
    exit 1
fi

# Check if the line offset is a valid number
if ! [[ "$line_offset" =~ ^[0-9]+$ ]]; then
    echo "Error: Line offset '$line_offset' is not a valid number."
    exit 1
fi

# Initialize a flag to check if any file is matched
found_match=false

# Find all .out files in the specified directory and its subdirectories
for file in $(find "$base_dir" -type f -name "*.out"); do
    # Get the line number containing "# Size"
    size_line=$(grep -n "# Size" "$file" | cut -d':' -f1)
    
    # Check if "# Size" was found
    if [ -z "$size_line" ]; then
        echo "File '$file' does not contain '# Size'."
        continue
    fi

    # Calculate the target line based on the offset
    target_line=$((size_line + line_offset))

    # Extract the target line
    line=$(sed -n "${target_line}p" "$file")
    
    # Check if the line exists
    if [ -z "$line" ]; then
        echo "File '$file' does not have a valid line at $target_line."
        continue
    fi

    # Extract the second column
    second_column=$(echo "$line" | awk '{print $2}')

    # Check if the second column matches the desired float format
    if [[ "$second_column" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        # Extract the digits before the decimal point
        integer_part=$(echo "$second_column" | cut -d'.' -f1)

        # Check if the integer part has less than four digits
        if [ ${#integer_part} -lt 4 ]; then
            echo "Matched file: $file, Line: $target_line, Second column: $second_column"
            found_match=true
        fi
    fi
done

# Final message if no matches found
if [ "$found_match" = false ]; then
    echo "No files matched the criteria in directory '$base_dir'."
fi

