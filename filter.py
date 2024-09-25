import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
This script should be placed inside the rvl-cdip folder.
It reads the text files (test.txt, train.txt, val.txt) located in the 'labels' folder.
For each file, it creates a corresponding subfolder (e.g., 'test_file', 'train_file', 'val_file').
Within each subfolder, a 'filtered_labels.txt' file is generated, containing only the directory paths 
from the original file that match specific ID values. The target ID values are defined in 'filter_ids'.

In the second part, the script:
1. Reads the 'filtered_labels.txt' file in each subfolder.
2. Creates a 'files' directory in each subfolder.
3. Copies or moves the files listed in 'filtered_labels.txt' to the 'files' directory.
The base path for the images is rvl-cdip/images/.
"""

# Save the current directory and define paths
current_directory = os.getcwd()  # Get the current working directory
labels_directory = os.path.join(current_directory, 'labels')  # Path to the 'labels' directory
images_base_directory = os.path.join(current_directory, 'images')  # Base directory for images

# List of input files to process, located in the 'labels' directory
input_files = ['test.txt'] #, 'train.txt', 'val.txt'

# Define a set of IDs to filter by, as strings. This allows filtering for multiple ID values.
filter_ids = {'14'}  # Use a set for fast membership testing

# Process each file in the 'input_files' list
for input_file in input_files:
    # Construct the full path to the input file within the 'labels' directory
    input_file_path = os.path.join(labels_directory, input_file)
    
    # Create an output directory based on the input file name without '.txt'
    # Add '_file' to distinguish the output directory from the original input name
    output_directory = os.path.join(current_directory, os.path.splitext(input_file)[0] + '_file')
    os.makedirs(output_directory, exist_ok=True)  # Create the directory if it doesn't exist
    
    # Define the path for the output 'filtered_labels.txt' file within the output directory
    output_file_path = os.path.join(output_directory, 'filtered_labels.txt')

    # Check if the input file exists before processing
    if os.path.isfile(input_file_path):
        # Open the input file for reading and the output file for writing
        with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
            # Read each line in the input file
            for line in infile:
                # Split the line into 'path' and 'number'
                path, number = line.split()
                # If 'number' matches any of the values in 'filter_ids', write the 'path' to the output file
                if number in filter_ids:
                    outfile.write(f"{path}\n")  # Write only the path followed by a newline
        print(f"Filtered paths from {input_file} written to {output_file_path}")
    else:
        # Print a warning if the input file doesn't exist
        print(f"The file {input_file_path} does not exist!")

# Second part of the script: Process each '_file' directory

def copy_file(source_path, destination_path):
    """Function to copy a file from source_path to destination_path."""
    try:
        shutil.copy(source_path, destination_path)
        return f"Copied {source_path} to {destination_path}"
    except Exception as e:
        return f"Failed to copy {source_path}. Error: {str(e)}"

# Number of parallel workers to use
max_workers = 8

for input_file in input_files:
    # Create the corresponding output directory name (without '.txt' and add '_file')
    output_directory = os.path.join(current_directory, os.path.splitext(input_file)[0] + '_file')
    filtered_labels_path = os.path.join(output_directory, 'filtered_labels.txt')

    # Define the 'files' subdirectory where the files will be copied/moved
    files_directory = os.path.join(output_directory, 'files')
    os.makedirs(files_directory, exist_ok=True)  # Create 'files' directory if it doesn't exist

    # Check if 'filtered_labels.txt' exists before processing
    if os.path.isfile(filtered_labels_path):
        with open(filtered_labels_path, 'r') as file:
            file_paths = [line.strip() for line in file]

        # Use ThreadPoolExecutor for parallel file copying
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {}
            for relative_path in file_paths:
                # Construct the full path to the source file in the 'images' directory
                source_path = os.path.join(images_base_directory, relative_path)

                # Check if the source file exists
                if os.path.isfile(source_path):
                    # Define destination path
                    destination_path = os.path.join(files_directory, os.path.basename(source_path))
                    # Submit the copy task to the thread pool
                    future = executor.submit(copy_file, source_path, destination_path)
                    future_to_file[future] = source_path
                else:
                    print(f"Source file {source_path} does not exist and could not be copied!")

            # Process completed futures
            for future in as_completed(future_to_file):
                print(future.result())  # Print the result of the copy operation
    else:
        # Print a warning if 'filtered_labels.txt' does not exist
        print(f"The file {filtered_labels_path} does not exist!")

print("DONE")