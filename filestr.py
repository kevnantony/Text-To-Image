import os
from pathlib import Path

def list_files(directory: Path, indent_level: int = 0) -> str:
    """
    Recursively list files and directories in the given directory.
    
    :param directory: The directory to list.
    :param indent_level: The level of indentation for nested directories.
    :return: A string representation of the file structure.
    """
    file_structure = ""
    indent = "  " * indent_level

    for entry in os.scandir(directory):
        if entry.is_dir():
            file_structure += f"{indent}{entry.name}/\n"
            file_structure += list_files(entry.path, indent_level + 1)
        else:
            file_structure += f"{indent}{entry.name}\n"
    
    return file_structure

def save_file_structure(directory: Path, output_file: str):
    """
    Save the file structure of the given directory to a text file.
    
    :param directory: The directory to list.
    :param output_file: The path to the output text file.
    """
    structure = list_files(directory)
    with open(output_file, 'w') as file:
        file.write(structure)

if __name__ == "__main__":
    # Change this to the path of your project root directory
    project_root = Path('/home/ubuntu/image_generation')
    
    # Path to the output file
    output_file = 'file_structure.txt'
    
    save_file_structure(project_root, output_file)
    print(f"File structure saved to {output_file}")
