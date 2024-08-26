import subprocess

def list_installed_packages(output_file):
    result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
    with open(output_file, 'w') as file:
        file.write(result.stdout)

if __name__ == "__main__":
    output_file = 'installed_packages.txt'  # Define the output file name
    list_installed_packages(output_file)
    print(f"Installed packages have been saved to {output_file}")
