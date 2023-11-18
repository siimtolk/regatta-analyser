# __main__.py
from .analyse import analyse
import sys

def main():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: python -m regatta_analyser <file_path>")
        sys.exit(1)

    # Get the file path from the command-line arguments
    file_path = sys.argv[1]

    # Call the analyse function with the provided file path
    analyse(file_path)

if __name__ == "__main__":
    main()