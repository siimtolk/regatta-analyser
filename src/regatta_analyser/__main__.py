# __main__.py
from .analyse import Analyser
import sys

def main():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 2:
        print("Usage: regatta_analyser <regatta log path.csv>")
        sys.exit(1)

    # Get the file path from the command-line arguments
    log_file_path = sys.argv[1]

    # Call the analyse function with the provided file path
    a = Analyser(log_file_path)
    

if __name__ == "__main__":
    main()