# __main__.py
from .analyse import Analyser
import sys

def main():
    
    if '--weather' in sys.argv:
        import subprocess
        subprocess.run(['python', 'src/regatta_analyser/weather_analyser.py'])
        return 1
    
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 4:
        print("Usage: regatta_analyser <tag> <regatta log path.csv> <ORC speed guide path> OR --weather")
        sys.exit(1)

    # Get the file path from the command-line arguments
    tag = sys.argv[1]
    log_file_path = sys.argv[2]
    orc_guide_path = sys.argv[3]

    

    # Call the analyse function with the provided file path
    a = Analyser(tag, log_file_path, orc_guide_path)


if __name__ == "__main__":
    main()