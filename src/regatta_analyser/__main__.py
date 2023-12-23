# __main__.py
from .regatta_data import RegattaData
import sys

def main():
    
    if '--weather' in sys.argv:
        import subprocess
        subprocess.run(['python', 'src/regatta_analyser/weather_analyser.py'])
        return 1
    

    rdb = RegattaData(recreate=False)


if __name__ == "__main__":
    main()