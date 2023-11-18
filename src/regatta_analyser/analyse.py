from . import *
from .utils import *
import duckdb 


def analyse(regatta_log_path):
    create_file_if_not_there(TMP_DB_FILE_PATH)
    print('asd')
    #print(f'Analyse logs in {regatta_log_path}')
