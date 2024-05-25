
# Config
DATABASE_FILE = 'data/regatta_data.duckdb'
ORC_SPEED_GUIDE_FILE = 'data/input/ORC_Speed_Guide_Ref_04340002PIL.csv'

ROLLING_AVG_SECONDS = 10

class TableNames:
    orc_model               = "tbl_orc_target"        # ORC target speeds etc for TWS and TWA
    tmp_raw_logs            = "tmp_race_logs"         # Preproccesed logs from the current inout file
    # Raw log preprocessing
    tmp_logs_import         = "tmp_logs_import"        #import
    tmp_logs_interpolate    = "tmp_logs_interpolate"   #interpolate over all seconds
    tmp_logs_preprocessed   = "tmp_logs_preprocessed"
    # Table that contains all preprocessed raw logs
    raw_logs                = "tbl_race_logs"         # Preprocessed logs for all the processed logs (incremental table)

    # Include ORC Model Tartgets
    view_targets            = "view_targets"




INPUT_FOLDER_PATH='data/input/'

SOURCE_FILES = [
    {'race_tag':'2023_09_07', 'source_file': 'kolmak_2023_09_07.csv', 'start_dt': None, 'end_dt': None},
    {'race_tag':'2023_09_27', 'source_file': 'kolmak_2023_09_27.csv', 'start_dt': None, 'end_dt': None},
    {'race_tag':'2023_10_04', 'source_file': 'kolmak_2023_10_04.csv', 'start_dt': None, 'end_dt': None},
    {'race_tag':'2024_05_04', 'source_file': 'avavoistlus_2024_05_04.csv', 'start_dt': None, 'end_dt': None},
    {'race_tag':'2024_05_08', 'source_file': 'kolmak-2024-05-08.csv', 'start_dt': None, 'end_dt': None},
    {'race_tag':'2024_05_15', 'source_file': 'kolmak-2024-05-15.csv', 'start_dt': None, 'end_dt': None},
    {'race_tag':'2024_05_22', 'source_file': 'kolmak-2024-05-22.csv', 'start_dt': None, 'end_dt': None}
    ]