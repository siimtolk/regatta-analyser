import os

def create_file_if_not_there(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        with open(file_path, 'w'):
            pass  # Creates an empty file

