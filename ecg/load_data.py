import pandas as pd
import os
import zipfile


def load_data():
    ecg_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = os.path.join(ecg_directory, 'ecg_data')
    data_file = os.path.join(data_directory, 'ecg_data.csv')

    if (not os.path.exists(data_file)):
        print('Data file not found. Downloading and extracting data...')
        zip_file = os.path.join(ecg_directory, 'ecg_data.zip')
        zipfile.ZipFile(zip_file).extractall(ecg_directory)

    data = pd.read_csv(data_file, index_col=0)
    return data
