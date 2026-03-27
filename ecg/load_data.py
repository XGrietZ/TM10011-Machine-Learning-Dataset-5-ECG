import pandas as pd
import os
import zipfile


def load_data():
    ecg_directory = os.path.dirname(os.path.abspath(__file__))              # Get the directory of the current file
    data_directory = os.path.join(ecg_directory, 'ecg_data')                # Define the path to the data directory
    data_file = os.path.join(data_directory, 'ecg_data.csv')                # Define the path to the data file

    if (not os.path.exists(data_file)):                                     # Check if the data file exists
        print('Data file not found. Downloading and extracting data...')    # If the data file does not exist, download and extract the data
        zip_file = os.path.join(ecg_directory, 'ecg_data.zip')              # Define the path to the zip file
        zipfile.ZipFile(zip_file).extractall(ecg_directory)                 # Extract the zip file to the current directory

    data = pd.read_csv(data_file, index_col=0)                              # Load the data from the CSV file, using the first column as the index
    return data                                                             # Return the loaded data as a pandas DataFrame
