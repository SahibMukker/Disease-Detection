import wfdb
import pandas as pd

def load_ecg_files(base_path):
    '''
    Loading data from .dat, .hea files
    
    Parameters:
        dat_path (str): Base path to the .dat files without extensions
        
    Returns:
        tuple: wfdb Record object and wfdb Annotation object
    '''
    
    # Load the record and annotation using the base path
    record = wfdb.rdrecord(base_path)
    annotations = wfdb.rdann(base_path, ext='atr')
    
    return record, annotations

def load_healthy_ecg_data(csv_path):
    '''
    Load healthy ECG dataset from a CSV file
    
    Parameters:
        csv_path (str): Path to the CSV file
    
    Returns:
        DataFrame: Pandas DataFrame containing healthy ECG data
    '''
    return pd.read_csv(csv_path)

def load_datasets(dat_path, csv_path):
    '''
    Load both raw ECG signals (from .dea/.hea files and healthy ECG data from .CSV file)
    
    Parameters:
        dat_path (str): Base path to the .dat files without extensions
        csv_path (str): Path to the CSV file
    
    Returns:
        tuple: (wfdb Record object, wfdb Annotation object, DataFrame)
    '''
    record, annotaitons = load_ecg_files(dat_path)
    healthy_ecg = load_healthy_ecg_data(csv_path)
    
    return record, annotaitons, healthy_ecg