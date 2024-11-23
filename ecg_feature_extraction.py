import numpy as np
from scipy.signal import find_peaks

def extract_rr_intervals(ecg_data, fs):
    '''
    Extract the RR intervals from the ECG signal by detecting R-peaks
    R-Peaks are the highest point in the QRS complex, which is a part of the heart beat cycle
    RR intervals are the time between two consecutive R-peaks
    
    Parameters:
        ecg_data(ndarray): The ECG signal
        fs (float): Sampling frequency in Hz
        
    Returns:
        ndarray: The RR intervals in seconds
    '''
    # Detect R-peaks usign peak detection method
    peaks, _ = find_peaks(ecg_data, distance = fs*0.6) # Minimum heart rate = 100 bpm
    rr_intervals = np.diff(peaks) / fs # Time differences between consecutive R-peaks
    return rr_intervals

def calculate_heart_rate(rr_intervals):
    '''
    Calculating heart rate from RR intervals
    
    Parameters:
        rr_intervals(ndarray): The RR intervals in seconds
        
    Returns:
        float: The heart rate in beats per minute (bpm)
    '''
    mean_rr = np.mean(rr_intervals) # Average RR interval in seconds
    heart_rate = 60 / mean_rr # Heart rate in bpm
    return heart_rate

def extract_statistical_features(rr_intervals):
    '''
    Calculate statistical features using RR intervals
    
    Parameters:
        rr_intervals(ndarray): The RR intervals in seconds
        
    Returns:
        dict: A dictionary of statistical features (mean, std, skewness, kurtosis)
    '''
    features = {
        'mean_rr' : np.mean(rr_intervals),
        'std_rr' : np.std(rr_intervals),
        'min_rr' : np.min(rr_intervals),
        'max_rr' : np.max(rr_intervals),
        'range_rr' : np.max(rr_intervals) - np.min(rr_intervals)
    }
    return features

def extract_frequency_features(ecg_data, fs):
    '''
    Extract frequency-domain features from the ECG signal
    
    Parameters:
        ecg_data(ndarray): The ECG signal
        fs (float): Sampling frequency in Hz
        
    Returns:
        dict: A dictionary of frequency-domain features
    '''
    freqs = np.fft.fttfreq(len(ecg_data), d=1/fs) # Frequency spectrum of the ECG signal
    fft_values = np.abs(np.fft.fft(ecg_data)) # Amplitude spectrum of the ECG signal
    
    # Calculate power in specific frequency bands
    power_vlf = np.sum(fft_values[(freqs > 0.003) & (freqs <= 0.04)]) # Very low frequency range (0.003 - 0.04 Hz)
    power_lf = np.sum(fft_values[(freqs > 0.04) & (freqs <= 0.15)]) # Low frequency range (0.04 - 0.15 Hz)
    power_hf = np.sum(fft_values[(freqs > 0.15) & (freqs <= 0.5)]) # High frequency range (0.15 - 0.5 Hz)
    
    return {
        'power_vlf' : power_vlf,
        'power_lf' : power_lf,
        'power_hf' : power_hf
    }