import numpy as np
from scipy.signal import butter, filtfilt, resample

def butter_lowpass_filter(data, cutoff,fs, order=5):
    '''
    Apply a lowpass Butterworth filter to the singal (removes high-frequency noise)
    
    Parameters:
        data(ndarray): The ECG signal
        cutoff (float): Cutoff frequency in Hz
        fs (float): Sampling frequency in Hz
        order(int): The order of the Butterworth filter
        
    Returns:
        ndarray: The filtered ECG signal
    '''
    nyquist = 0.5 * fs # nyquist frequency (nyquist frequency is 0.5 times the sampling rate, this is done to avoid aliasing (high frequency noise))
    normal_cutoff = cutoff / nyquist # normalizing the cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False) # butterworth filter
    
    return filtfilt(b, a, data) # applying the filter with zero-phase shift

def butter_highpass_filter(data, cutoff,fs, order=5):
    '''
    Apply a highpass Butterworth filter to the singal (removes low-frequency noise)
    
    Parameters:
        data(ndarray): The ECG signal
        cutoff (float): Cutoff frequency in Hz
        fs (float): Sampling frequency in Hz
        order(int): The order of the Butterworth filter
        
    Returns:
        ndarray: The filtered ECG signal
    '''
    nyquist = 0.5 * fs # nyquist frequency
    normal_cutoff = cutoff / nyquist # normalizing the cutoff frequency
    b, a = butter(order, normal_cutoff, btype='high', analog=False) # butterworth filter
    
    return filtfilt(b, a, data) # applying the filter with zero-phase shift

def resample_data(data, original_fs, target_fs):
    '''
    Resample the ECG signal to a new sampling frequency
    
    Parameters:
        data(ndarray): The ECG signal
        original_fs (float): Original sampling frequency in Hz
        target_fs (float): Target sampling frequency in Hz
        
    Returns:
        ndarray: The resampled ECG signal
    '''
    if target_fs == original_fs or original_fs is None:
        return data # if the target sampling frequency is the same as the original sampling frequency, return the original signal
    
    num_samples = int(len(data) * target_fs / original_fs)
    
    return resample(data, num_samples)

def normalize_data(data):
    '''
    Normalize the ECG signal to the range [-1, 1]
    
    Parameters:
        data(ndarray): The ECG signal
        
    Returns:
        ndarray: The normalized ECG signal
    '''
    return 2 * (data - np.min(data)) / (np.max(data) - np.min(data)) - 1