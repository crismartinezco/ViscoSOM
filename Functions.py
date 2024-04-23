import librosa
from scipy.signal import find_peaks
import numpy as np
import constants as c

class Preprocessing:
    """Loads all functions for data preprocessing"""
    def __init__(self):

        self.n_fft = 4096
        self.fft_freqs = librosa.fft_frequencies(sr = c.SR, n_fft=self.n_fft) # 4096 gives results that approach the 1/4 Tone resolution

    # Docstring for load_data function
    def load_data(self, signal):
        """
        Loads a time series data from a txt file.

        Args:
            signal (str): Path to the signal file.

        Returns:
            np.ndarray: The loaded time series data.
        """
        data = np.loadtxt(signal)
        return data#[:24000]

    # Docstring for fft_librosa_laplace function
    def fft_librosa_laplace(self, data):
        """
        Calculates the Short-Time Fourier Transform (STFT) of a time series signal using librosa library.

        Args:
            data (np.ndarray): The time series data.

        Returns:
            np.ndarray: The magnitude of the STFT (complex absolute value).
        """
        signal_fft = librosa.stft(data, n_fft=self.n_fft, hop_length=96) # 8192 = 46.8 ms
        signal_fft_abs = np.abs(signal_fft)
        return signal_fft_abs

    # Docstring for peak_extraction function
    def peak_extraction(self, signal_fft_abs, time_value, distance, width):
        """
        Extracts peaks from the magnitude of the STFT using scipy.signal.find_peaks function.

        Args:
            signal_fft_abs (np.ndarray): The magnitude of the STFT.
            time_value (list): Time t at which the peak picking is performed.
            distance (int): Minimum distance (in samples) between two peaks.
            width (int): Minimum prominence of a peak.

        Returns:
            list: List of peak indices in the STFT.
        """
        # modify the range function according to the time steps aimed at
        if len(time_value) == 1:
            extracted_peaks, _ = find_peaks (signal_fft_abs.T[time_value[0]], distance=distance, width=width)

        else:
            for time in time_value:
                extracted_peaks, _ = find_peaks (signal_fft_abs.T[time], distance=distance, width=width)
        return extracted_peaks

    # Docstring for data_processing_laplace function
    def data_processing_laplace(self, signal):
        """
        Performs data processing for Laplace analysis:
         - Loads data
         - Calculates STFT magnitude

        Args:
            signal (str): Path to the signal file.

        Returns:
            np.ndarray: The magnitude of the STFT.
        """
        data = self.load_data(signal)
        signal_fft = self.fft_librosa_laplace(data) # using librosa
        return signal_fft

    # Docstring for laplace_extraction function
    def laplace_extraction(self, file):
        """
        Extracts Laplace value from the filename based on a defined format.

        Args:
            file (str): Path to the signal file.

        Returns:
            list: List containing the extracted Laplace value (float).
        """
        file = str (file)
        parts = file.split ('_')
        laplace_value = []
        if len (parts) >= 5:
            # Adjust the index based on your file name structure
            desired_part = parts[3] #.split ('.')
        laplace_value.append (desired_part)
        laplace_value = [int (value.replace (',', '.')) for value in laplace_value]
        return laplace_value

    # Docstring for damping_extraction function
    def damping_extraction(self, file):
        """
        Extracts damping value from the filename based on a defined format (for plots only).

        Args:
            file (str): Path to the signal file.

        Returns:
            list: List containing the extracted damping value
        """
        file = str (file)
        parts = file.split ('_')
        damping_value = []
        if len (parts) >= 5:
            # Adjust the index based on your file name structure
            desired_part = parts[4].split ('.')[0]
        damping_value.append (desired_part)
        damping_value = [float(value.replace(',', '.')) for value in damping_value]
        return damping_value

    def analyze_spectrum(self, path_to_dataset, ht_length, vdf_frequencies):#, self.n_fft):
        laplace_variations_analysis_vector = []
        for length in ht_length:
            for freq in vdf_frequencies:
                for file in sorted (path_to_dataset.glob (f'FrameDrum_{length}_{freq}*.txt')):
                    fft_result = self.data_processing_laplace (file)
                    peaks_result = self.peak_extraction (fft_result,
                                                         c.TIME_VALUE,
                                                         distance=c.DISTANCE,
                                                         width=c.WIDTH)
                    laplace_variations_analysis_vector.append ({
                        "archivo": file.name,
                        "n_buff": length,
                        "Frequency": freq,
                        "LaPlace": self.laplace_extraction (file),
                        "Damping": self.damping_extraction (file),
                        "FFT": fft_result,
                        "Peaks": peaks_result
                    })
        return laplace_variations_analysis_vector