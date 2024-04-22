import librosa
from scipy.signal import find_peaks
import numpy as np
import constants as c


class Preprocessing:
    """Loads all functions for data preprocessing"""
    def __init__(self):

        self.n_fft = 4096
        self.fft_freqs = librosa.fft_frequencies(sr = c.SR, n_fft=self.n_fft) # 4096 gives results that approach the 1/4 Tone resolution

    # load time series
    def load_data(self, signal):
        data = np.loadtxt(signal)
        return data#[:24000]

    def fft_librosa_laplace(self, data):
        signal_fft = librosa.stft(data, n_fft=self.n_fft, hop_length=96) # 8192 = 46.8 ms
        signal_fft_abs = np.abs(signal_fft)
        return signal_fft_abs

    def peak_extraction(self, signal_fft_abs, time_value, distance, width):
        # modify the range function according to the time steps aimed at
        if len(time_value) == 1:
            extracted_peaks, _ = find_peaks (signal_fft_abs.T[time_value[0]], distance=distance, width=width)

        else:
            for time in time_value:
                extracted_peaks, _ = find_peaks (signal_fft_abs.T[time], distance=distance, width=width)
                # peaks_per_time.append (extracted_peaks)
        return extracted_peaks

    def data_processing_laplace(self, signal):
        data = self.load_data(signal)
        signal_fft = self.fft_librosa_laplace(data) # using librosa
        return signal_fft

    def laplace_extraction(self, file):
        file = str (file)
        parts = file.split ('_')
        laplace_value = []
        if len (parts) >= 5:
            # Adjust the index based on your file name structure
            desired_part = parts[3] #.split ('.')
        laplace_value.append (desired_part)
        laplace_value = [int (value.replace (',', '.')) for value in laplace_value]
        return laplace_value

    def damping_extraction(self, file): # only used for plots
        file = str (file)
        parts = file.split ('_')
        damping_value = []
        if len (parts) >= 5:
            # Adjust the index based on your file name structure
            desired_part = parts[4].split ('.')[0]
        damping_value.append (desired_part)
        damping_value = [float(value.replace(',', '.')) for value in damping_value]
        return damping_value

    def gammaconst_damping_extraction(self, file): # only used for plots
        file = str (file)
        parts = file.split ('_')
        Damping_value = []
        if len (parts) >= 9:
            desired_part = parts[6]#.split ('.')[0]  # Adjust the index based on your file name structure
        Damping_value.append (desired_part)
        return Damping_value


# def analyze_spectrum(path_to_dataset, ht_length, vdf_frequencies):
#     laplace_variations_analysis_vector = []
#     for length in ht_length:
#         for freq in vdf_frequencies:
#             for file in sorted (path_to_dataset.glob (f'FrameDrum_{length}_{freq}*.txt')):
#                 fft_result = data_processing_laplace (file)
#                 peaks_result = peak_extraction (fft_result,
#                                                   time_value,
#                                                   distance=c.DISTANCE,
#                                                   width=c.WIDTH)
#                 laplace_variations_analysis_vector.append ({
#                     "archivo": file.name,
#                     "n_buff": length,
#                     "Frequency": freq,
#                     "LaPlace": laplace_extraction (file),
#                     "Damping": damping_extraction (file),
#                     "FFT": fft_result,
#                     "Peaks": peaks_result
#                 })
#     return laplace_variations_analysis_vector