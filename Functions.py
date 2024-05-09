import librosa
from scipy.signal import find_peaks
import constants as c
import numpy as np


class Preprocessing:
    """Loads all functions for data preprocessing"""
    def __init__(self, n_fft):
        self.n_fft = n_fft
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

#
# class CreateVector:
#     def __init__(self, analysis_vector, n_buff, VDF, min_freq, max_freq):
#         self.analysis_vector = analysis_vector
#         # self.ht_analysis = ht_analysis
#         self.n_buff = n_buff
#         self.VDF = VDF
#         self.min_freq = min_freq
#         self.max_freq = max_freq
#
#     def prep_fft_freqs(self):
#         # Create an instance of ClassA
#         f = Preprocessing ()
#
#         # Call the method from ClassA using the instance
#         # f.fft_freqs ()
#
#     def extract_peaks_from_fr(self, analysis_vector, VDF, min_freq, max_freq, f):
#         """
#         Extracts peaks within a specific frequency range from Laplace analysis results.
#
#         Args:
#             analysis_vector (list): List of dictionaries containing Laplace analysis results for each signal file.
#             ht_analysis (bool): if True, only the time integration defined by n_buff will be analyzed. If False, all T are analyzed.
#             n_buff (int): Target integration time to consider (if length_ht is True). 384 = 4ms, 672 = 7ms, 960 = 10ms.
#             VDF (int): Viscoelatically Damped Frequency.
#             min_freq (float): Lower bound of the frequency range to extract peaks from.
#             max_freq (float): Upper bound of the frequency range to extract peaks from.
#             f (Preprocessing object): An instance of the Preprocessing class.
#
#         Returns:
#             list: List of peak frequencies within the specified range for each analysis result.
#         """
#         peaks_in_freq_range = []
#         for laplace_result in analysis_vector:
#             # Check for VDF and handle length_ht condition in one step
#             if laplace_result['Frequency'] == VDF:
#                 for peaks in laplace_result["Peaks"]:
#                     if min_freq < f.fft_freqs[peaks] < max_freq:
#                         peaks_in_freq_range.append (f.fft_freqs[peaks])
#         return peaks_in_freq_range
#
#
#     def extract_peaks_info(self, sorted_peaks, f):
#         # TODO: HOW TO REPLACE F?
#         """
#         Extracts information about peaks from a sorted list of frequencies.
#
#         Args:
#             sorted_peaks (list): A list of frequencies in ascending order.
#             f (Preprocessing object): An instance of the Preprocessing class.
#
#         Returns:
#             list: A list containing two sub-lists:
#                   - The first sub-list contains the extracted peak frequencies.
#                   - The second sub-list contains the corresponding indices in f.fft_freqs.
#         """
#         peaks_info = [[], []]
#         for peaks in sorted_peaks:
#             for idx, hz in enumerate (list (f.fft_freqs)):
#                 if peaks == hz:
#                     peaks_info[0].append (hz)
#                     peaks_info[1].append (list (f.fft_freqs).index (peaks))
#         return peaks_info
#
#     def create_feature_vector(self, analysis_vector, peaks_info, time_value, n_buff, ht_analysis):
#       """
#       This function creates a dictionary containing feature vectors for Self-Organizing Map (SOM) training.
#
#       Args:
#           analysis_vector (list): A list of dictionaries containing analysis results for each integration time.
#               Each dictionary should have keys like 'Frequency', 'Damping', 'LaPlace', and 'FFT'.
#           peaks_info (tuple): A tuple containing two lists:
#               - The first list contains peak frequencies identified in the analysis.
#               - The second list contains corresponding peak amplitudes from the STFT data.
#           time_value (int): The time step index to use from the STFT data (e.g., for a specific time window).
#           n_buff (int): The specific integration time to be analyzed (relevant if ht_analysis is True).
#           ht_analysis (bool): A flag indicating whether to consider all integration times (False)
#                               or only the one specified by n_buff (True).
#
#       Returns:
#           dict: A dictionary containing feature vectors for SOM training.
#               - Keys represent features like 'Damping', 'LaPlace', 'buff', and peak frequencies.
#               - Values are lists containing the corresponding feature values for each data point.
#       """
#
#       # Initialize a dictionary to store feature vectors for different keys
#       feature_vector_dict = {value: [] for value in peaks_info[0]}
#
#       # Add metadata keys for SOM training (Damping, LaPlace, buff)
#       som_metadata = ["Damping", "LaPlace", "buff"]
#       feature_vector_dict.update({key: [] for key in som_metadata})
#
#       # Loop through each analysis result in the vector
#       for idx, laplace_result in enumerate(analysis_vector):
#         # Check if the current analysis result has the desired frequency (VDF)
#         if laplace_result['Frequency'] == VDF:
#           # Branch based on ht_analysis flag
#           if not ht_analysis:
#             # Consider all buffer sizes if ht_analysis is False
#             # Append damping, laplace, and buffer size for this data point
#             feature_vector_dict['Damping'].append(laplace_result['Damping'][0])
#             feature_vector_dict['LaPlace'].append(laplace_result['LaPlace'][0])
#             feature_vector_dict['buff'].append(laplace_result['n_buff'])
#
#             # Append peak values based on peak frequencies and indices
#             for peak_freqs, peak_index in zip(peaks_info[0], peaks_info[1]):
#               key_to_append = peak_freqs  # Use peak frequency as the key
#               if key_to_append in feature_vector_dict:
#                 # Access the specific peak value at the desired time step
#                 feature_vector_dict[key_to_append].append(laplace_result["FFT"][peak_index][time_value][0])
#
#           elif laplace_result['n_buff'] == n_buff:
#             # Consider only the specified buffer size (n_buff) if ht_analysis is True
#             # Similar logic as the previous block for appending data
#             feature_vector_dict['Damping'].append(laplace_result['Damping'][0])
#             feature_vector_dict['LaPlace'].append(laplace_result['LaPlace'][0])
#             feature_vector_dict['buff'].append(laplace_result['n_buff'])
#
#             for peak_freqs, peak_index in zip(peaks_info[0], peaks_info[1]):
#               key_to_append = peak_freqs
#               if key_to_append in feature_vector_dict:
#                 feature_vector_dict[key_to_append].append(laplace_result["FFT"][peak_index][time_value][0])
#
#       # Return the dictionary containing the created feature vectors
#       return feature_vector_dict
#
#     def dict_for_som(self, analysis_vector,
#                      ht_analysis=False, n_buff=None, VDF=None,
#                      min_freq=None, max_freq=None, f=None, time_value=None):
#       """
#       This function prepares a dictionary containing feature vectors for Self-Organizing Map (SOM) training.
#
#       Args:
#           analysis_vector (list): A list of dictionaries containing analysis results for each buffer size.
#               Each dictionary should have keys like 'Frequency', 'Damping', 'LaPlace', and 'FFT'.
#           ht_analysis (bool, optional): A flag indicating whether to consider all buffer sizes (False)
#                                           or only the specified buffer size (True) (default: False).
#           n_buff (int, optional): The specific buffer size of interest (relevant if ht_analysis is True) (default: None).
#           VDF (float, optional): The target resonant frequency (VDF) for peak identification (default: None).
#           min_freq (float, optional): The minimum frequency in the considered frequency range (default: None).
#           max_freq (float, optional): The maximum frequency in the considered frequency range (default: None).
#           f (list, optional): A list containing the original frequencies used in the analysis (default: None).
#           time_value (int, optional): The time step index to use from the 'FFT' data (e.g., for a specific time window) (default: None).
#
#       Returns:
#           dict: A dictionary containing feature vectors for SOM training.
#               - Keys represent features like 'Damping', 'LaPlace', 'buff', and peak frequencies.
#               - Values are lists containing the corresponding feature values for each data point.
#
#       Raises:
#           ValueError: If any of the required arguments (n_buff, VDF, min_freq, max_freq, f, time_value) are None.
#       """
#
#       # Check for missing required arguments
#       if None in (n_buff, VDF, min_freq, max_freq, f, time_value):
#           raise ValueError("Missing required arguments.")
#
#       # Step 1: Extract peaks within the desired frequency range
#       peaks_in_freq_range = extract_peaks_from_fr(analysis_vector=analysis_vector,
#                                                    ht_analysis=ht_analysis,
#                                                    n_buff=n_buff,
#                                                    VDF=VDF,
#                                                    min_freq=min_freq,
#                                                    max_freq=max_freq,
#                                                    f=f)
#       # Assuming extract_peaks_from_fr returns a list of peak values
#
#       # Step 2: Remove duplicate peaks and sort them numerically
#       sorted_peaks = list(sorted(dict.fromkeys(peaks_in_freq_range)))
#
#       # Step 3: Extract frequency and amplitude information for each peak
#       peaks_info = extract_peaks_info(sorted_peaks, f)
#       # Assuming extract_peaks_info takes a list of peaks and a list of frequencies (f) as input
#       # and returns a data structure (e.g., tuple) containing peak frequencies and indices
#
#       # Step 4: Create the feature vector dictionary using the extracted information
#       feature_vector_dict = create_feature_vector(analysis_vector=analysis_vector,
#                                                    peaks_info=peaks_info,
#                                                    time_value=time_value,
#                                                    ht_analysis=ht_analysis,
#                                                    n_buff=n_buff)
#
#       # Return the dictionary containing the feature vectors
#       return feature_vector_dict
