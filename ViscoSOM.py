from Functions import Preprocessing
from pathlib import Path
import constants as c

f = Preprocessing ()
def analyze_spectrum(path_to_dataset, ht_length, vdf_frequencies, n_fft):
    laplace_variations_analysis_vector = []
    for length in ht_length:
        for freq in vdf_frequencies:
            for file in sorted (path_to_dataset.glob (f'FrameDrum_{length}_{freq}*.txt')):
                fft_result = f.data_processing_laplace (signal=file)
                peaks_result = f.peak_extraction (fft_result,
                                                  time_value,
                                                  distance=c.DISTANCE,
                                                  width=c.WIDTH)
                laplace_variations_analysis_vector.append ({
                    "archivo": file.name,
                    "n_buff": length,
                    "Frequency": freq,
                    "LaPlace": f.laplace_extraction (file),
                    "Damping": f.damping_extraction (file),
                    "FFT": fft_result,
                    "Peaks": peaks_result
                })
    return laplace_variations_analysis_vector

def extract_peaks_from_fr(analysis_vector):
    # Create an empty list to store dictionaries for DataFrame
    peaks_in_freq_range = []
    # Iterate over laplace_variations_analysis_vector
    for laplace_result in analysis_vector:
        # if laplace_result['n_buff'] == n_buff and laplace_result['Frequency'] == VDF:
        if laplace_result['Frequency'] == VDF:
            for peaks in laplace_result["Peaks"]:
                if min_freq < f.fft_freqs[peaks] < max_freq:
                    peaks_in_freq_range.append (f.fft_freqs[peaks])
    return peaks_in_freq_range

def extract_peaks_info(sorted_peaks):
    # [0] == peak frequencies, [1] == peak indices
    peaks_info = [[], []]
    for peaks in sorted_peaks:
        for idx, hz in enumerate (list (f.fft_freqs)):
            if peaks == hz:
                # print(peaks, hz, idx)
                peaks_info[0].append (hz)
                peaks_info[1].append (list (f.fft_freqs).index (peaks))
    return peaks_info

def create_feature_vector(analysis_vector):
    # global feature_vector_dict
    # create a key for each frequency inside the frequency range
    # feature_vector_dict = {value: [] for value in peaks_info[0]}
    # # add the keys "Damping" and "LaPlace"
    # feature_vector_dict["Damping"] = []
    # feature_vector_dict["LaPlace"] = []
    # feature_vector_dict["buff"] = []

    # #tryout
    # feature_vector_dict = dict.fromkeys (peaks_info, [])
    feature_vector_dict = {tuple (row): [] for row in peaks_info}
    feature_vector_dict.update ({"Damping": [], "LaPlace": [], "buff": []})

    # extract and append values to the dictionary
    # for idx, laplace_result in enumerate (analysis_vector):
    #     # TODO: create the case for a specific n_buff
    #     # if laplace_result['n_buff'] == n_buff and laplace_result['Frequency'] == VDF:
    #     if laplace_result['Frequency'] == VDF:
    #         feature_vector_dict['Damping'].append (laplace_result['Damping'][0])
    #         feature_vector_dict['LaPlace'].append (laplace_result['LaPlace'][0])
    #         feature_vector_dict['buff'].append (laplace_result['n_buff'])
    #
    #         for peak_freqs, peak_index in zip (peaks_info[0], peaks_info[1]):
    #             key_to_append = peak_freqs
    #
    #             if key_to_append in feature_vector_dict:
    #                 feature_vector_dict[key_to_append].append (laplace_result["FFT"][peak_index][time_value][0])
    # return feature_vector_dict


    for idx, laplace_result in enumerate(analysis_vector):
        # Filter based on n_buff if needed (uncomment and define logic)
        # if laplace_result['n_buff'] == n_buff:

        if laplace_result['Frequency'] == VDF:
            feature_vector_dict['Damping'].append(laplace_result['Damping'][0])
            feature_vector_dict['LaPlace'].append(laplace_result['LaPlace'][0])
            feature_vector_dict['buff'].append(laplace_result['n_buff'])

            # Append peak values using dictionary.get and list comprehension
            feature_vector_dict.update({peak_freqs: laplace_result["FFT"][peak_index][time_value][0] for peak_freqs, peak_index in zip(peaks_info[0], peaks_info[1])})
        else:
            # Exit loop if Frequency doesn't match VDF
            break

    return feature_vector_dict



# Parameters of the simulation
ht_length = [384, 672, 960]
vdf_frequencies = [700, 2000, 3300, 6400, 12000]
directory_path_laplace = Path (r'C:\Users\LEGION\PycharmProjects\pythonProject\venv\Laplace variations')
directory_path_gammaconst = Path (r'C:\Users\LEGION\PycharmProjects\pythonProject\venv\gammaconst')

# Parameters of the pipeline

time_value = [60]  # time t at which peak picking is performed
# n_buff = 672
VDF = 12000
frequency_range = 1000 / 2
n_fft = 4096

# Specify the frequency range
min_freq = VDF - frequency_range
max_freq = VDF + frequency_range

"""Run STFT and Peak Extraction"""
laplace_variations_analysis_vector = analyze_spectrum (path_to_dataset=directory_path_laplace,
                                                       ht_length=ht_length,
                                                       vdf_frequencies=vdf_frequencies,
                                                       n_fft=n_fft)

# FEATURE VECTOR
peaks_in_freq_range = extract_peaks_from_fr (laplace_variations_analysis_vector)

# erasing duplicated peaks and sorting numerically
sorted_peaks = list(sorted(dict.fromkeys(peaks_in_freq_range)))

# extract freq. and amp. from each peak
peaks_info = extract_peaks_info (sorted_peaks)

# create the dictionary for the feature vector
feature_vector_dict = create_feature_vector (laplace_variations_analysis_vector)

#%%

"""
# # Create a DataFrame from the list of dictionaries
feature_vector_dataframe = pd.DataFrame(feature_vector_dict)

# SOM
target1 = np.array(feature_vector_dataframe['Damping'].values)
target2 = np.array(feature_vector_dataframe['LaPlace'].values)
target3 = np.array(feature_vector_dataframe['buff'].values)

labels = list(sorted(set(feature_vector_dataframe['Damping'].values)))
data = feature_vector_dataframe[feature_vector_dataframe.columns[:-3]]

# data normalization
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values"""


print(feature_vector_dict)
