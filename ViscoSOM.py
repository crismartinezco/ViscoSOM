from Functions import Preprocessing
from pathlib import Path
import constants as c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from minisom import MiniSom

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
    feature_vector_dict = {value: [] for value in peaks_info[0]}

    # add the keys "Damping" and "LaPlace"
    feature_vector_dict["Damping"] = []
    feature_vector_dict["LaPlace"] = []
    feature_vector_dict["buff"] = []

    # extract and append values to the dictionary
    for idx, laplace_result in enumerate (analysis_vector):
        # TODO: create the case for a specific n_buff
        # if laplace_result['n_buff'] == n_buff and laplace_result['Frequency'] == VDF:
        if laplace_result['Frequency'] == VDF:
            feature_vector_dict['Damping'].append (laplace_result['Damping'][0])
            feature_vector_dict['LaPlace'].append (laplace_result['LaPlace'][0])
            feature_vector_dict['buff'].append (laplace_result['n_buff'])

            for peak_freqs, peak_index in zip (peaks_info[0], peaks_info[1]):
                key_to_append = peak_freqs

                if key_to_append in feature_vector_dict:
                    feature_vector_dict[key_to_append].append (laplace_result["FFT"][peak_index][time_value][0])
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

# Create a DataFrame from the feature vector dict
feature_vector_dataframe = pd.DataFrame(feature_vector_dict)

# The last three columns correspond to metadata for map calibration
data = feature_vector_dataframe[feature_vector_dataframe.columns[:-3]]

# data normalization
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
data = data.values

# CREATING THE SOM - CALIBRATION

target1 = np.array(feature_vector_dataframe['Damping'].values) # Re values
target2 = np.array(feature_vector_dataframe['LaPlace'].values) # Gamma values
target3 = np.array(feature_vector_dataframe['buff'].values) # Integration time / length of h(T)
labels = list(sorted(set(feature_vector_dataframe['Damping'].values)))

# U-matrix colors
cmapcolor = 'bone'
alpha = 1

# Initialization and training
n_neurons = 20
m_neurons = 20
nf = 'gaussian'
som = MiniSom(n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5,
              neighborhood_function=nf, random_seed=0, topology='rectangular')

som.pca_weights_init(data)
som.train(data, 1000, verbose=True)  # training

# create the U-matrix
u_matrix = som.distance_map ()
# ww = som.get_weights ()

# coordenates for the scatter plot
w_x, w_y = zip(*[som.winner(d) for d in data])
w_x = np.array(w_x)
w_y = np.array(w_y)

"""Calibration according to targets.
- The higher the value of Re, the darker the color of the data point.
- The smaller the value of gamma, the bigger the size of the data point.
"""

# Define the colormap range and colors
cmap = plt.cm.viridis_r
colors = cmap(np.linspace(0, 1, 256))

# Set the color for the Reference Cases (Re = 0)
colors[0] = [0.5, 0.5, 0.5, 1.0]  # Gray color for the 0 value

# Create a new colormap with the modified colors
cmap_modified = mcolors.ListedColormap(colors)

plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap=cmapcolor, alpha=alpha)
# Eliminate white space around plot elements

# without legend
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

# with legend
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.1, top=0.95)
# plt.colorbar()

unique_values = np.unique(target1)

for idx, c in enumerate(unique_values):
    idx_target = target1 == c

    # Choose a color from the Viridis colormap
    color = cmap_modified (idx / len (unique_values))

    # Define size based on target2 values (scale and normalize)
    size_values = (target2[idx_target] - np.min (target2[idx_target])) / \
                  (np.max (target2[idx_target]) - np.min (target2[idx_target])) * 300  # Adjust max_size as needed
    plt.scatter(w_x[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .5,
                w_y[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .8,
                s=size_values, marker='o', color=color, label=f'{c}', alpha=1, edgecolors='black', linewidths=1)

# plt.title(f'VDF: 700Hz - FR: {min_freq}Hz - {max_freq}Hz - {time_value[0]}ms')
# plt.grid()
plt.yticks(range(0, n_neurons+1, 5), fontsize = 'x-large')
plt.xticks(range(0, n_neurons+1, 5), fontsize = 'x-large')

# plt.yticks([])
# plt.xticks ([])

# plt.legend(loc='upper center', bbox_to_anchor=(0.55, -0.03), ncol=6, fontsize = 'large')
plt.tight_layout()
plt.savefig('SOM LaPlace and Damping.png', dpi=300)
plt.show()
plt.close()

# Plot the contour of the spectrum of the feature vector

win_map = som.win_map(data)

plt.figure(figsize=(10, 10))
# plt.pcolor(som.distance_map().T, cmap=cmapcolor, alpha=0.5)
# plt.subplots_adjust(left=0.05, right=0.98, bottom=0.03, top=0.99, wspace=0.6, hspace=0.3)
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
plt.yticks(range(0, n_neurons+1, 5), fontsize = 'x-large')
plt.xticks(range(0, n_neurons+1, 5), fontsize = 'x-large')

the_grid = GridSpec(m_neurons, n_neurons)
for position in win_map.keys():
    row, col = m_neurons-1-position[1], position[0]
    ax = plt.subplot(the_grid[row, col])

    # plt.plot(np.min(win_map[position], axis=0), color='gray', alpha=.5)
    # plt.plot(np.max(win_map[position], axis=0), color='gray', alpha=.5)
    plt.plot (np.mean (win_map[position], axis=0), linewidth=1)
    # plt.yticks (fontsize='small')
    plt.yticks([])
    plt.xticks ([])

plt.tight_layout()
plt.savefig('time series.png', dpi=600)
plt.show()
plt.close()