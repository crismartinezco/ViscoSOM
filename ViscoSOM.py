from Functions import Preprocessing
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from minisom import MiniSom

def extract_peaks_from_fr(analysis_vector, length_ht, n_buff, VDF, min_freq, max_freq, f):
    """
    Extracts peaks within a specific frequency range from Laplace analysis results.

    Args:
        analysis_vector (list): List of dictionaries containing Laplace analysis results for each signal file.
        length_ht (bool): if True, only the time integration defined by n_buff will be analyzed. If False, all T are analyzed.
        n_buff (int): Target integration time to consider (if length_ht is True). 384 = 4ms, 672 = 7ms, 960 = 10ms.
        VDF (int): Viscoelatically Damped Frequency.
        min_freq (float): Lower bound of the frequency range to extract peaks from.
        max_freq (float): Upper bound of the frequency range to extract peaks from.
        f (Preprocessing object): An instance of the Preprocessing class.

    Returns:
        list: List of peak frequencies within the specified range for each analysis result.
    """
    peaks_in_freq_range = []
    for laplace_result in analysis_vector:
        # Check for VDF and handle length_ht condition in one step
        if laplace_result['Frequency'] == VDF and (length_ht is False or laplace_result['n_buff'] == n_buff):
            for peaks in laplace_result["Peaks"]:
                if min_freq < f.fft_freqs[peaks] < max_freq:
                    peaks_in_freq_range.append(f.fft_freqs[peaks])
    return peaks_in_freq_range


def extract_peaks_info(sorted_peaks, f):
    """
    Extracts information about peaks from a sorted list of frequencies.

    Args:
        sorted_peaks (list): A list of frequencies in ascending order.
        f (Preprocessing object): An instance of the Preprocessing class.

    Returns:
        list: A list containing two sub-lists:
              - The first sub-list contains the extracted peak frequencies.
              - The second sub-list contains the corresponding indices in f.fft_freqs.
    """
    peaks_info = [[], []]
    for peaks in sorted_peaks:
        for idx, hz in enumerate(list(f.fft_freqs)):
            if peaks == hz:
                peaks_info[0].append(hz)
                peaks_info[1].append(list(f.fft_freqs).index(peaks))
    return peaks_info


def create_feature_vector(analysis_vector, peaks_info, time_value, n_buff):
    """
    Creates a feature vector dictionary from Laplace analysis results.

    Args:
        analysis_vector (list): List of dictionaries containing Laplace analysis results for each signal file.
        peaks_info (list): List containing peak frequencies and their corresponding indices in f.fft_freqs.
        time_value (int): Time t at which the peak picking is performed.
        n_buff (int): Target integration time to consider (if length_ht is True). 384 = 4ms, 672 = 7ms, 960 = 10ms.

    Returns:
        dict: A dictionary containing features for each analysis result.
    """
    # Define keys based on frequency values in the peaks_info list
    feature_vector_dict = {value: [] for value in peaks_info[0]}

    # Add extra keys for Damping, LaPlace, and buff
    som_metadata = ["Damping", "LaPlace", "buff"]
    feature_vector_dict.update({key: [] for key in som_metadata})

    # Extract and append values to the dictionary
    for idx, laplace_result in enumerate(analysis_vector):
        # Check for n_buff and VDF
        if laplace_result['n_buff'] == n_buff and laplace_result['Frequency'] == VDF:
            feature_vector_dict['Damping'].append(laplace_result['Damping'][0])
            feature_vector_dict['LaPlace'].append(laplace_result['LaPlace'][0])
            feature_vector_dict['buff'].append(laplace_result['n_buff'])

            for peak_freqs, peak_index in zip(peaks_info[0], peaks_info[1]):
                key_to_append = peak_freqs
                if key_to_append in feature_vector_dict:
                    feature_vector_dict[key_to_append].append(laplace_result["FFT"][peak_index][time_value][0])
    return feature_vector_dict

def create_vector_for_som():
    global feature_vector_dataframe
    # FEATURE VECTOR
    peaks_in_freq_range = extract_peaks_from_fr (laplace_variations_analysis_vector, length_ht=False, n_buff=n_buff,
                                                 VDF=VDF, min_freq=min_freq, max_freq=max_freq, f=f)
    # erasing duplicated peaks and sorting numerically
    sorted_peaks = list (sorted (dict.fromkeys (peaks_in_freq_range)))
    # extract freq. and amp. from each peak
    peaks_info = extract_peaks_info (sorted_peaks, f)
    # create the dictionary for the feature vector
    feature_vector_dict = create_feature_vector (laplace_variations_analysis_vector, peaks_info,
                                                 time_value=time_value, n_buff=n_buff)
    # Create a DataFrame from the feature vector dict
    feature_vector_dataframe = pd.DataFrame (feature_vector_dict)

    return feature_vector_dataframe
f = Preprocessing ()

# Parameters of the simulation
ht_length = [384, 672, 960]
vdf_frequencies = [700, 2000, 3300, 6400, 12000]
directory_path_laplace = Path (r'C:\Users\LEGION\PycharmProjects\pythonProject\venv\Laplace variations')
directory_path_gammaconst = Path (r'C:\Users\LEGION\PycharmProjects\pythonProject\venv\gammaconst')

"""Run STFT and Peak Extraction"""
laplace_variations_analysis_vector = f.analyze_spectrum (path_to_dataset=directory_path_laplace,
                                                       ht_length=ht_length,
                                                       vdf_frequencies=vdf_frequencies)

# Parameters of the pipeline

time_value = [60]  # time t at which peak picking is performed
n_buff = 960 # integration time T
VDF = 12000
frequency_range = 1000 / 2 # FR of the analysis Freq. Range
n_fft = 4096 # Window size of the n_fft

# Specify the frequency range
min_freq = VDF - frequency_range
max_freq = VDF + frequency_range

feature_vector_dataframe = create_vector_for_som ()

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
unique_values = np.unique(target1)

for idx, c in enumerate(unique_values):
    idx_target = target1 == c
    color = cmap_modified (idx / len (unique_values))

    # Define size based on target2 values (scale and normalize)
    size_values = (target2[idx_target] - np.min (target2[idx_target])) / \
                  (np.max (target2[idx_target]) - np.min (target2[idx_target])) * 300  # Adjust max_size as needed
    plt.scatter(w_x[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .5,
                w_y[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .8,
                s=size_values, marker='o', color=color, label=f'{c}', alpha=1, edgecolors='black', linewidths=1)

plt.yticks([])
plt.xticks ([])
plt.tight_layout()
# plt.savefig('SOM LaPlace and Damping.png', dpi=600)
plt.show()
plt.close()



#%%
# def plot_som(length_ht, legend, title, ticks, save, umatrix_colorbar, markers):
#     # global c
#
#     if legend:
#         figsize = (10, 11)
#         bottom_adjust = 0.1
#     else:
#         figsize = (10, 10)
#         bottom_adjust = 0.05
#
#     plt.figure (figsize=figsize)
#     plt.subplots_adjust (left=0.05, right=0.95, bottom=bottom_adjust, top=0.95)
#
#     plt.pcolor (som.distance_map ().T, cmap=cmapcolor, alpha=alpha)
#
#     if umatrix_colorbar:
#         plt.colorbar()
#
#     if length_ht:
#         unique_values = np.unique (target3)
#         colors = ['red', 'blue', 'yellow']
#
#         for idx, c in enumerate (unique_values):
#             idx_target = target3 == c
#
#             if c == 384:
#                 # marker = markers[0]
#                 label = f'4ms'
#             elif c == 672:
#                 # marker = markers[1]
#                 label = f'7ms'
#             elif c == 960:
#                 # marker = markers[2]
#                 label = f'10ms'
#             else:
#                 marker = 'o'  # Default marker if target3 doesn't match any condition
#
#             color = colors[idx]
#
#     else:
#         # color according to Re
#         unique_values = np.unique (target1)
#         for idx, c in enumerate (unique_values):
#             idx_target = target1 == c
#             color = cmap_modified (idx / len (unique_values))
#             label = f'{c}'
#             marker = 'o'
#
#     size_values = (target2[idx_target] - np.min (target2[idx_target])) / \
#                   (np.max (target2[idx_target]) - np.min (target2[idx_target])) * 300  # Adjust max_size as needed
#
#     plt.scatter(w_x[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .5,
#                 w_y[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .8,
#                 s=size_values, marker='o', color=color, label=f'{c}', alpha=1, edgecolors='black', linewidths=1)
#
#     # # Define size based on target2 values (scale and normalize)
#     # size_values = (target2[idx_target] - np.min (target2[idx_target])) / \
#     #               (np.max (target2[idx_target]) - np.min (target2[idx_target])) * 300  # Adjust max_size
#     #
#     # plt.scatter (w_x[idx_target] + .5 + (np.random.rand (np.sum (idx_target)) - .5) * .5,
#     #              w_y[idx_target] + .5 + (np.random.rand (np.sum (idx_target)) - .5) * .8,
#     #              s=size_values, marker=marker, color=color, label=label, alpha=1, edgecolors='black', linewidths=1)
#
#     if legend:
#         plt.legend (loc='upper center', bbox_to_anchor=(0.55, -0.03), ncol=6, fontsize='xx-large')
#
#     if legend and length_ht:
#         plt.legend (loc='upper center', bbox_to_anchor=(0.55, -0.03), ncol=3, fontsize='xx-large')
#
#     if legend and length_ht and markers:
#         handles, labels = plt.gca ().get_legend_handles_labels ()
#         by_label = dict (zip (labels, handles))
#         plt.legend (by_label.values (), by_label.keys (), loc='upper center', bbox_to_anchor=(0.55, -0.03), ncol=3,
#                     fontsize='xx-large')
#
#     if title:
#         plt.title(f'{VDF}Hz - FR: {min_freq}Hz - {max_freq}Hz - t = {time_value[0]}ms')
#
#     if title and length_ht:
#         plt.title (f'{VDF}Hz - FR: {min_freq}Hz - {max_freq}Hz - t = {time_value[0]}ms - INTEGRATION TIME ANALYSIS')
#     # plt.grid()
#
#     if ticks:
#         plt.yticks (range (0, n_neurons + 1, 5), fontsize='x-large')
#         plt.xticks (range (0, n_neurons + 1, 5), fontsize='x-large')
#
#     else:
#         plt.yticks([])
#         plt.xticks([])
#
#     plt.tight_layout ()
#     if save:
#         plt.savefig ('SOM Re and Gamma.png', dpi=300)
#     plt.show ()
#     plt.close ()
#
# plot_som (length_ht=False, legend=True, title=True, ticks=True, save=False, umatrix_colorbar=False, markers=False)

# def plot_som(legend):
#     # global c
#
#     if legend:
#         figsize = (10, 11)
#         bottom_adjust = 0.1
#     else:
#         figsize = (10, 10)
#         bottom_adjust = 0.05
#
#     plt.figure (figsize=figsize)
#     plt.subplots_adjust (left=0.05, right=0.95, bottom=bottom_adjust, top=0.95)
#
#     plt.pcolor (som.distance_map ().T, cmap=cmapcolor, alpha=alpha)
#
#     unique_values = np.unique(target1)
#
#     for idx, c in enumerate(unique_values):
#         idx_target = target1 == c
#
#         color = cmap_modified (idx / len (unique_values))
#
#         # Define size based on target2 values (scale and normalize)
#         size_values = (target2[idx_target] - np.min (target2[idx_target])) / \
#                       (np.max (target2[idx_target]) - np.min (target2[idx_target])) * 300  # Adjust max_size as needed
#         plt.scatter(w_x[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .5,
#                     w_y[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .8,
#                     s=size_values, marker='o', color=color, label=f'{c}', alpha=1, edgecolors='black', linewidths=1)
#     if legend:
#         plt.legend (loc='upper center', bbox_to_anchor=(0.55, -0.03), ncol=6, fontsize='large')
#
#     plt.tight_layout()
#     # plt.savefig('SOM LaPlace and Damping.png', dpi=600)
#     plt.show()
#     plt.close()
#
#
# plot_som(legend=True)

# Plot the contour of the spectrum of the feature vector

# win_map = som.win_map(data)
#
# plt.figure(figsize=(10, 10))
# # plt.pcolor(som.distance_map().T, cmap=cmapcolor, alpha=0.5)
# # plt.subplots_adjust(left=0.05, right=0.98, bottom=0.03, top=0.99, wspace=0.6, hspace=0.3)
# plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
# plt.yticks(range(0, n_neurons+1, 5), fontsize = 'x-large')
# plt.xticks(range(0, n_neurons+1, 5), fontsize = 'x-large')
#
# the_grid = GridSpec(m_neurons, n_neurons)
# for position in win_map.keys():
#     row, col = m_neurons-1-position[1], position[0]
#     ax = plt.subplot(the_grid[row, col])
#
#     # plt.plot(np.min(win_map[position], axis=0), color='gray', alpha=.5)
#     # plt.plot(np.max(win_map[position], axis=0), color='gray', alpha=.5)
#     plt.plot (np.mean (win_map[position], axis=0), linewidth=1)
#     # plt.yticks (fontsize='small')
#     plt.yticks([])
#     plt.xticks ([])
#
# plt.tight_layout()
# plt.savefig('time series.png', dpi=600)
# plt.show()
# plt.close()