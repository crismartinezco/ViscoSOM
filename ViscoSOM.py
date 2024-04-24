import constants
from Functions import Preprocessing
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from minisom import MiniSom


def extract_peaks_from_fr(analysis_vector, ht_analysis, n_buff, VDF, min_freq, max_freq, f):
    """
    Extracts peaks within a specific frequency range from Laplace analysis results.

    Args:
        analysis_vector (list): List of dictionaries containing Laplace analysis results for each signal file.
        ht_analysis (bool): if True, only the time integration defined by n_buff will be analyzed. If False, all T are analyzed.
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
        if laplace_result['Frequency'] == VDF:
            # if laplace_result['Frequency'] == VDF and ht_analysis is False or (ht_analysis is True and laplace_result['n_buff'] == n_buff):
            for peaks in laplace_result["Peaks"]:
                if min_freq < f.fft_freqs[peaks] < max_freq:
                    peaks_in_freq_range.append (f.fft_freqs[peaks])
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
        for idx, hz in enumerate (list (f.fft_freqs)):
            if peaks == hz:
                peaks_info[0].append (hz)
                peaks_info[1].append (list (f.fft_freqs).index (peaks))
    return peaks_info

def create_feature_vector(analysis_vector, peaks_info, time_value, n_buff, ht_analysis):
  """
  This function creates a dictionary containing feature vectors for Self-Organizing Map (SOM) training.

  Args:
      analysis_vector (list): A list of dictionaries containing analysis results for each integration time.
          Each dictionary should have keys like 'Frequency', 'Damping', 'LaPlace', and 'FFT'.
      peaks_info (tuple): A tuple containing two lists:
          - The first list contains peak frequencies identified in the analysis.
          - The second list contains corresponding peak amplitudes from the STFT data.
      time_value (int): The time step index to use from the STFT data (e.g., for a specific time window).
      n_buff (int): The specific integration time to be analyzed (relevant if ht_analysis is True).
      ht_analysis (bool): A flag indicating whether to consider all integration times (False)
                          or only the one specified by n_buff (True).

  Returns:
      dict: A dictionary containing feature vectors for SOM training.
          - Keys represent features like 'Damping', 'LaPlace', 'buff', and peak frequencies.
          - Values are lists containing the corresponding feature values for each data point.
  """

  # Initialize a dictionary to store feature vectors for different keys
  feature_vector_dict = {value: [] for value in peaks_info[0]}

  # Add metadata keys for SOM training (Damping, LaPlace, buff)
  som_metadata = ["Damping", "LaPlace", "buff"]
  feature_vector_dict.update({key: [] for key in som_metadata})

  # Loop through each analysis result in the vector
  for idx, laplace_result in enumerate(analysis_vector):
    # Check if the current analysis result has the desired frequency (VDF)
    if laplace_result['Frequency'] == VDF:
      # Branch based on ht_analysis flag
      if not ht_analysis:
        # Consider all buffer sizes if ht_analysis is False
        # Append damping, laplace, and buffer size for this data point
        feature_vector_dict['Damping'].append(laplace_result['Damping'][0])
        feature_vector_dict['LaPlace'].append(laplace_result['LaPlace'][0])
        feature_vector_dict['buff'].append(laplace_result['n_buff'])

        # Append peak values based on peak frequencies and indices
        for peak_freqs, peak_index in zip(peaks_info[0], peaks_info[1]):
          key_to_append = peak_freqs  # Use peak frequency as the key
          if key_to_append in feature_vector_dict:
            # Access the specific peak value at the desired time step
            feature_vector_dict[key_to_append].append(laplace_result["FFT"][peak_index][time_value][0])

      elif laplace_result['n_buff'] == n_buff:
        # Consider only the specified buffer size (n_buff) if ht_analysis is True
        # Similar logic as the previous block for appending data
        feature_vector_dict['Damping'].append(laplace_result['Damping'][0])
        feature_vector_dict['LaPlace'].append(laplace_result['LaPlace'][0])
        feature_vector_dict['buff'].append(laplace_result['n_buff'])

        for peak_freqs, peak_index in zip(peaks_info[0], peaks_info[1]):
          key_to_append = peak_freqs
          if key_to_append in feature_vector_dict:
            feature_vector_dict[key_to_append].append(laplace_result["FFT"][peak_index][time_value][0])

  # Return the dictionary containing the created feature vectors
  return feature_vector_dict

def dict_for_som(analysis_vector,
                 ht_analysis=False, n_buff=None, VDF=None,
                 min_freq=None, max_freq=None, f=None, time_value=None):
  """
  This function prepares a dictionary containing feature vectors for Self-Organizing Map (SOM) training.

  Args:
      analysis_vector (list): A list of dictionaries containing analysis results for each buffer size.
          Each dictionary should have keys like 'Frequency', 'Damping', 'LaPlace', and 'FFT'.
      ht_analysis (bool, optional): A flag indicating whether to consider all buffer sizes (False)
                                      or only the specified buffer size (True) (default: False).
      n_buff (int, optional): The specific buffer size of interest (relevant if ht_analysis is True) (default: None).
      VDF (float, optional): The target resonant frequency (VDF) for peak identification (default: None).
      min_freq (float, optional): The minimum frequency in the considered frequency range (default: None).
      max_freq (float, optional): The maximum frequency in the considered frequency range (default: None).
      f (list, optional): A list containing the original frequencies used in the analysis (default: None).
      time_value (int, optional): The time step index to use from the 'FFT' data (e.g., for a specific time window) (default: None).

  Returns:
      dict: A dictionary containing feature vectors for SOM training.
          - Keys represent features like 'Damping', 'LaPlace', 'buff', and peak frequencies.
          - Values are lists containing the corresponding feature values for each data point.

  Raises:
      ValueError: If any of the required arguments (n_buff, VDF, min_freq, max_freq, f, time_value) are None.
  """

  # Check for missing required arguments
  if None in (n_buff, VDF, min_freq, max_freq, f, time_value):
      raise ValueError("Missing required arguments.")

  # Step 1: Extract peaks within the desired frequency range
  peaks_in_freq_range = extract_peaks_from_fr(analysis_vector=analysis_vector,
                                               ht_analysis=ht_analysis,
                                               n_buff=n_buff,
                                               VDF=VDF,
                                               min_freq=min_freq,
                                               max_freq=max_freq,
                                               f=f)
  # Assuming extract_peaks_from_fr returns a list of peak values

  # Step 2: Remove duplicate peaks and sort them numerically
  sorted_peaks = list(sorted(dict.fromkeys(peaks_in_freq_range)))

  # Step 3: Extract frequency and amplitude information for each peak
  peaks_info = extract_peaks_info(sorted_peaks, f)
  # Assuming extract_peaks_info takes a list of peaks and a list of frequencies (f) as input
  # and returns a data structure (e.g., tuple) containing peak frequencies and indices

  # Step 4: Create the feature vector dictionary using the extracted information
  feature_vector_dict = create_feature_vector(analysis_vector=analysis_vector,
                                               peaks_info=peaks_info,
                                               time_value=time_value,
                                               ht_analysis=ht_analysis,
                                               n_buff=n_buff)

  # Return the dictionary containing the feature vectors
  return feature_vector_dict

# %%
pre_p = Preprocessing (n_fft=4096)

#%%
# Parameters of the simulation
ht_length = [384, 672, 960]
vdf_frequencies = [700, 2000, 3300, 6400, 12000]
directory_path_laplace = Path (r'C:\Users\LEGION\PycharmProjects\pythonProject\venv\Laplace variations')
directory_path_gammaconst = Path (r'C:\Users\LEGION\PycharmProjects\pythonProject\venv\gammaconst')

# Run STFT and Peak Extraction
laplace_variations_analysis_vector = pre_p.analyze_spectrum (path_to_dataset=directory_path_laplace,
                                                             ht_length=ht_length,
                                                             vdf_frequencies=vdf_frequencies)

# %%
# Parameters of the pipeline

n_fft = 4096  # Window size of the n_fft
time_value = [60]  # time t at which peak picking is performed
n_buff = 672  # integration time T
VDF = 700
frequency_range = 1000 / 2  # FR of the analysis Freq. Range
# Specify the frequency range
min_freq = VDF - frequency_range
max_freq = VDF + frequency_range

# create_v = CreateVector (analysis_vector=laplace_variations_analysis_vector,
#                          n_buff=n_buff,
#                          VDF=VDF,
#                          min_freq=min_freq,
#                          max_freq=max_freq)


feature_vector_dataframe = pd.DataFrame (create_v.dict_for_som (laplace_variations_analysis_vector,
                                                       ht_analysis=False,
                                                       n_buff=n_buff,
                                                       VDF=VDF,
                                                       min_freq=min_freq,
                                                       max_freq=max_freq,
                                                       f=pre_p, time_value=time_value))

# The last three columns correspond to metadata for map calibration
data = feature_vector_dataframe[feature_vector_dataframe.columns[:-3]]

# data normalization
data = (data - np.mean (data, axis=0)) / np.std (data, axis=0)
data = data.values

# CREATING THE SOM - CALIBRATION

target1 = np.array (feature_vector_dataframe['Damping'].values)  # Re values
target2 = np.array (feature_vector_dataframe['LaPlace'].values)  # Gamma values
target3 = np.array (feature_vector_dataframe['buff'].values)  # Integration time / length of h(T)
labels = list (sorted (set (feature_vector_dataframe['Damping'].values)))

# U-matrix colors
cmapcolor = 'bone'
alpha = 1

# Initialization and training
n_neurons = 20
m_neurons = 20
nf = 'gaussian'
som = MiniSom (n_neurons, m_neurons, data.shape[1], sigma=1.5, learning_rate=.5,
               neighborhood_function=nf, random_seed=0, topology='rectangular')

som.pca_weights_init (data)
som.train (data, 1000, verbose=True)  # training

# create the U-matrix
u_matrix = som.distance_map ()
# ww = som.get_weights ()

# coordenates for the scatter plot
w_x, w_y = zip (*[som.winner (d) for d in data])
w_x = np.array (w_x)
w_y = np.array (w_y)

"""Calibration according to targets.
- The higher the value of Re, the darker the color of the data point.
- The smaller the value of gamma, the bigger the size of the data point.
"""

# Define the colormap range and colors
cmap = plt.cm.viridis_r
colors = cmap (np.linspace (0, 1, 256))

# Set the color for the Reference Cases (Re = 0)
colors[0] = [0.5, 0.5, 0.5, 1.0]  # Gray color for the 0 value

# Create a new colormap with the modified colors
cmap_modified = mcolors.ListedColormap (colors)


# %%
def plot_som(legend=True,
             umatrix_colorbar=False,
             ticks=True,
             title=True,
             save=True):
    figsize = (10, 11) if legend else (10, 10)
    bottom_adjust = 0.1 if legend else 0.05

    plt.figure (figsize=figsize)
    plt.subplots_adjust (left=0.05, right=0.95, bottom=bottom_adjust, top=0.95)
    p = plt.pcolor (som.distance_map ().T, cmap=cmapcolor, alpha=alpha)  # Store the plot object

    if umatrix_colorbar:
        plt.colorbar (p)  # Use the plot object for the colorbar

    unique_values = np.unique (target1)
    color_vals = cmap_modified (np.arange (len (unique_values)) / len (unique_values))  # Vectorize color assignment
    random_offsets_x = np.random.rand (len (target1)) - 0.5  # Pre-compute random offsets (x)
    random_offsets_y = np.random.rand (len (target1)) - 0.5  # Pre-compute random offsets (y)

    for idx, c in enumerate (unique_values):
        idx_target = target1 == c
        size_values = (target2[idx_target] - np.min (target2[idx_target])) / \
                      (np.max (target2[idx_target]) - np.min (target2[idx_target])) * 300

        plt.scatter (w_x[idx_target] + 0.5 + random_offsets_x[idx_target] * 0.5,
                     w_y[idx_target] + 0.5 + random_offsets_y[idx_target] * 0.8,
                     s=size_values, marker='o', color=color_vals[idx], label=f'{c}', alpha=1, edgecolors='black',
                     linewidths=1)

    T = 4 if n_buff == 384 else 7 if n_buff == 672 else 10

    ti = fr'$VDF$ = {VDF}Hz - $T$ = {T}ms - $FR$ = {min_freq}Hz - {max_freq}Hz - $t$ = {time_value[0]}ms'

    if title:
        # plt.title(f'{VDF}Hz - FR: {min_freq}Hz - {max_freq}Hz - t = {time_value[0]}ms', fontsize='xx-large')
        plt.title (f'{ti}', fontsize='xx-large')

    if legend:
        plt.legend (loc='upper center', bbox_to_anchor=(0.55, -0.03), ncol=6, fontsize='large')

    if ticks:
        plt.yticks (range (0, n_neurons + 1, 5), fontsize='x-large')
        plt.xticks (range (0, n_neurons + 1, 5), fontsize='x-large')
    else:
        plt.yticks ([])
        plt.xticks ([])

    plt.tight_layout ()

    if save:
        plt.savefig ('SOM.png', dpi=300)
    plt.show ()
    plt.close ()

plot_som ()

#%%

def plot_integration_time(legend=True,
                          umatrix_colorbar=False,
                          ticks=True,
                          title=True,
                          save=True,
                          shape=False):

    figsize = (10, 11) if legend else (10, 10)
    bottom_adjust = 0.1 if legend else 0.05

    plt.figure(figsize=figsize)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=bottom_adjust, top=0.95)
    p = plt.pcolor(som.distance_map().T, cmap=cmapcolor, alpha=alpha)  # Store the plot object

    if umatrix_colorbar:
        plt.colorbar(p)  # Use the plot object for the colorbar

    if shape:

        markers = ['o', '^', 'P']
        marker_labels = {
            markers[0]: '4ms',
            markers[1]: '7ms',
            markers[2]: '10ms'
        }
        unique_values = np.unique(target1)

        for idx, c in enumerate (unique_values):
            idx_target = target1 == c
            color = cmap_modified (idx / len (unique_values))

            # Define size based on target2 values (scale and normalize)
            size_values = (target2[idx_target] - np.min (target2[idx_target])) / \
                          (np.max (target2[idx_target]) - np.min (
                              target2[idx_target])) * 300  # Adjust max_size as needed

            # Set marker based on target3 value
            for i, t in enumerate (target3[idx_target]):
                if t == 384:
                    marker = markers[0]
                    label = f'{marker} 4ms'
                elif t == 672:
                    marker = markers[1]
                    label = f'{marker} 7ms'
                elif t == 960:
                    marker = markers[2]
                    label = f'{marker} 10ms'
                else:
                    marker = 'o'  # Default marker if target3 doesn't match any condition

                plt.scatter (w_x[idx_target][i] + .5 + (np.random.rand () - .5) * .5,
                             w_y[idx_target][i] + .5 + (np.random.rand () - .5) * .8,
                             s=size_values[i], marker=marker, color=color, label=marker_labels[marker], alpha=1,
                             edgecolors='black',
                             linewidths=1)

        T = 4 if n_buff == 384 else 7 if n_buff == 672 else 10

        ti = fr'$VDF$ = {VDF}Hz - $T$ = {T}ms - $FR$ = {min_freq}Hz - {max_freq}Hz - $t$ = {time_value[0]}ms'

    else:

        unique_values = np.unique(target3)
        colors = ['red', 'blue', 'yellow']


        for idx, c in enumerate(unique_values):
            idx_target = target3 == c

            if c == 384:
                label = f'4ms'
            elif c == 672:
                label = f'7ms'
            elif c == 960:
                label = f'10ms'
            else:
                marker = 'o'  # Default marker if target3 doesn't match any condition

            color = colors[idx]

            size_values = (target2[idx_target] - np.min(target2[idx_target])) / \
                          (np.max(target2[idx_target]) - np.min(target2[idx_target])) * 300

            plt.scatter(w_x[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .5,
                        w_y[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .8,
                        s=size_values, marker='o', color=color, label=label, alpha=1, edgecolors='black', linewidths=1)

            ti = fr'$VDF$ = {VDF}Hz - $T$ comparison - $FR$ = {min_freq}Hz - {max_freq}Hz - $t$ = {time_value[0]}ms'

    if title:
        plt.title(f'{ti}', fontsize='xx-large')

    if legend:
        if shape:
            # Create legend with custom labels
            handles, labels = plt.gca ().get_legend_handles_labels ()
            by_label = dict (zip (labels, handles))
            plt.legend (by_label.values (), by_label.keys (), loc='upper center', bbox_to_anchor=(0.55, -0.03), ncol=3,
                        fontsize='xx-large')
        else:
            plt.legend(loc='upper center', bbox_to_anchor=(0.55, -0.03), ncol=6, fontsize='xx-large')

    if ticks:
        plt.yticks(range(0, n_neurons + 1, 5), fontsize='x-large')
        plt.xticks(range(0, n_neurons + 1, 5), fontsize='x-large')
    else:
        plt.yticks([])
        plt.xticks([])

    plt.tight_layout()

    if save:
        plt.savefig('SOM.png', dpi=300)
    plt.show()
    plt.close()

plot_integration_time(shape=True, legend=True)
